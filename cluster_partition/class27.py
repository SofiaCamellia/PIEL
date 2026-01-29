# -*- coding: utf-8 -*-
"""
PARTITION INTEGRATED PIPELINE:
NetCDF Partition Filter -> 6 Base Models -> Spatial Stacker -> PINN
------------------------------------------------------------
1. Filters IAP data based on Partition ID (1-13) from NetCDF mask.
2. [NEW] Counts and prints data volume for all partitions (1-13).
3. Trains 6 Base Models (LGB, RF, CB, XGB, ERT, KNN).
4. Trains Spatial-Probabilistic Stacker.
5. Trains PINN using Stacker output + Physics constraints.
"""

import os
import time
import warnings
import pickle
import joblib
import numpy as np
import pandas as pd
import xarray as xr  # [修改] 用于读取 nc 文件

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# ML Models
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configs
warnings.filterwarnings("ignore")
RANDOM_SEED = 24
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =======================
# USER CONFIGURATION
# =======================
# 1. 聚类掩码文件路径 (NetCDF)
PARTITION_MASK_PATH = "ocean_pujulei_mask.nc"
# 2. 原始数据路径
DATA_PATH = "/home/bingxing2/home/scx7l1f/IAP_TSDO.npy"

# 3. [关键] 你想训练哪个分区？(填入 1-13 中的一个数字)
# 代码运行开始时会打印所有分区的计数，你可以根据那个结果回来修改这里
TARGET_PARTITION_ID = 1

# 4. 输出路径
REGION_NAME = f"partition_{TARGET_PARTITION_ID}"
BASE_PATH = f"/home/bingxing2/home/scx7l1f/rec/cluster_partition/{REGION_NAME}"

# 创建目录
os.makedirs(f"{BASE_PATH}/models", exist_ok=True)
os.makedirs(f"{BASE_PATH}/results", exist_ok=True)
os.makedirs(f"{BASE_PATH}/data", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 5


# =====================================================================================
# PART 1: Data Filtering (Partition Based) - [已修改]
# =====================================================================================
def get_data_by_partition():
    """
    1. 加载数据和 NetCDF 掩码。
    2. 对齐坐标 (0-360 -> -180~180)。
    3. 统计 1-13 每个类别的数量。
    4. 返回 TARGET_PARTITION_ID 的数据。
    """
    print(">>> Step 1: Loading data and matching partitions...")

    # 1. Load Raw Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    print(f"Loading raw data from {DATA_PATH}...")
    raw = np.load(DATA_PATH, allow_pickle=True)  # shape: (N, 8)

    # 2. Basic Filter (Year/Depth - 保持 arctic.py 逻辑)
    year = raw[:, 2]
    depth = raw[:, 4]
    mask_basic = (year >= 1980) & (depth <= 2000)
    data = raw[mask_basic]
    print(f"Base filtering (Year/Depth) remaining: {len(data)}")

    # 3. Load Partition Mask (NetCDF)
    if not os.path.exists(PARTITION_MASK_PATH):
        raise FileNotFoundError(f"Mask file not found: {PARTITION_MASK_PATH}")

    ds = xr.open_dataset(PARTITION_MASK_PATH)

    # 自动查找 partition 变量名
    possible_names = ["partition", "mask", "cluster", "region", "basin"]
    var_name = None
    for name in possible_names:
        if name in ds.variables:
            var_name = name
            break
    if var_name is None:
        raise KeyError(f"NetCDF文件中找不到 Partition 变量，请检查变量名: {list(ds.variables)}")

    grid_mask = ds[var_name]

    # 4. Coordinate Alignment & Extraction
    # 原始数据 lat, lon 分别是第 0, 1 列
    pts_lat = data[:, 0]
    pts_lon = data[:, 1]  # 0 ~ 360

    # [关键] 经度转换：0~360 -> -180~180
    # 逻辑：大于180的减去360
    pts_lon_conv = np.where(pts_lon > 180, pts_lon - 360, pts_lon)

    # 转换为 xarray DataArray 用于索引
    tgt_lat = xr.DataArray(pts_lat, dims="points")
    tgt_lon = xr.DataArray(pts_lon_conv, dims="points")

    print("Matching points to partition grid (Nearest Neighbor)...")
    # 使用 xarray 的 sel 方法进行最近邻查找
    extracted_ids = grid_mask.sel(
        lat=tgt_lat,
        lon=tgt_lon,
        method='nearest'
    ).values

    # 5. [统计] 打印每个分区的数据量
    print("\n" + "=" * 40)
    print(f"DATA COUNTS PER PARTITION (Total Points: {len(data)})")
    print(f"{'Partition ID':<15} | {'Count':<15}")
    print("-" * 35)

    # 统计 1-13 每个 ID
    valid_ids = extracted_ids[~np.isnan(extracted_ids)]  # 去除无效值
    unique, counts = np.unique(valid_ids, return_counts=True)
    count_dict = dict(zip(unique.astype(int), counts))

    for pid in range(1, 14):  # 1 到 13
        c = count_dict.get(pid, 0)
        mark = "<--- TARGET" if pid == TARGET_PARTITION_ID else ""
        print(f"{pid:<15} | {c:<15} {mark}")
    print("=" * 40 + "\n")

    # 6. Filtering Target
    # 筛选出 extracted_ids 等于我们目标 ID 的点
    # 处理 NaN
    extracted_ids = np.nan_to_num(extracted_ids, nan=-9999)

    partition_mask = (extracted_ids == TARGET_PARTITION_ID)
    final_data = data[partition_mask]

    print(f"Selected Partition: {TARGET_PARTITION_ID}")
    print(f"Final Training Data Count: {len(final_data)}")

    if len(final_data) == 0:
        raise ValueError(f"错误：目标分区 {TARGET_PARTITION_ID} 数据量为 0！请参考上面的统计表修改 TARGET_PARTITION_ID。")

    # 保存筛选后的数据
    save_path = f"{BASE_PATH}/data/partition_{TARGET_PARTITION_ID}_filtered.npy"
    np.save(save_path, final_data)

    return final_data


def year_based_split(X, y, years, train_ratio=0.8, val_ratio=0.1):
    uniq_years = np.array(sorted(np.unique(years).astype(int)))
    nY = len(uniq_years)
    n_trainY = max(1, int(round(train_ratio * nY)))
    n_valY = max(1, int(round(val_ratio * nY)))

    train_years = set(uniq_years[:n_trainY])
    val_years = set(uniq_years[n_trainY:n_trainY + n_valY])
    test_years = set(uniq_years[n_trainY + n_valY:])

    idx_train = np.array([yy in train_years for yy in years])
    idx_val = np.array([yy in val_years for yy in years])
    idx_test = np.array([yy in test_years for yy in years])

    return (X[idx_train], y[idx_train]), (X[idx_val], y[idx_val]), (X[idx_test], y[idx_test])


def make_time_ordered_es_split(X_fit, y_fit, frac=0.15):
    year = X_fit[:, 2].astype(int)
    month = X_fit[:, 3].astype(int)
    order = np.lexsort((month, year))
    n = len(X_fit)
    n_es = max(1, int(round(frac * n)))
    return X_fit[order[:-n_es]], y_fit[order[:-n_es]], X_fit[order[-n_es:]], y_fit[order[-n_es:]]


# =====================================================================================
# PART 2: 6 Base Models
# =====================================================================================
def get_model_factories():
    return {
        "lgb": lambda: lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=156, max_depth=12, min_child_samples=30,
                                         learning_rate=0.05, n_estimators=3500, subsample=0.8, colsample_bytree=0.8,
                                         reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, random_state=RANDOM_SEED,
                                         verbosity=-1),
        "rf": lambda: RandomForestRegressor(n_estimators=40, random_state=RANDOM_SEED, max_depth=20,
                                            min_samples_leaf=50, n_jobs=8, verbose=2, oob_score=False,
                                            max_features='sqrt'),
        "cb": lambda: CatBoostRegressor(iterations=3500, depth=12, learning_rate=0.05, loss_function='RMSE',
                                        eval_metric='RMSE', subsample=0.8, l2_leaf_reg=4, grow_policy='SymmetricTree',
                                        bootstrap_type='Bernoulli', random_seed=RANDOM_SEED, verbose=50,
                                        early_stopping_rounds=20, task_type='GPU', devices='0'),
        "xgb": lambda: XGBRegressor(n_estimators=3500, learning_rate=0.05, max_depth=10, subsample=0.8,
                                    colsample_bytree=0.8, objective="reg:squarederror", random_state=RANDOM_SEED,
                                    n_jobs=8, device='cuda', tree_method='hist', early_stopping_rounds=20, verbosity=1),
        "ert": lambda: ExtraTreesRegressor(n_estimators=40, max_depth=20, min_samples_leaf=50, max_features='sqrt',
                                           n_jobs=8, random_state=RANDOM_SEED, verbose=2),
        "knn": lambda: KNeighborsRegressor(n_neighbors=40, weights='distance', algorithm='auto', leaf_size=30, p=2,
                                           n_jobs=8)
    }


def run_base_models(X_tr, y_tr, X_val, X_test, scaler_knn, base_path):
    print("\n>>> Step 2: Training 6 Base Models (OOF + Full)...")
    models = get_model_factories()
    model_names = list(models.keys())
    n_tr = len(X_tr)

    oof_preds = {k: np.zeros(n_tr, dtype=np.float32) for k in model_names}
    years_tr = X_tr[:, 2].astype(int)
    gkf = GroupKFold(n_splits=K_FOLDS)
    X_tr_sc = scaler_knn.transform(X_tr)

    print(f"   Generating OOF ({K_FOLDS} folds)...")
    for fold, (idx_fit, idx_val) in enumerate(gkf.split(X_tr, y_tr, groups=years_tr), 1):
        print(f"     Fold {fold}/{K_FOLDS}...")
        X_f, y_f = X_tr[idx_fit], y_tr[idx_fit]
        X_v, y_v = X_tr[idx_val], y_tr[idx_val]
        X_inn, y_inn, X_es, y_es = make_time_ordered_es_split(X_f, y_f)

        # Train Loop
        m = models['lgb']();
        m.fit(X_inn, y_inn, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(50, verbose=False)]);
        oof_preds['lgb'][idx_val] = m.predict(X_v)
        m = models['xgb']();
        m.fit(X_inn, y_inn, eval_set=[(X_es, y_es)], verbose=False);
        oof_preds['xgb'][idx_val] = m.predict(X_v)
        m = models['cb']();
        m.fit(X_inn, y_inn, eval_set=(X_es, y_es), early_stopping_rounds=50, verbose=False);
        oof_preds['cb'][idx_val] = m.predict(X_v)
        m = models['rf']();
        m.fit(X_f, y_f);
        oof_preds['rf'][idx_val] = m.predict(X_v)
        m = models['ert']();
        m.fit(X_f, y_f);
        oof_preds['ert'][idx_val] = m.predict(X_v)
        m = models['knn']();
        m.fit(X_tr_sc[idx_fit], y_f);
        oof_preds['knn'][idx_val] = m.predict(X_tr_sc[idx_val])

    print("   Training Full Models for Inference & Saving...")
    val_preds = {};
    test_preds = {}
    X_val_sc = scaler_knn.transform(X_val)
    X_test_sc = scaler_knn.transform(X_test)
    X_inn, y_inn, X_es, y_es = make_time_ordered_es_split(X_tr, y_tr)

    model_dir = f"{base_path}/models"

    for name in model_names:
        print(f"     Fitting {name}...")
        model = models[name]()
        # Fit
        if name in ['lgb', 'xgb', 'cb']:
            model.fit(X_inn, y_inn, eval_set=[(X_es, y_es)], verbose=False) if name != 'lgb' else model.fit(X_inn,
                                                                                                            y_inn,
                                                                                                            eval_set=[(
                                                                                                                      X_es,
                                                                                                                      y_es)],
                                                                                                            callbacks=[
                                                                                                                lgb.early_stopping(
                                                                                                                    50,
                                                                                                                    verbose=False)])
        elif name == 'knn':
            model.fit(X_tr_sc, y_tr)
        else:
            model.fit(X_tr, y_tr)

        # Predict
        if name == 'knn':
            val_preds[name] = model.predict(X_val_sc);
            test_preds[name] = model.predict(X_test_sc)
        else:
            val_preds[name] = model.predict(X_val);
            test_preds[name] = model.predict(X_test)

        # Save
        if name == 'lgb':
            joblib.dump(model, f"{model_dir}/lgb_model.pkl")
        elif name == 'rf':
            joblib.dump(model, f"{model_dir}/rf_model.joblib")
        elif name == 'ert':
            joblib.dump(model, f"{model_dir}/ert_model.joblib")
        elif name == 'xgb':
            model.save_model(f"{model_dir}/xgb_model.json")
        elif name == 'cb':
            model.save_model(f"{model_dir}/cb_model.cbm")
        elif name == 'knn':
            with open(f"{model_dir}/knn_model.pkl", "wb") as f:
                pickle.dump(model, f)

    return oof_preds, val_preds, test_preds


# =====================================================================================
# PART 3: Spatial Stacker
# =====================================================================================
class SpatialProbabilisticStacker(nn.Module):
    def __init__(self, spatial_dim, n_models, quantiles=[0.05, 0.5, 0.95]):
        super().__init__()
        self.gating_net = nn.Sequential(
            nn.Linear(spatial_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU()
        )
        self.weight_head = nn.Linear(64, n_models)
        self.quantile_head = nn.Sequential(
            nn.Linear(64 + n_models, 32), nn.ReLU(), nn.Linear(32, len(quantiles))
        )

    def forward(self, spatial_x, base_preds):
        context = self.gating_net(spatial_x)
        weights = torch.softmax(self.weight_head(context), dim=1)
        point_pred = torch.sum(weights * base_preds, dim=1, keepdim=True)
        q_pred = self.quantile_head(torch.cat([context, base_preds], dim=1))
        return point_pred, weights, q_pred


def pinball_loss(preds, target, quantiles):
    loss = 0.0
    for i, q in enumerate(quantiles):
        error = target - preds[:, i:i + 1]
        loss += torch.mean(torch.max((q * error), ((q - 1) * error)))
    return loss


def run_stacker(X_sp_tr, Z_tr, y_tr, X_sp_val, Z_val, y_val, X_sp_test, Z_test, base_path):
    print("\n>>> Step 3: Training Spatial Stacker...")
    t_x_tr = torch.FloatTensor(X_sp_tr).to(DEVICE);
    t_z_tr = torch.FloatTensor(Z_tr).to(DEVICE);
    t_y_tr = torch.FloatTensor(y_tr).view(-1, 1).to(DEVICE)
    t_x_val = torch.FloatTensor(X_sp_val).to(DEVICE);
    t_z_val = torch.FloatTensor(Z_val).to(DEVICE);
    t_y_val = torch.FloatTensor(y_val).view(-1, 1).to(DEVICE)

    model = SpatialProbabilisticStacker(5, 6).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.005)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    best_loss = float('inf');
    best_state = None;
    batch_size = 65536

    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(t_x_tr), device=DEVICE)
        for i in range(0, len(t_x_tr), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pp, _, qp = model(t_x_tr[idx], t_z_tr[idx])
            loss = nn.MSELoss()(pp, t_y_tr[idx]) + 0.2 * pinball_loss(qp, t_y_tr[idx], [0.05, 0.5, 0.95])
            loss.backward();
            opt.step()

        model.eval()
        with torch.no_grad():
            vpp, _, vqp = model(t_x_val, t_z_val)
            vloss = nn.MSELoss()(vpp, t_y_val) + 0.2 * pinball_loss(vqp, t_y_val, [0.05, 0.5, 0.95])
        sch.step(vloss)
        if vloss < best_loss: best_loss = vloss; best_state = model.state_dict()
        if epoch % 20 == 0: print(f"   Stacker Epoch {epoch}: Val Loss {vloss.item():.4f}")

    model.load_state_dict(best_state)
    torch.save(best_state, f"{base_path}/models/best_stacker.pth")

    def infer(X, Z):
        model.eval();
        p_list = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                tx = torch.FloatTensor(X[i:i + batch_size]).to(DEVICE)
                tz = torch.FloatTensor(Z[i:i + batch_size]).to(DEVICE)
                p, _, _ = model(tx, tz);
                p_list.append(p.cpu().numpy())
        return np.vstack(p_list).flatten()

    return infer(X_sp_tr, Z_tr), infer(X_sp_val, Z_val), infer(X_sp_test, Z_test), model


# =====================================================================================
# PART 4: PINN
# =====================================================================================
class AvgPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 256), nn.BatchNorm1d(256), nn.Tanh(), nn.Linear(256, 128),
                                 nn.BatchNorm1d(128), nn.Tanh(), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, stack_pred, phys_feats): return self.net(torch.cat([stack_pred, phys_feats], dim=1)).squeeze()


def physical_constraint_loss(y_pred, features):
    lat, depth, temp, salt = features[:, 0], features[:, 1], features[:, 2], features[:, 3]
    pressure = depth
    T_k = temp + 273.15
    A1, A2, A3, A4 = -177.7888, 255.5907, 146.4813, -22.2040
    B1, B2, B3 = -0.037362, 0.016504, -0.0020564
    ln_DO = (-177.7888 + 255.5907 * (100 / T_k) + 146.4813 * torch.log(T_k / 100) - 22.2040 * (T_k / 100) + salt * (
                -0.037362 + 0.016504 * (T_k / 100) - 0.0020564 * ((T_k / 100) ** 2)))
    DO_sat = torch.exp(ln_DO) * 44.66 * (1 + (0.032 * pressure / 1000))
    max_allowed = torch.zeros_like(DO_sat)
    max_allowed[depth <= 50] = DO_sat[depth <= 50] * 1.5
    max_allowed[(depth > 50) & (depth <= 200)] = DO_sat[(depth > 50) & (depth <= 200)] * 1.15
    max_allowed[depth > 200] = DO_sat[depth > 200]
    penalty = torch.maximum(torch.zeros_like(y_pred), y_pred - max_allowed)
    neg_pen = torch.maximum(torch.zeros_like(y_pred), -y_pred)
    return torch.mean(penalty) + 50.0 * torch.mean(neg_pen)


class PINNDataset(Dataset):
    def __init__(self, stack_pred, X_raw, y, scaler):
        self.s = torch.tensor(stack_pred, dtype=torch.float32).unsqueeze(1)
        self.p_raw = torch.tensor(X_raw[:, [0, 4, 5, 6]], dtype=torch.float32)
        self.p_sc = torch.tensor(scaler.transform(X_raw[:, [0, 4, 5, 6]]), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return self.s[i], self.p_sc[i], self.p_raw[i], self.y[i]


def run_pinn(p_stack_tr, X_tr, y_tr, p_stack_val, X_val, y_val, p_stack_test, X_test, y_test, base_path):
    print("\n>>> Step 4: Training PINN...")
    pinn_scaler = StandardScaler();
    pinn_scaler.fit(X_tr[:, [0, 4, 5, 6]])
    bs = 65536
    tr_loader = DataLoader(PINNDataset(p_stack_tr, X_tr, y_tr, pinn_scaler), batch_size=bs, shuffle=True)
    val_loader = DataLoader(PINNDataset(p_stack_val, X_val, y_val, pinn_scaler), batch_size=bs)
    te_loader = DataLoader(PINNDataset(p_stack_test, X_test, y_test, pinn_scaler), batch_size=bs)

    model = AvgPINN().to(DEVICE);
    opt = optim.Adam(model.parameters(), lr=0.005)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5);
    scaler = GradScaler()
    w_phys, w_teach, best_loss = 0.5, 0.4, float('inf')

    for epoch in range(50):
        model.train()
        if epoch > 0 and epoch % 5 == 0: w_phys = min(1.0, w_phys + 0.1); w_teach = max(0.1, w_teach - 0.05)
        for s, p_sc, p_raw, y in tr_loader:
            s, p_sc, p_raw, y = s.to(DEVICE), p_sc.to(DEVICE), p_raw.to(DEVICE), y.to(DEVICE);
            opt.zero_grad()
            with autocast():
                pred = model(s, p_sc)
                loss = nn.MSELoss()(pred, y.squeeze()) + w_phys * physical_constraint_loss(pred,
                                                                                           p_raw) + w_teach * nn.MSELoss()(
                    pred, s.squeeze())
            scaler.scale(loss).backward();
            scaler.step(opt);
            scaler.update()

        model.eval();
        v_loss = 0
        with torch.no_grad():
            for s, p_sc, _, y in val_loader:
                s, p_sc, y = s.to(DEVICE), p_sc.to(DEVICE), y.to(DEVICE)
                with autocast(): v_loss += nn.MSELoss()(model(s, p_sc), y.squeeze()).item()
        v_loss /= len(val_loader);
        sch.step(v_loss)
        if epoch % 5 == 0: print(f"   PINN Epoch {epoch}: Val MSE {v_loss:.4f} (Phys W: {w_phys:.1f})")
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), f"{base_path}/models/best_pinn.pth")  # <--- 修改这里
    model.load_state_dict(torch.load(f"{base_path}/models/best_pinn.pth"));
    model.eval();
    preds = []
    with torch.no_grad():
        for s, p_sc, _, _ in te_loader: preds.append(model(s.to(DEVICE), p_sc.to(DEVICE)).float().cpu().numpy())
    return np.concatenate(preds)


def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mb = np.mean(y_pred - y_true)
    print(f"{name} | RMSE: {rmse:.4f} | R2: {r2:.4f} | MAE: {mae:.4f} | MB: {mb:.4f}")
    print(f"预测值范围: {y_pred.min():.2f} ~ {y_pred.max():.2f}")


# =====================================================================================
# Main
# =====================================================================================
# ... (前面的 import, Configs, 以及 run_base_models 等函数定义保持不变) ...

def main():
    t0_global = time.time()
    print("===== PARTITION INTEGRATED PIPELINE (BATCH LOOP) STARTED =====")

    # -------------------------------------------------------
    # 1. 预加载数据 (在循环外面只做一次，极大提高速度)
    # -------------------------------------------------------
    print(">>> [全局] 读取原始数据和掩码文件...")
    if not os.path.exists(DATA_PATH): raise FileNotFoundError(DATA_PATH)

    # 加载 .npy
    raw = np.load(DATA_PATH, allow_pickle=True)
    # 基础筛选 (年份/深度)
    year = raw[:, 2];
    depth = raw[:, 4]
    raw = raw[(year >= 1980) & (depth <= 2000)]
    print(f"全局数据加载完成，共 {len(raw)} 条")

    # 加载 .nc 掩码
    if not os.path.exists(PARTITION_MASK_PATH): raise FileNotFoundError(PARTITION_MASK_PATH)
    ds = xr.open_dataset(PARTITION_MASK_PATH)
    vname = [n for n in ["partition", "mask", "cluster"] if n in ds][0]
    grid_mask = ds[vname]  # 保持在内存里

    # -------------------------------------------------------
    # 2. 开始循环 (1 到 13)
    # -------------------------------------------------------
    # target_ids = [12] # 如果只想跑一个，就写 [12]
    target_ids = range(2, 8)  # 跑 1 到 13

    for partition_id in target_ids:
        print(f"\n{'>' * 20} 开始处理 Partition ID: {partition_id} {'<' * 20}")

        try:
            # === A. 动态设置保存路径 ===
            CURRENT_REGION_NAME = f"partition_{partition_id}"
            # 注意：这里定义当前循环的 BASE_PATH
            CURRENT_BASE_PATH = f"/home/bingxing2/home/scx7l1f/rec/mask_partition/{CURRENT_REGION_NAME}"

            # 创建目录
            os.makedirs(f"{CURRENT_BASE_PATH}/models", exist_ok=True)
            os.makedirs(f"{CURRENT_BASE_PATH}/results", exist_ok=True)
            os.makedirs(f"{CURRENT_BASE_PATH}/data", exist_ok=True)

            # === B. 筛选数据 (使用之前写好的逻辑) ===
            # 这里直接把筛选逻辑写进来，或者调用函数都行
            # 原始数据 lat, lon
            pts_lat = raw[:, 0]
            pts_lon = raw[:, 1]
            pts_lon_conv = np.where(pts_lon > 180, pts_lon - 360, pts_lon)

            tgt_lat = xr.DataArray(pts_lat, dims="points")
            tgt_lon = xr.DataArray(pts_lon_conv, dims="points")

            # 匹配 ID
            extracted_ids = grid_mask.sel(lat=tgt_lat, lon=tgt_lon, method='nearest').values
            extracted_ids = np.nan_to_num(extracted_ids, nan=-9999)

            # 拿到当前 ID 的数据
            data = raw[extracted_ids == partition_id]

            # === C. 检查数据量 (太少就跳过) ===
            if len(data) < 100:
                print(f"⚠️ Partition {partition_id} 数据量仅 {len(data)} 条，跳过训练。")
                continue  # 进入下一个循环

            print(f"✅ 数据筛选完成: {len(data)} 条")
            # 备份一下数据
            np.save(f"{CURRENT_BASE_PATH}/data/partition_{partition_id}_filtered.npy", data)

            # === D. 准备训练集 ===
            X = data[:, :-1].astype(np.float32)
            y = data[:, -1].astype(np.float32)
            years = X[:, 2].astype(int)

            (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = year_based_split(X, y, years)

            if len(X_tr) < 50:
                print("⚠️ 训练集过小，跳过。");
                continue

            # === E. 训练模型 (注意传入 CURRENT_BASE_PATH) ===

            # 1. Base Models
            scaler_knn = StandardScaler();
            scaler_knn.fit(X_tr)

            # !!!!!!!! 重要 !!!!!!!!
            # 请确保你的 run_base_models 函数接受最后一个 path 参数
            # 如果没改函数定义，这里会报错。
            # 如果不想改函数定义，可以在这里临时修改全局变量 (不推荐但能用)
            # global BASE_PATH; BASE_PATH = CURRENT_BASE_PATH


            oof_preds, val_preds, test_preds = run_base_models(
                X_tr, y_tr, X_val, X_test, scaler_knn,
                base_path=CURRENT_BASE_PATH  # <--- 这里把当前分区的路径传进去
            )
            # 保存中间结果
            np.savez(f"{CURRENT_BASE_PATH}/results/oof_base.npz", **oof_preds)
            np.savez(f"{CURRENT_BASE_PATH}/results/test_base.npz", **test_preds)

            # 2. Stacker
            SP_IDXS = [0, 1, 4, 5, 6]
            sp_scaler = StandardScaler()
            X_sp_tr = sp_scaler.fit_transform(X_tr[:, SP_IDXS])
            X_sp_val = sp_scaler.transform(X_val[:, SP_IDXS])
            X_sp_test = sp_scaler.transform(X_test[:, SP_IDXS])

            Z_tr = np.column_stack([oof_preds[k] for k in oof_preds])
            Z_val = np.column_stack([val_preds[k] for k in val_preds])
            Z_test = np.column_stack([test_preds[k] for k in test_preds])

            p_stack_tr, p_stack_val, p_stack_test, _ = run_stacker(
                X_sp_tr, Z_tr, y_tr, X_sp_val, Z_val, y_val, X_sp_test, Z_test, base_path=CURRENT_BASE_PATH)

            # 3. PINN
            p_pinn_test = run_pinn(
                p_stack_tr, X_tr, y_tr, p_stack_val, X_val, y_val, p_stack_test, X_test, y_test, base_path=CURRENT_BASE_PATH)


            # === F. 结果评估 ===
            print(f"\n[Partition {partition_id}] 结果:")
            evaluate(y_test, p_stack_test, "Stacker")
            evaluate(y_test, p_pinn_test, "PINN")

            np.savez(f"{CURRENT_BASE_PATH}/results/final_predictions.npz",
                     stacker=p_stack_test, pinn=p_pinn_test, y_true=y_test)

        except Exception as e:
            print(f"❌ Partition {partition_id} 发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue  # 出错后继续跑下一个 ID，不要停

    print(f"===== 所有分区处理完毕，总耗时 {(time.time() - t0_global) / 60:.1f} mins =====")


if __name__ == "__main__":
    main()

