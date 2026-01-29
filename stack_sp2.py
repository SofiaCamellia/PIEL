# -*- coding: utf-8 -*-
# 六个基模型集成，后三个完整版，前三个在sp1
"""
GLOBAL | Incremental Stacking | Reuse LGB/RF/CB | Train XGB/ERT/KNN
Includes:
1. Loading existing models/OOF from stack_sp1
2. Training only NEW models (XGB, ERT, KNN)
3. Spatial-Probabilistic Stacking with 6 models
"""

import os
import time
import warnings
import pickle
import joblib
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# ML Models
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor  # Added XGBoost
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
RANDOM_SEED = 24
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =======================
# Paths & Configs
# =======================
# 请确保这里指向 stack_sp1 运行结果的目录，因为我们要读取那里的旧模型
DATA_PATH = "/home/bingxing2/home/scx7l1f/IAP_TSDO.npy"
BASE_PATH = "/home/bingxing2/home/scx7l1f/rec/BMA/SPATIAL_STACKING1"
BASE_PATH2 = "/home/bingxing2/home/scx7l1f/rec/BMA/SPATIAL_STACKING_6models"
# 如果你想把结果存到新目录，可以修改 BASE_PATH，但读取旧模型时需要指定 OLD_MODEL_PATH
# 这里假设我们在同一个目录下继续工作

# === [新增修复] 自动创建新路径的文件夹 ===
os.makedirs(f"{BASE_PATH2}/models", exist_ok=True)
os.makedirs(f"{BASE_PATH2}/data", exist_ok=True)
os.makedirs(f"{BASE_PATH2}/results", exist_ok=True)

K_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stacker Hyperparams
META_EPOCHS = 100
META_LR = 0.005
META_BATCH_SIZE = 65536
QUANTILES = [0.05, 0.5, 0.95]
PINBALL_LAMBDA = 0.2


# =====================================================================================
# Part 1) Data Utils
# =====================================================================================
def global_filter(data):
    year = data[:, 2]
    depth = data[:, 4]
    mask = (year >= 1980) & (depth <= 2000)
    return data[mask]


# ==========================================
# [新增] 从 sp1 找回 evaluate 函数
# ==========================================
def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    mb = float(np.mean(y_pred - y_true))  # Mean Bias

    print(
        f"  {name} | RMSE={rmse:.4f} | R2={r2:.4f} | MAE={mae:.4f} | MB={mb:.4f} | range=[{y_pred.min():.3f}, {y_pred.max():.3f}]")

    return {"rmse": rmse, "r2": r2, "mae": mae, "mb": mb}

def year_based_split(X, y, years, train_ratio=0.8, val_ratio=0.1):
    uniq_years = np.array(sorted(np.unique(years).astype(int)))
    nY = len(uniq_years)
    n_trainY = max(1, int(round(train_ratio * nY)))
    n_valY = max(1, int(round(val_ratio * nY)))
    if n_trainY + n_valY >= nY:
        n_trainY = max(1, nY - 2)
        n_valY = 1

    train_years = set(uniq_years[:n_trainY].tolist())
    val_years = set(uniq_years[n_trainY:n_trainY + n_valY].tolist())
    test_years = set(uniq_years[n_trainY + n_valY:].tolist())

    idx_train = np.array([yy in train_years for yy in years], dtype=bool)
    idx_val = np.array([yy in val_years for yy in years], dtype=bool)
    idx_test = np.array([yy in test_years for yy in years], dtype=bool)

    return (X[idx_train], y[idx_train], sorted(train_years)), \
           (X[idx_val], y[idx_val], sorted(val_years)), \
           (X[idx_test], y[idx_test], sorted(test_years))


def make_time_ordered_es_split(X_fit, y_fit, frac=0.15):
    year = X_fit[:, 2].astype(int)
    month = X_fit[:, 3].astype(int)
    order = np.lexsort((month, year))
    n = len(X_fit)
    n_es = max(1, int(round(frac * n)))
    idx_es = order[-n_es:]
    idx_tr = order[:-n_es]
    return X_fit[idx_tr], y_fit[idx_tr], X_fit[idx_es], y_fit[idx_es]


# =====================================================================================
# Part 2) Model Definitions (Old + New)
# =====================================================================================
# --- OLD MODELS (Saved in stack_sp1) ---
# We define them just in case, but we prefer loading them.
def make_lgbm(): return lgb.LGBMRegressor(random_state=RANDOM_SEED)  # Placeholder


def make_rf(): return RandomForestRegressor(random_state=RANDOM_SEED)  # Placeholder


def make_cb(): return CatBoostRegressor(random_state=RANDOM_SEED)  # Placeholder


# --- NEW MODELS (Using User's Params) ---
def make_xgb():
    return XGBRegressor(
        n_estimators=3500,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=8,
        # GPU params
        device='cuda',
        tree_method='hist',
        early_stopping_rounds=20,
        verbosity=1 # [修改] 开启日志 (0=silent, 1=warning, 2=info)
    )
# depth=10 结果不如LGBM和CB，应该设12的

def make_ert():
    # ExtraTrees (Extremely Randomized Trees)
    return ExtraTreesRegressor(
        n_estimators=40,
        max_depth=20,
        min_samples_leaf=50,
        max_features='sqrt',
        n_jobs=8,
        random_state=RANDOM_SEED,
        verbose=2
    )


def make_knn():
    # KNN需要标准化数据
    return KNeighborsRegressor(
        n_neighbors=40,
        weights='distance',
        algorithm='auto',
        leaf_size=30,
        p=2,
        n_jobs=8
    )


# =====================================================================================
# Part 3) Smart OOF Generation (Load Old + Calc New)
# =====================================================================================
def get_hybrid_oof(X_train, y_train, years_train, K=5):
    n = len(X_train)

    final_oof_path = f"{BASE_PATH2}/results/oof_base_preds_ALL6.npz"
    if os.path.exists(final_oof_path):
        print(f"\n[Smart Skip] Found complete OOF file at {final_oof_path}")
        print("             -> Loading directly and skipping OOF generation.")
        data = np.load(final_oof_path)
        # 还原回字典格式
        return {k: data[k] for k in data.files}

    oof_path = f"{BASE_PATH}/results/oof_base_preds.npz"  # Old OOF path

    # 1. 尝试加载旧的 OOF
    old_oof = {}
    if os.path.exists(oof_path):
        print(f"  -> Found existing OOF file at {oof_path}")
        tmp = np.load(oof_path)
        # 只保留旧模型
        for key in ['lgb', 'rf', 'cb']:
            if key in tmp:
                old_oof[key] = tmp[key]
                print(f"     Loaded OOF for: {key}")

    # 2. 准备新的 OOF 容器
    new_models = ['xgb', 'ert', 'knn']
    final_oof = {k: np.zeros(n, dtype=np.float32) for k in new_models}

    # 将旧的复制进去
    for k, v in old_oof.items():
        final_oof[k] = v

    # 3. 计算新模型的 OOF
    print(f"\nComputing OOF for NEW models: {new_models} ...")
    gkf = GroupKFold(n_splits=K)

    for fold, (idx_fit, idx_oof) in enumerate(gkf.split(X_train, y_train, groups=years_train), 1):
        print(f"  [OOF] Fold {fold}/{K} for New Models...")

        X_fit, y_fit = X_train[idx_fit], y_train[idx_fit]
        X_oof = X_train[idx_oof]
        X_inner, y_inner, X_es, y_es = make_time_ordered_es_split(X_fit, y_fit, frac=0.15)

        # --- XGBoost ---
        xgb = make_xgb()
        # 注意: XGB fit 需要 eval_set 才能触发 early_stopping
        xgb.fit(X_inner, y_inner, eval_set=[(X_es, y_es)], verbose=100)
        final_oof["xgb"][idx_oof] = xgb.predict(X_oof).astype(np.float32)

        # --- ERT ---
        ert = make_ert()
        ert.fit(X_fit, y_fit)
        final_oof["ert"][idx_oof] = ert.predict(X_oof).astype(np.float32)

        # --- KNN (Scaling required inside fold) ---
        # --- KNN (修改处: 增加计时打印) ---
        print("  > Fitting KNN...")
        t_knn = time.time()  # 开始计时
        scaler = StandardScaler()
        X_fit_sc = scaler.fit_transform(X_fit)
        X_oof_sc = scaler.transform(X_oof)
        knn = make_knn()
        knn.fit(X_fit_sc, y_fit)

        # 打印耗时
        print(f"    (KNN index built in {time.time() - t_knn:.2f}s)")

        final_oof["knn"][idx_oof] = knn.predict(X_oof_sc).astype(np.float32)

        del X_fit, X_oof, X_fit_sc, X_oof_sc

    print("  New OOF Generation Complete.")

    # 保存所有合并后的 OOF，防止下次再跑
    new_oof_path = f"{BASE_PATH2}/results/oof_base_preds_ALL6.npz"
    np.savez(new_oof_path, **final_oof)
    print(f"  Saved combined OOF to {new_oof_path}")

    return final_oof


# =====================================================================================
# Part 4) Smart Full Model Loading/Training
# =====================================================================================
def get_hybrid_full_models(X_train, y_train):
    """
    Loads LGB, RF, CB from disk.
    Trains XGB, ERT, KNN from scratch.
    Returns all 6 models.
    """
    # Paths for OLD models
    lgb_path = f"{BASE_PATH}/models/lgb_model.pkl"
    rf_path = f"{BASE_PATH}/models/rf_model.joblib"
    cb_path = f"{BASE_PATH}/models/catboost_model.cbm"

    # Paths for NEW models
    xgb_path = f"{BASE_PATH2}/models/xgb_model.json"
    ert_path = f"{BASE_PATH2}/models/ert_model.joblib"
    knn_path = f"{BASE_PATH2}/models/knn_bundle.pkl"

    print("\n--- Model Loading Phase ---")

    # 1. Load Old Models
    if os.path.exists(lgb_path) and os.path.exists(rf_path) and os.path.exists(cb_path):
        print("  -> Loading existing LGB, RF, CB...")
        with open(lgb_path, "rb") as f:
            lgbm = pickle.load(f)
        rf = joblib.load(rf_path)
        cb = CatBoostRegressor()
        cb.load_model(cb_path)
    else:
        raise FileNotFoundError(
            "Could not find old models (LGB/RF/CB). Please run stack_sp1.py first or allow retraining.")

    # 2. Train New Models
        # === [修改点 2] 检查新模型是否已存在 ===
    if os.path.exists(xgb_path) and os.path.exists(ert_path) and os.path.exists(knn_path):
        print("  -> Found existing NEW models (XGB, ERT, KNN). Loading directly...")

        # Load XGB
        xgb = XGBRegressor()
        xgb.load_model(xgb_path)

        # Load ERT
        ert = joblib.load(ert_path)

        # Load KNN
        with open(knn_path, "rb") as f: (knn, scaler_knn) = pickle.load(f)

        print("  -> New models loaded. Skipping training.")
        return lgbm, rf, cb, xgb, ert, (knn, scaler_knn)

    print("  -> Training NEW models (XGB, ERT, KNN) on full data...")
    X_inner, y_inner, X_es, y_es = make_time_ordered_es_split(X_train, y_train, frac=0.15)

    # XGB
    print("     Training XGBoost...")
    xgb = make_xgb()
    xgb.fit(X_inner, y_inner, eval_set=[(X_es, y_es)], verbose=100)
    xgb.save_model(xgb_path)

    # ERT
    print("     Training ExtraTrees...")
    ert = make_ert()
    ert.fit(X_train, y_train)
    joblib.dump(ert, ert_path)

    # KNN
    print("     Training KNN (with scaler)...")
    # --- KNN (修改处: 增加计时) ---
    print("\n     Training KNN (Full)...")
    t_start = time.time()

    scaler_knn = StandardScaler()
    X_train_sc = scaler_knn.fit_transform(X_train)
    knn = make_knn()
    knn.fit(X_train_sc, y_train)
    print(f"     -> KNN Index Built in {time.time() - t_start:.2f}s")

    with open(knn_path, "wb") as f:
        pickle.dump((knn, scaler_knn), f)

    print("  -> All models ready.")
    return lgbm, rf, cb, xgb, ert, (knn, scaler_knn)


def predict_all(models, X):
    lgbm, rf, cb, xgb, ert, (knn, scaler_knn) = models
    X_sc = scaler_knn.transform(X)  # KNN specific

    return {
        "lgb": lgbm.predict(X).astype(np.float32),
        "rf": rf.predict(X).astype(np.float32),
        "cb": cb.predict(X).astype(np.float32),
        "xgb": xgb.predict(X).astype(np.float32),
        "ert": ert.predict(X).astype(np.float32),
        "knn": knn.predict(X_sc).astype(np.float32),
    }


# =====================================================================================
# Part 5) Stacking (Same as before)
# =====================================================================================
class SpatialProbabilisticStacker(nn.Module):
    def __init__(self, spatial_dim, n_models, quantiles):
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
        context_feats = self.gating_net(spatial_x)
        weights = torch.softmax(self.weight_head(context_feats), dim=1)
        point_pred = torch.sum(weights * base_preds, dim=1, keepdim=True)
        quantile_preds = self.quantile_head(torch.cat([context_feats, base_preds], dim=1))
        return point_pred, weights, quantile_preds


def pinball_loss(preds, target, quantiles):
    loss = 0.0
    for i, q in enumerate(quantiles):
        error = target - preds[:, i:i + 1]
        loss += torch.mean(torch.max((q * error), ((q - 1) * error)))
    return loss


def train_stacker_fast(X_spatial_tr, Base_preds_tr, y_tr, X_spatial_val, Base_preds_val, y_val):
    spatial_dim = X_spatial_tr.shape[1]
    n_models = Base_preds_tr.shape[1]

    X_sp_tr_dev = torch.FloatTensor(X_spatial_tr).to(DEVICE)
    Base_p_tr_dev = torch.FloatTensor(Base_preds_tr).to(DEVICE)
    y_tr_dev = torch.FloatTensor(y_tr).view(-1, 1).to(DEVICE)
    X_sp_val_dev = torch.FloatTensor(X_spatial_val).to(DEVICE)
    Base_p_val_dev = torch.FloatTensor(Base_preds_val).to(DEVICE)
    y_val_dev = torch.FloatTensor(y_val).view(-1, 1).to(DEVICE)

    model = SpatialProbabilisticStacker(spatial_dim, n_models, QUANTILES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=META_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_loss = float('inf')
    best_state = None
    patience = 0
    n_samples = X_sp_tr_dev.shape[0]
    n_batches = int(np.ceil(n_samples / META_BATCH_SIZE))

    for epoch in range(META_EPOCHS):
        model.train()
        perm = torch.randperm(n_samples, device=DEVICE)
        for i in range(n_batches):
            idx = perm[i * META_BATCH_SIZE: (i + 1) * META_BATCH_SIZE]
            optimizer.zero_grad()
            pp, _, qp = model(X_sp_tr_dev[idx], Base_p_tr_dev[idx])
            loss = nn.MSELoss()(pp, y_tr_dev[idx]) + PINBALL_LAMBDA * pinball_loss(qp, y_tr_dev[idx], QUANTILES)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vpp, _, vqp = model(X_sp_val_dev, Base_p_val_dev)
            vloss = (nn.MSELoss()(vpp, y_val_dev) + PINBALL_LAMBDA * pinball_loss(vqp, y_val_dev, QUANTILES)).item()

        scheduler.step(vloss)
        if vloss < best_loss:
            best_loss = vloss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= 15: break

    model.load_state_dict(best_state)
    return model


def inference_stacker(model, X_spatial, Base_preds):
    model.eval()
    batch_size = META_BATCH_SIZE * 4
    n_samples = len(X_spatial)
    all_p, all_w, all_q = [], [], []
    with torch.no_grad():
        for i in range(int(np.ceil(n_samples / batch_size))):
            s = i * batch_size
            e = min((i + 1) * batch_size, n_samples)
            sp = torch.FloatTensor(X_spatial[s:e]).to(DEVICE)
            bp = torch.FloatTensor(Base_preds[s:e]).to(DEVICE)
            p, w, q = model(sp, bp)
            all_p.append(p.cpu().numpy())
            all_w.append(w.cpu().numpy())
            all_q.append(q.cpu().numpy())
    return np.vstack(all_p).flatten(), np.vstack(all_w), np.vstack(all_q)


# =====================================================================================
# Main
# =====================================================================================
MODEL_SETS = {
    # "ALL_6": ["lgb", "rf", "cb", "xgb", "ert", "knn"],  # Main focus
    # "New_3": ["xgb", "ert", "knn"],
    # "Old_3": ["lgb", "rf", "cb"]
    "LGB": ["lgb"],
    "RF": ["rf"],
    "CB": ["cb"],
    "XGB": ["xgb"],
    "ERT": ["ert"],
    "KNN": ["knn"],

    # === 组合实验 ===
    "Tree_5": ["lgb", "rf", "cb", "xgb", "ert"],  # 5个树模型 (无KNN)
    "ALL_6": ["lgb", "rf", "cb", "xgb", "ert", "knn"],  # 6个全模型

    # === 对照组 (可选) ===
    "Old_3": ["lgb", "rf", "cb"]
}


def main():
    t0 = time.time()
    print("\n===== INCREMENTAL SPATIAL STACKING (6 Models) =====")

    # 1. Load Data
    raw = np.load(DATA_PATH, allow_pickle=True)
    data = global_filter(raw)
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32)
    years = X[:, 2].astype(int)

    (X_tr, y_tr, tr_yrs), (X_val, y_val, val_yrs), (X_test, y_test, test_yrs) = \
        year_based_split(X, y, years)

    # 2. Hybrid OOF (Load Old + Calc New)
    oof = get_hybrid_oof(X_tr, y_tr, X_tr[:, 2].astype(int), K=K_FOLDS)

    # 3. Hybrid Full Models (Load Old + Train New)
    models = get_hybrid_full_models(X_tr, y_tr)

    # 4. Predict Val/Test
    val_preds = predict_all(models, X_val)
    test_preds = predict_all(models, X_test)

    # 5. Spatial Features
    scaler = StandardScaler()
    X_sp_tr = scaler.fit_transform(X_tr[:, [0, 1, 4, 5, 6]])
    X_sp_val = scaler.transform(X_val[:, [0, 1, 4, 5, 6]])
    X_sp_test = scaler.transform(X_test[:, [0, 1, 4, 5, 6]])

    # 6. Ablation
    # all_results = {}
    # for name, keys in MODEL_SETS.items():
    #     print(f"\n>>> Experiment: {name} {keys}")
    #
    #     # Stack Predictions
    #     Z_tr = np.column_stack([oof[k] for k in keys]).astype(np.float32)
    #     Z_val = np.column_stack([val_preds[k] for k in keys]).astype(np.float32)
    #     Z_test = np.column_stack([test_preds[k] for k in keys]).astype(np.float32)
    #
    #     # Train Stacker
    #     stacker = train_stacker_fast(X_sp_tr, Z_tr, y_tr, X_sp_val, Z_val, y_val)
    #
    #     # === [建议补充] 保存 Stacker 模型参数 ===
    #     if name == "ALL_6":
    #         torch.save(stacker.state_dict(), f"{BASE_PATH2}/models/Stacker_ALL_6.pth")
    #         # 顺便保存一下空间特征的归一化器，预测新数据必须用它
    #         with open(f"{BASE_PATH2}/models/spatial_scaler_stacker.pkl", "wb") as f:
    #             pickle.dump(scaler, f)
    #     # ======================================
    #
    #     # Inference
    #     p_test, w_test, q_test = inference_stacker(stacker, X_sp_test, Z_test)
    #
    #     # Metrics
    #     rmse = np.sqrt(mean_squared_error(y_test, p_test))
    #     r2 = r2_score(y_test, p_test)
    #     print(f"  TEST | RMSE={rmse:.4f} | R2={r2:.4f}")
    #     print(f"  Avg Weights: {np.mean(w_test, axis=0).round(3)}")
    #
    #     # Save
    #     if name == "ALL_6":
    #         np.savez(f"{BASE_PATH2}/results/{name}_preds.npz",
    #                  test_pred=p_test, test_weights=w_test, test_quantiles=q_test)
    # 6. Ablation (完全替换原来的 for 循环)
    all_results = {}  # 用于存储最终汇总

    for name, keys in MODEL_SETS.items():
        print(f"\n>>> Experiment: {name} {keys}")

        # Stack Predictions
        Z_tr = np.column_stack([oof[k] for k in keys]).astype(np.float32)
        Z_val = np.column_stack([val_preds[k] for k in keys]).astype(np.float32)
        Z_test = np.column_stack([test_preds[k] for k in keys]).astype(np.float32)

        # Train Stacker
        stacker = train_stacker_fast(X_sp_tr, Z_tr, y_tr, X_sp_val, Z_val, y_val)

        # 保存模型参数 (针对 ALL_6)
        if name == "ALL_6":
            torch.save(stacker.state_dict(), f"{BASE_PATH2}/models/Stacker_ALL_6.pth")
            # 顺便保存一下空间特征的归一化器
            with open(f"{BASE_PATH2}/models/spatial_scaler_stacker.pkl", "wb") as f:
                pickle.dump(scaler, f)

        # Inference (验证集 + 测试集)
        # [修复] 增加验证集推理，用于后续评估和保存
        p_val, w_val, q_val = inference_stacker(stacker, X_sp_val, Z_val)
        p_test, w_test, q_test = inference_stacker(stacker, X_sp_test, Z_test)

        # Metrics (使用 evaluate 函数)
        val_res = evaluate(y_val, p_val, "VAL ")
        test_res = evaluate(y_test, p_test, "TEST")

        # [修复] 计算不确定性 (95%分位数 - 5%分位数)
        avg_uncertainty = np.mean(q_test[:, 2] - q_test[:, 0])
        print(f"  Avg Weights: {np.mean(w_test, axis=0).round(3)}")
        print(f"  Avg Uncertainty Width: {avg_uncertainty:.4f}")

        # Save Results (保存所有需要的字段，包括 val_pred)
        if name == "ALL_6":
            np.savez(f"{BASE_PATH2}/results/{name}_preds.npz",
                     test_pred=p_test, test_weights=w_test, test_quantiles=q_test,
                     val_pred=p_val, val_weights=w_val)  # <--- [关键修复]
            print(f"  Saved {name}_preds.npz (with val_pred)")

        # 收集结果用于汇总
        all_results[name] = {
            "test": test_res,
            "val": val_res,
            "uncertainty": avg_uncertainty
        }

    print(f"\nDone. Time: {(time.time() - t0) / 60:.1f} min")
    # [新增] 最终排行榜 (Summary)
    print("\n===== Summary (Sorted by TEST RMSE) =====")
    # 按 TEST RMSE 排序
    ranking = sorted(all_results.items(), key=lambda x: x[1]["test"]["rmse"])

    for name, res in ranking:
        test_rmse = res['test']['rmse']
        r2 = res['test']['r2']
        unc = res['uncertainty']
        print(f"{name:10s} | TEST RMSE={test_rmse:.4f} | R2={r2:.4f} | Unc={unc:.4f}")

    # 保存最终结果字典
    with open(f"{BASE_PATH2}/results/ALL_results_sp2.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print(f"\nDone. Time: {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    main()