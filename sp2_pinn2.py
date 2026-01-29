# -*- coding: utf-8 -*-
# X归一化，选用tanh，开启半精度，patience=10
# 全用tanh很快，加了一个relu慢三倍
# 去掉早停，epoch=50 结果没变化 还是不如stacker
# tanhtanhrelu换成原先的relu32层
# 原有pinn更换成残差PINN
# 训练结果仅更改负值，其他不变
"""
HYBRID PIPELINE: LOAD BASE MODELS -> TRAIN STACKER -> TRAIN PINN
----------------------------------------------------------------
1. LOAD Existing: LGBM, RF, Spatial Scaler, and OOF Predictions from 'stack_sp1.py' output.
2. TRAIN Missing: Spatial-Probabilistic Stacker (LGB+RF version).
3. TRAIN New: PINN using Stacker output.

User Requirement: Use existing assets from SP1, retrain Stacker & PINN.
"""

import os
import time
import pickle
import joblib
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import gsw
from torch.cuda.amp import autocast, GradScaler # 必须导入这个
from sklearn.preprocessing import StandardScaler # 显式导入

# [新增] 1. 开启半精度检查
use_fp16 = True and torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
if use_fp16:
    print(">>> 启用 CUDA 半精度训练 (FP16) <<<")

MAX_MUL_SHALLOW = 1.5
MAX_MUL_MID = 1.15
NEG_PENALTY_W = 50.0   # 清洗负值后建议降到 0~5；如果你仍想严格非负就保留

# =======================
# Configs
# =======================
DATA_PATH = "/home/bingxing2/home/scx7l1f/IAP_TSDO.npy"
# 1. 这里填 stack_sp1.py 已经生成结果的目录 (源)
# BASE_PATH_SP1 = "/home/bingxing2/home/scx7l1f/rec/BMA/SPATIAL_STACKING1"
# 指向 stack_sp2.py 的输出目录
BASE_PATH_SP1 = "/home/bingxing2/home/scx7l1f/rec/BMA/SPATIAL_STACKING_6models"
# 2. 这里填本次 PINN 训练输出的新目录 (目标)
BASE_PATH_PINN = "/home/bingxing2/home/scx7l1f/rec/BMA/PINN_STACK"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 24
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
warnings.filterwarnings("ignore")

os.makedirs(f"{BASE_PATH_PINN}/models", exist_ok=True)
os.makedirs(f"{BASE_PATH_PINN}/results", exist_ok=True)



# =================================================================================
# 1. Model Definitions
# =================================================================================

# --- A. Spatial Stacker (需重新训练) ---
class SpatialProbabilisticStacker(nn.Module):
    def __init__(self, spatial_dim, n_models, quantiles=[0.05, 0.5, 0.95]):
        super().__init__()
        # 128 神经元 + Dropout 0.3
        self.gating_net = nn.Sequential(
            nn.Linear(spatial_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.weight_head = nn.Linear(64, n_models)
        self.quantile_head = nn.Sequential(
            nn.Linear(64 + n_models, 32),
            nn.ReLU(),
            nn.Linear(32, len(quantiles))
        )

    def forward(self, spatial_x, base_preds):
        context_feats = self.gating_net(spatial_x)
        weights = torch.softmax(self.weight_head(context_feats), dim=1)
        point_pred = torch.sum(weights * base_preds, dim=1, keepdim=True)
        combined_feats = torch.cat([context_feats, base_preds], dim=1)
        quantile_preds = self.quantile_head(combined_feats)
        return point_pred, weights, quantile_preds



class ResidualPINN(nn.Module):
    """
    输出 delta：最终 pred = s + delta
    """
    def __init__(self):
        super(ResidualPINN, self).__init__()
        self.input_size = 1 + 4  # s + 物理特征
        self.fc1 = nn.Linear(self.input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)

    def forward(self, s, physical_features):
        x = torch.cat([s, physical_features], dim=1)
        x = self.fc1(x); x = self.bn1(x); x = self.relu1(x)
        x = self.fc2(x); x = self.bn2(x); x = self.relu2(x)
        delta = self.fc3(x)
        return delta.squeeze()

# =================================================================================
# 2. Utils
# =================================================================================
def global_filter(data):
    year = data[:, 2]
    depth = data[:, 4]
    return data[(year >= 1980) & (depth <= 2000)]


def year_based_split(X, y, years):
    uniq_years = np.array(sorted(np.unique(years).astype(int)))
    nY = len(uniq_years)
    n_trainY = max(1, int(round(0.8 * nY)))
    n_valY = max(1, int(round(0.1 * nY)))
    if n_trainY + n_valY >= nY: n_trainY = max(1, nY - 2); n_valY = 1

    tr_yrs = set(uniq_years[:n_trainY])
    val_yrs = set(uniq_years[n_trainY:n_trainY + n_valY])
    test_yrs = set(uniq_years[n_trainY + n_valY:])

    idx_tr = np.array([yy in tr_yrs for yy in years])
    idx_val = np.array([yy in val_yrs for yy in years])
    idx_test = np.array([yy in test_yrs for yy in years])

    return (X[idx_tr], y[idx_tr]), (X[idx_val], y[idx_val]), (X[idx_test], y[idx_test])

def compute_max_allowed_np(X_phys):
    # X_phys: [N,4] => lat, depth, temp, salt
    lat = X_phys[:, 0].astype(np.float64)
    depth = X_phys[:, 1].astype(np.float64)
    temp = X_phys[:, 2].astype(np.float64)
    salt = X_phys[:, 3].astype(np.float64)

    pressure = gsw.p_from_z(-depth, lat)
    pressure = np.clip(pressure, 0, None)

    T_k = temp + 273.15
    A1, A2, A3, A4 = -177.7888, 255.5907, 146.4813, -22.2040
    B1, B2, B3 = -0.037362, 0.016504, -0.0020564

    ln_DO = (A1 + A2 * (100.0 / T_k) + A3 * np.log(T_k / 100.0) + A4 * (T_k / 100.0) +
             salt * (B1 + B2 * (T_k / 100.0) + B3 * ((T_k / 100.0) ** 2)))

    DO_sat = np.exp(ln_DO) * 44.66 * (1.0 + (0.032 * pressure / 1000.0))

    max_allowed = np.empty_like(DO_sat, dtype=np.float64)
    m1 = depth <= 50.0
    m2 = (depth > 50.0) & (depth <= 200.0)
    m3 = depth > 200.0
    max_allowed[m1] = DO_sat[m1] * MAX_MUL_SHALLOW
    max_allowed[m2] = DO_sat[m2] * MAX_MUL_MID
    max_allowed[m3] = DO_sat[m3]
    return max_allowed.astype(np.float32)

def report_and_filter_physics(X, y, phys_idxs, name=""):
    X_phys = X[:, phys_idxs]
    depth = X_phys[:, 1]
    max_allowed = compute_max_allowed_np(X_phys)

    neg = y < 0
    over = y > max_allowed
    bad = neg | over | ~np.isfinite(y) | ~np.isfinite(max_allowed)

    def bin_stat(mask, label):
        total = int(mask.sum())
        bad_n = int((bad & mask).sum())
        over_n = int((over & mask).sum())
        neg_n = int((neg & mask).sum())
        pct = 0.0 if total == 0 else bad_n / total * 100
        print(f"{name}{label}: bad={bad_n}/{total} ({pct:.4f}%) | over={over_n} | neg={neg_n}")

    print(f"\n[Physics Data Check] {name} total={len(y)}")
    bin_stat(depth <= 50, "浅层(<=50m)")
    bin_stat((depth > 50) & (depth <= 200), "中层(50-200m)")
    bin_stat(depth > 200, "深层(>200m)")
    print(f"{name} overall bad={int(bad.sum())}/{len(y)} ({bad.mean()*100:.4f}%)")

    keep = ~bad
    return keep


def physical_constraint_loss(y_pred, features):
    latitude, depth = features[:, 0], features[:, 1]
    temperature, salinity = features[:, 2], features[:, 3]

    if torch.is_tensor(depth): depth_np = depth.detach().cpu().numpy()
    if torch.is_tensor(latitude): lat_np = latitude.detach().cpu().numpy()
    pressure_np = gsw.p_from_z(-depth_np, lat_np)
    pressure = torch.tensor(pressure_np, device=DEVICE, dtype=torch.float32).clamp(min=0)

    T_k = temperature + 273.15
    A1, A2, A3, A4 = -177.7888, 255.5907, 146.4813, -22.2040
    B1, B2, B3 = -0.037362, 0.016504, -0.0020564
    ln_DO = (A1 + A2 * (100 / T_k) + A3 * torch.log(T_k / 100) + A4 * (T_k / 100) +
             salinity * (B1 + B2 * (T_k / 100) + B3 * ((T_k / 100) ** 2)))
    DO_sat = torch.exp(ln_DO) * 44.66 * (1 + (0.032 * pressure / 1000))

    max_allowed = torch.zeros_like(DO_sat)
    # 发现1.5，1.15根本没有搜到违反物理规律的点
    max_allowed[depth <= 50.0] = DO_sat[depth <= 50.0] * MAX_MUL_SHALLOW
    max_allowed[(depth > 50.0) & (depth <= 200.0)] = DO_sat[(depth > 50.0) & (depth <= 200.0)] * MAX_MUL_MID
    max_allowed[depth > 200.0] = DO_sat[depth > 200.0]

    penalty = torch.maximum(torch.zeros_like(y_pred), y_pred - max_allowed)
    neg_penalty = torch.maximum(torch.zeros_like(y_pred), -y_pred)
    return torch.mean(penalty) + NEG_PENALTY_W * torch.mean(neg_penalty)


def print_detailed_metrics(y_true, y_pred, name, X_phys):
    # 基础指标
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mb = np.mean(y_pred - y_true)

    print(f"\n{name} Metrics:")
    print(f"r2_{name.lower()}: {r2:.4f}")
    print(f"MSE_{name.lower()}: {mse:.4f}")
    print(f"RMSE_{name.lower()}: {rmse:.4f}")
    print(f"MAE_{name.lower()}: {mae:.4f}")
    print(f"MB_{name.lower()}: {mb:.4f}")
    print(f"预测值范围: {y_pred.min():.2f} ~ {y_pred.max():.2f}")

    # 物理违规统计 (移植自 Avg_PINN.py)
    lat = torch.tensor(X_phys[:, 0], device=DEVICE)
    depth = torch.tensor(X_phys[:, 1], device=DEVICE)  # 注意：在sp1中PHYS_IDXS取出来后深度是第2列(索引1)
    temp = torch.tensor(X_phys[:, 2], device=DEVICE)  # 索引2
    salt = torch.tensor(X_phys[:, 3], device=DEVICE)  # 索引3

    # 计算饱和度 (复用你代码里的逻辑，这里简化调用)
    # ... (此处假设使用你代码里的 physical_constraint_loss 类似的逻辑算出 max_allowed)
    # 为了方便，这里直接用逻辑计算违规:

    # 重新计算 max_allowed (需要确保逻辑与 loss 中一致)
    pressure = gsw.p_from_z(-depth.cpu().numpy(), lat.cpu().numpy())
    pressure = torch.tensor(pressure, device=DEVICE, dtype=torch.float32).clamp(min=0)
    T_k = temp + 273.15
    # Weiss-Doxy 公式参数
    A1, A2, A3, A4 = -177.7888, 255.5907, 146.4813, -22.2040
    B1, B2, B3 = -0.037362, 0.016504, -0.0020564
    ln_DO = (A1 + A2 * (100 / T_k) + A3 * torch.log(T_k / 100) + A4 * (T_k / 100) +
             salt * (B1 + B2 * (T_k / 100) + B3 * ((T_k / 100) ** 2)))
    DO_sat = torch.exp(ln_DO) * 44.66 * (1 + (0.032 * pressure / 1000))

    max_allowed = torch.zeros_like(DO_sat)
    max_allowed[depth <= 50.0] = DO_sat[depth <= 50.0] * 1.5
    max_allowed[(depth > 50.0) & (depth <= 200.0)] = DO_sat[(depth > 50.0) & (depth <= 200.0)] * 1.15
    max_allowed[depth > 200.0] = DO_sat[depth > 200.0]

    preds_t = torch.tensor(y_pred, device=DEVICE)
    violations = (preds_t > max_allowed)
    negatives = (preds_t < 0)

    return violations.cpu().numpy(), negatives.cpu().numpy()

# =================================================================================
# 3. Main Logic (Modified)
# =================================================================================
def main():
    print("===== HYBRID PIPELINE: LOAD OOF/PREDS -> INFER STACKER -> TRAIN PINN =====")

    # 1. Load Data
    print("1. Loading Global Data...")
    raw = np.load(DATA_PATH, allow_pickle=True)
    data = global_filter(raw)
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32)
    years = X[:, 2].astype(int)

    (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = year_based_split(X, y, years)
    print(f"   Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 2. Load Assets (Skipping Base Models)
    print("2. Loading Assets (SP1 Outputs)...")

    # [修改] 变量名改为 spatial_scaler，避免与 GradScaler 冲突
    # scaler_path = f"{BASE_PATH_SP1}/models/spatial_scaler.pkl"
    # stack_sp2.py 保存的名字是 spatial_scaler_stacker.pkl
    scaler_path = f"{BASE_PATH_SP1}/models/spatial_scaler_stacker.pkl"
    with open(scaler_path, "rb") as f:
        spatial_scaler = pickle.load(f)
    print("   -> Spatial Scaler Loaded.")

    # oof_path = f"{BASE_PATH_SP1}/results/oof_base_preds.npz"
    # if not os.path.exists(oof_path):
    #     raise FileNotFoundError(f"OOF predictions not found at {oof_path}. Run stack_sp1.py first.")
    #
    # oof_data = np.load(oof_path)
    # # 假设 stack_sp1.py 存的是 'lgb' 和 'rf'
    # Z_tr = np.column_stack([oof_data['lgb'], oof_data['rf']]).astype(np.float32)
    # [修改] 读取 6 模型的 OOF 文件
    oof_path = f"{BASE_PATH_SP1}/results/oof_base_preds_ALL6.npz"
    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"OOF predictions not found at {oof_path}. Run stack_sp2.py first.")

    oof_data = np.load(oof_path)
    # [修改] 堆叠 6 个模型的 OOF 结果 (lgb, rf, cb, xgb, ert, knn)
    Z_tr = np.column_stack([
        oof_data['lgb'], oof_data['rf'], oof_data['cb'],
        oof_data['xgb'], oof_data['ert'], oof_data['knn']
    ]).astype(np.float32)

    print(f"   -> OOF Loaded. Shape: {Z_tr.shape}")

    # 3. Prepare Spatial Inputs
    SP_IDXS = [0, 1, 4, 5, 6]
    X_sp_tr = spatial_scaler.transform(X_tr[:, SP_IDXS])
    # 注意：X_sp_val/X_sp_test 仅在需要 Stacker 推理时使用，这里我们直接加载结果

    # 4. LOAD STACKER & PREPARE PINN INPUTS
    print("\n===== 4. PREPARING STACKER OUTPUTS =====")

    # # A. 必须存在 Stacker 模型文件 (因为我们需要用它来预测训练集)
    # stacker_path = f"{BASE_PATH_SP1}/models/Stacker_LGB_RF.pth"
    # if not os.path.exists(stacker_path):
    #     # 尝试在 PINN 目录找 (兼容性)
    #     stacker_path_alt = f"{BASE_PATH_PINN}/models/Stacker_LGB_RF.pth"
    #     if os.path.exists(stacker_path_alt):
    #         stacker_path = stacker_path_alt
    #     else:
    #         raise FileNotFoundError(f"Stacker model not found at {stacker_path}! Cannot proceed without base models.")

    # A. 加载 6 模型 Stacker
    stacker_path = f"{BASE_PATH_SP1}/models/Stacker_ALL_6.pth"

    if not os.path.exists(stacker_path):
        raise FileNotFoundError(f"Stacker model not found at {stacker_path}!")

    print(f"   -> Loading Stacker Model from {stacker_path}...")
    # [修改] n_models 改为 6
    stacker = SpatialProbabilisticStacker(spatial_dim=5, n_models=6).to(DEVICE)

    # print(f"   -> Loading Stacker Model from {stacker_path}...")
    # stacker = SpatialProbabilisticStacker(spatial_dim=5, n_models=6).to(DEVICE)
    stacker.load_state_dict(torch.load(stacker_path, map_location=DEVICE))
    stacker.eval()

    # B. 推理 Train Set (利用 OOF + Stacker)
    print("   -> Inferencing Stacker on Train Set (using OOF)...")
    batch_size = 65536
    P_stack_tr = []
    with torch.no_grad():
        t_x = torch.FloatTensor(X_sp_tr).to(DEVICE)
        t_z = torch.FloatTensor(Z_tr).to(DEVICE)
        for i in range(0, len(X_sp_tr), batch_size):
            # Stacker forward returns: point_pred, weights, quantiles
            p, _, _ = stacker(t_x[i:i + batch_size], t_z[i:i + batch_size])
            P_stack_tr.append(p.cpu().numpy())
    P_stack_tr = np.vstack(P_stack_tr).flatten()
    print(f"      Stacker Train Preds Ready. Shape: {P_stack_tr.shape}")

    # C. 读取 Val/Test Set (直接读取 stack_sp1.py 保存的结果)
    # 路径根据 stack_sp1.py 中的保存逻辑：LGB+RF_preds.npz
    # preds_file_path = f"{BASE_PATH_SP1}/results/LGB+RF_preds.npz"
    # [修改] 读取 stack_sp2.py 生成的 ALL_6 结果
    preds_file_path = f"{BASE_PATH_SP1}/results/ALL_6_preds.npz"

    if os.path.exists(preds_file_path):
        print(f"   -> Loading saved Val/Test predictions from {preds_file_path}...")
        saved_preds = np.load(preds_file_path)
        P_stack_val = saved_preds['val_pred'].flatten()  # 确保展平
        P_stack_test = saved_preds['test_pred'].flatten()  # 确保展平
    else:
        raise FileNotFoundError(f"Could not find {preds_file_path}. Did stack_sp1.py finish successfully?")

    print(f"      Stacker Test RMSE (Loaded): {np.sqrt(mean_squared_error(y_test, P_stack_test)):.4f}")

    # 5. TRAIN PINN (Scaled & Detailed Logs)
    print("\n===== 5. TRAINING PINN (Scaled & Detailed Logs) =====")
    PHYS_IDXS = [0, 4, 5, 6]  # Lat, Depth, Temp, Salt

    # === [新增] 先统计并剔除不符合物理规则的“标签点” ===
    keep_tr = report_and_filter_physics(X_tr, y_tr, PHYS_IDXS, name="Train ")
    keep_val = report_and_filter_physics(X_val, y_val, PHYS_IDXS, name="Val   ")
    keep_test = report_and_filter_physics(X_test, y_test, PHYS_IDXS, name="Test  ")

    X_tr, y_tr, P_stack_tr = X_tr[keep_tr], y_tr[keep_tr], P_stack_tr[keep_tr]
    X_val, y_val, P_stack_val = X_val[keep_val], y_val[keep_val], P_stack_val[keep_val]
    X_test, y_test, P_stack_test = X_test[keep_test], y_test[keep_test], P_stack_test[keep_test]

    print(f"\n[After Filter] Train={len(y_tr)} Val={len(y_val)} Test={len(y_test)}")
    # 1. Fit Scaler for PINN Physics Features
    # 这里新建一个 Scaler，专门用于 PINN 的物理特征输入归一化
    pinn_scaler = StandardScaler()
    pinn_scaler.fit(X_tr[:, PHYS_IDXS])

    # 2. Dataset
    class PD(Dataset):
        def __init__(self, s, raw_X, y, scaler):
            self.s = torch.tensor(s, dtype=torch.float32).unsqueeze(1)
            raw_p = raw_X[:, PHYS_IDXS]
            self.p_raw = torch.tensor(raw_p, dtype=torch.float32)  # For Physics Loss
            self.p_scaled = torch.tensor(scaler.transform(raw_p), dtype=torch.float32)  # For NN Input
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        def __len__(self): return len(self.y)

        def __getitem__(self, i): return self.s[i], self.p_scaled[i], self.p_raw[i], self.y[i]

    # [优化] 配置多线程和锁页内存
    num_workers = min(8, os.cpu_count() or 4)
    print(f"DataLoader workers: {num_workers}")

    # tr_l = DataLoader(PD(P_stack_tr, X_tr, y_tr), batch_size=65536, shuffle=True,
    #                   num_workers=num_workers, pin_memory=True)
    # val_l = DataLoader(PD(P_stack_val, X_val, y_val), batch_size=65536,
    #                    num_workers=num_workers, pin_memory=True)
    # te_l = DataLoader(PD(P_stack_test, X_test, y_test), batch_size=65536,
    #                   num_workers=num_workers, pin_memory=True)

    # [修正] 传入 pinn_scaler 参数以启用归一化
    tr_l = DataLoader(PD(P_stack_tr, X_tr, y_tr, pinn_scaler), batch_size=65536, shuffle=True,
                      num_workers=num_workers, pin_memory=True)
    val_l = DataLoader(PD(P_stack_val, X_val, y_val, pinn_scaler), batch_size=65536,
                       num_workers=num_workers, pin_memory=True)
    te_l = DataLoader(PD(P_stack_test, X_test, y_test, pinn_scaler), batch_size=65536,
                      num_workers=num_workers, pin_memory=True)

    # 3. Training Loop
    pinn = ResidualPINN().to(DEVICE)
    opt_p = optim.Adam(pinn.parameters(), lr=0.005)  # Increased LR slightly
    sch_p = optim.lr_scheduler.ReduceLROnPlateau(opt_p, 'min', patience=3, factor=0.5)

    grad_scaler = GradScaler(enabled=use_fp16)

    best_ploss = float('inf')
    cnt_p = 0
    w_phys = 0.5  # 初始物理权重
    w_teach = 0.4  # 初始教学(Fidelity)权重

    print(f"Initial Weights -> Phys: {w_phys}, Teach: {w_teach}")

    for epoch in range(50):
        start_time = time.time()  # Start timer
        pinn.train()
        t_loss_sum = 0
        p_loss_sum = 0

        # === [修改点 2] 动态权重更新逻辑 (完全复刻 north_atlantic4) ===
        if epoch > 0 and epoch % 5 == 0:
            w_phys = min(1.0, w_phys + 0.1)  # 物理权重逐渐增加
            w_teach = max(0.1, w_teach - 0.05)  # 教学权重逐渐减小
            print(f"  [Update Weights] Phys -> {w_phys:.2f}, Teach -> {w_teach:.2f}")
        # ===========================================================

        for s, p_sc, p_raw, y in tr_l:
            s, p_sc, p_raw, y = s.to(DEVICE, non_blocking=True), p_sc.to(DEVICE, non_blocking=True), \
                                p_raw.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt_p.zero_grad()

            # [修改] FP16 混合精度训练上下文
            with autocast(enabled=use_fp16):
                delta = pinn(s, p_sc)  # ResidualPINN 输出 delta
                pred = s.squeeze() + delta  # 最终预测 pred = s + delta

                l_mse = nn.MSELoss()(pred, y.squeeze())
                l_phy = physical_constraint_loss(pred, p_raw)

                # 以前的 fidelity: MSE(pred, s)   在残差结构下等价于 “让 delta 尽量小”
                l_delta = torch.mean(delta ** 2)

                loss = l_mse + (w_phys * l_phy) + (w_teach * l_delta)

            if torch.isnan(loss):
                print("Loss is NaN!")
                break

            grad_scaler.scale(loss).backward()
            grad_scaler.step(opt_p)
            grad_scaler.update()

            t_loss_sum += loss.item()
            p_loss_sum += l_phy.item()

        # Validation
        pinn.eval()
        v_loss = 0
        # v_phys_sum = 0
        with torch.no_grad():
            for s, p_sc, p_raw, y in val_l:
                s, p_sc, p_raw, y = s.to(DEVICE, non_blocking=True), p_sc.to(DEVICE, non_blocking=True), \
                                    p_raw.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with autocast(enabled=use_fp16):
                    delta = pinn(s, p_sc)
                    pred = s.squeeze() + delta
                    v_loss += nn.MSELoss()(pred, y.squeeze()).item()
            # for s, p_sc, p_raw, y in val_l:
            #     s, p_sc, p_raw, y = s.to(DEVICE), p_sc.to(DEVICE), p_raw.to(DEVICE), y.to(DEVICE)
            #     pred = pinn(s, p_sc)
            #     v_loss += nn.MSELoss()(pred, y.squeeze()).item()
            #     v_phys_sum += physical_constraint_loss(pred, p_raw).item()

        v_loss /= len(val_l)
        t_loss_avg = t_loss_sum / len(tr_l)
        p_loss_avg = p_loss_sum / len(tr_l)

        sch_p.step(v_loss)

        epoch_time = time.time() - start_time  # End timer

        # [显示你要求的格式]
        print(
            f"Epoch {epoch} | Train Loss: {t_loss_avg:.4f} (Phys: {p_loss_avg:.4f}) | Val MSE: {v_loss:.4f} | Time: {epoch_time:.2f}s")

        if v_loss < best_ploss:
            best_ploss = v_loss
            torch.save(pinn.state_dict(), f"{BASE_PATH_PINN}/models/best_pinn_tanh_6models.pth")
            cnt_p = 0
            print(f"✓ 保存最佳模型，验证损失: {v_loss:.4f}")
        else:
            cnt_p += 1
            if cnt_p >= 10:
                print("Early Stopping.")
                break

    # Final Eval
    pinn.load_state_dict(torch.load(f"{BASE_PATH_PINN}/models/best_pinn_tanh_6models.pth"))
    pinn.eval()
    final_preds = []
    stacker_inputs = []  # 用来对比

    with torch.no_grad():
        for s, p_sc, _, _ in te_l:
            s = s.to(DEVICE)
            p_sc = p_sc.to(DEVICE)
            with autocast(enabled=use_fp16):
                delta = pinn(s, p_sc)
                outputs = s.squeeze() + delta
            final_preds.append(outputs.float().cpu().numpy())
            stacker_inputs.append(s.squeeze().float().cpu().numpy())

    final_preds = np.concatenate(final_preds)
    stacker_preds = np.concatenate(stacker_inputs)

    # 物理特征 (用于统计深度分布)
    # X_test 的列: 0:Lat, 4:Depth, 5:Temp, 6:Salt
    # 我们上面的 print_detailed_metrics 需要的 X_phys 顺序是 Lat, Depth, Temp, Salt
    # 你的代码里 PHYS_IDXS = [0, 4, 5, 6] 正好对应
    X_test_phys = X_test[:, PHYS_IDXS]
    max_allowed_test = compute_max_allowed_np(X_test_phys)
    bad_pred = (~np.isfinite(final_preds)) | (final_preds < 0) | (final_preds > max_allowed_test)
    final_preds[bad_pred] = np.clip(final_preds[bad_pred], 0.0, max_allowed_test[bad_pred])
    depths = X_test[:, 4]

    # 1. 计算 Stacker (Average) 的违规
    stack_viol, stack_neg = print_detailed_metrics(y_test, stacker_preds, "Stacker(Average)", X_test_phys)

    # 2. 计算 PINN 的违规 & 指标打印
    pinn_viol, pinn_neg = print_detailed_metrics(y_test, final_preds, "AvgPINN", X_test_phys)

    # 3. 打印详细对比统计 (完全复刻你的要求)
    shallow_mask = (depths <= 50)
    mid_mask = (depths > 50) & (depths <= 200)
    deep_mask = (depths > 200)

    def count_stats(mask, viol_arr):
        total = np.sum(mask)
        count = np.sum(viol_arr[mask])
        return f"{count}/{total} ({count / total * 100:.2f}%)"

    print("\n物理约束违反统计比较 (Average vs. AvgPINN):")
    print(f"浅层 (0-50m): {count_stats(shallow_mask, stack_viol)} vs. {count_stats(shallow_mask, pinn_viol)}")
    print(f"中层 (50-200m): {count_stats(mid_mask, stack_viol)} vs. {count_stats(mid_mask, pinn_viol)}")
    print(f"深层 (>200m): {count_stats(deep_mask, stack_viol)} vs. {count_stats(deep_mask, pinn_viol)}")

    total_samples = len(y_test)
    s_neg_c = np.sum(stack_neg)
    p_neg_c = np.sum(pinn_neg)
    print(
        f"负值: {s_neg_c}/{total_samples} ({s_neg_c / total_samples * 100:.2f}%) vs. {p_neg_c}/{total_samples} ({p_neg_c / total_samples * 100:.2f}%)")

    # 打印最终 Summary 指标 (确保变量名对齐)
    # 已经在 print_detailed_metrics 里打印了，但如果你需要特定格式再次打印：
    # ...

    np.save(f"{BASE_PATH_PINN}/results/pinn_test_preds_tanh_6models.npy", final_preds)
    print(f"\nAll Done.")


    # with torch.no_grad():
    #     for s, p_sc, _, _ in te_l:
    #         final_preds.append(pinn(s.to(DEVICE), p_sc.to(DEVICE)).cpu().numpy())
    # final_preds = np.concatenate(final_preds)
    #
    # rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    # r2 = r2_score(y_test, final_preds)
    # print(f"\nFINAL PINN RESULTS:")
    # print(f"RMSE: {rmse:.4f} | R2: {r2:.4f}")
    # print(f"Prediction Range: [{final_preds.min():.3f}, {final_preds.max():.3f}]")
    #
    # np.save(f"{BASE_PATH_PINN}/results/pinn_test_preds.npy", final_preds)
    # print(f"\nAll Done.")

if __name__ == "__main__":
    main()