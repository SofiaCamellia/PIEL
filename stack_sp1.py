# -*- coding: utf-8 -*-
# Stacking神经元64-128翻倍与否都可以！不影响结果
"""
GLOBAL | Year-based outer split | GroupKFold(year) OOF | Spatial-Probabilistic Stacking | 7-set ablation

[Restored User's Verbose Settings]:
- RandomForest: verbose=2
- CatBoost: verbose=50
- LightGBM: log_evaluation(100) for full train
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
from torch.utils.data import TensorDataset, DataLoader

# ML & Metrics
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
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
DATA_PATH = "/home/bingxing2/home/scx7l1f/IAP_TSDO.npy"
BASE_PATH = "/home/bingxing2/home/scx7l1f/rec/BMA/SPATIAL_STACKING1"
K_FOLDS = 5

# Stacker Hyperparams
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
META_EPOCHS = 100
META_LR = 0.005
META_BATCH_SIZE = 65536
QUANTILES = [0.05, 0.5, 0.95]
PINBALL_LAMBDA = 0.2

os.makedirs(f"{BASE_PATH}/models", exist_ok=True)
os.makedirs(f"{BASE_PATH}/data", exist_ok=True)
os.makedirs(f"{BASE_PATH}/results", exist_ok=True)


# =====================================================================================
# Part 1) Data Utils
# =====================================================================================
def global_filter(data):
    year = data[:, 2]
    depth = data[:, 4]
    mask = (year >= 1980) & (depth <= 2000)
    return data[mask]


def year_based_split(X, y, years, train_ratio=0.8, val_ratio=0.1):
    uniq_years = np.array(sorted(np.unique(years).astype(int)))
    nY = len(uniq_years)
    if nY < 5:
        raise ValueError(f"Not enough unique years ({nY}) for year split. Need >=5.")

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
# Part 2) Base Models (With User's VERBOSE settings restored)
# =====================================================================================
def make_lgbm():
    return lgb.LGBMRegressor(
        boosting_type='gbdt',
        num_leaves=156,
        max_depth=12,
        min_child_samples=30,
        learning_rate=0.05,
        n_estimators=3500,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbosity=-1  # Keep -1 to suppress warnings, control logs via callbacks
    )


def make_rf():
    return RandomForestRegressor(
        n_estimators=40,
        random_state=RANDOM_SEED,
        max_depth=20,
        min_samples_leaf=50,
        n_jobs=8,  # Keep optimization
        verbose=2,
        oob_score=False,
        max_features='sqrt'
    )


def make_cb():
    return CatBoostRegressor(
        iterations=3500,
        depth=12,
        learning_rate=0.05,
        loss_function='RMSE',
        eval_metric='RMSE',
        subsample=0.8,
        l2_leaf_reg=4,
        grow_policy='SymmetricTree',
        bootstrap_type='Bernoulli',
        random_seed=RANDOM_SEED,
        verbose=50,  # <--- RESTORED: User wanted verbose=50
        early_stopping_rounds=20,
        task_type='GPU',  # Keep optimization
        devices='0'
    )


# =====================================================================================
# Part 3) OOF Generation & Full Training
# =====================================================================================
def fit_predict_oof_group_year(X_train, y_train, years_train, K=5):
    n = len(X_train)
    oof = {
        "lgb": np.zeros(n, dtype=np.float32),
        "rf": np.zeros(n, dtype=np.float32),
        "cb": np.zeros(n, dtype=np.float32),
    }

    gkf = GroupKFold(n_splits=K)
    for fold, (idx_fit, idx_oof) in enumerate(gkf.split(X_train, y_train, groups=years_train), 1):
        print(f"\n[OOF] Fold {fold}/{K} ...")  # Print newline for better readability with verbose

        X_fit, y_fit = X_train[idx_fit], y_train[idx_fit]
        X_oof = X_train[idx_oof]
        X_inner, y_inner, X_es, y_es = make_time_ordered_es_split(X_fit, y_fit, frac=0.15)

        # ---- LGBM ----
        # Restored: log_evaluation(0) for OOF to keep fold logs clean-ish, or change to 100 if you want logs here too
        lgbm = make_lgbm()
        lgbm.fit(
            X_inner, y_inner,
            eval_set=[(X_es, y_es)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        oof["lgb"][idx_oof] = lgbm.predict(X_oof).astype(np.float32)

        # ---- RF ----
        # Verbose handled by constructor (verbose=2)
        rf = make_rf()
        rf.fit(X_fit, y_fit)
        oof["rf"][idx_oof] = rf.predict(X_oof).astype(np.float32)

        # ---- CatBoost ----
        # Verbose handled by constructor (verbose=50)
        cb = make_cb()
        cb.fit(X_inner, y_inner, eval_set=(X_es, y_es), use_best_model=True)
        oof["cb"][idx_oof] = cb.predict(X_oof).astype(np.float32)

    print("\n  OOF Generation Complete.")
    return oof


def train_full_base_models_train_only(X_train, y_train):
    X_inner, y_inner, X_es, y_es = make_time_ordered_es_split(X_train, y_train, frac=0.15)

    print("\nTraining FULL LightGBM...")
    lgbm = make_lgbm()
    # RESTORED: log_evaluation(100)
    lgbm.fit(X_inner, y_inner, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

    print("\nTraining FULL RandomForest...")
    rf = make_rf()
    rf.fit(X_train, y_train)

    print("\nTraining FULL CatBoost...")
    cb = make_cb()
    cb.fit(X_inner, y_inner, eval_set=(X_es, y_es), use_best_model=True)

    return lgbm, rf, cb


def predict_base_models(models, X):
    lgbm, rf, cb = models
    return {
        "lgb": lgbm.predict(X).astype(np.float32),
        "rf": rf.predict(X).astype(np.float32),
        "cb": cb.predict(X).astype(np.float32),
    }


# =====================================================================================
# Part 4) Meta Layer: Spatial-Probabilistic Stacker (PyTorch)
# =====================================================================================
# class SpatialProbabilisticStacker(nn.Module):
#     def __init__(self, spatial_dim, n_models, quantiles):
#         super().__init__()
#         self.n_models = n_models
#         self.quantiles = quantiles
#
#         self.gating_net = nn.Sequential(
#             nn.Linear(spatial_dim, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         self.weight_head = nn.Linear(32, n_models)
#         self.quantile_head = nn.Sequential(
#             nn.Linear(32 + n_models, 32),
#             nn.ReLU(),
#             nn.Linear(32, len(quantiles))
#         )

class SpatialProbabilisticStacker(nn.Module):
    def __init__(self, spatial_dim, n_models, quantiles):
        super().__init__()
        self.n_models = n_models
        self.quantiles = quantiles

        # === [修改 1] 增强门控网络容量 ===
        self.gating_net = nn.Sequential(
            nn.Linear(spatial_dim, 128),  # 原来是 64 -> 改为 128
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),  # 原来是 0.2 -> 改为 0.3 (防止被差模型带偏)
            nn.Linear(128, 64),  # 原来是 32 -> 改为 64
            nn.ReLU()
        )

        # === [修改 2] 对应调整 Weight Head 输入维度 ===
        self.weight_head = nn.Linear(64, n_models)  # 输入从 32 变为 64

        # === [修改 3] 对应调整 Quantile Head 输入维度 ===
        self.quantile_head = nn.Sequential(
            nn.Linear(64 + n_models, 32),  # 输入从 32+n 变为 64+n
            nn.ReLU(),
            nn.Linear(32, len(quantiles))
        )

    def forward(self, spatial_x, base_preds):
        context_feats = self.gating_net(spatial_x)
        raw_weights = self.weight_head(context_feats)
        weights = torch.softmax(raw_weights, dim=1)
        point_pred = torch.sum(weights * base_preds, dim=1, keepdim=True)
        combined_feats = torch.cat([context_feats, base_preds], dim=1)
        quantile_preds = self.quantile_head(combined_feats)
        return point_pred, weights, quantile_preds


def pinball_loss(preds, target, quantiles):
    loss = 0.0
    for i, q in enumerate(quantiles):
        error = target - preds[:, i:i + 1]
        loss += torch.mean(torch.max((q * error), ((q - 1) * error)))
    return loss


def train_stacker_fast(X_spatial_tr, Base_preds_tr, y_tr,
                       X_spatial_val, Base_preds_val, y_val,
                       quantiles=QUANTILES):
    spatial_dim = X_spatial_tr.shape[1]
    n_models = Base_preds_tr.shape[1]

    # Whole-Data-On-GPU
    X_sp_tr_dev = torch.FloatTensor(X_spatial_tr).to(DEVICE)
    Base_p_tr_dev = torch.FloatTensor(Base_preds_tr).to(DEVICE)
    y_tr_dev = torch.FloatTensor(y_tr).view(-1, 1).to(DEVICE)

    X_sp_val_dev = torch.FloatTensor(X_spatial_val).to(DEVICE)
    Base_p_val_dev = torch.FloatTensor(Base_preds_val).to(DEVICE)
    y_val_dev = torch.FloatTensor(y_val).view(-1, 1).to(DEVICE)

    n_samples = X_sp_tr_dev.shape[0]
    n_batches = int(np.ceil(n_samples / META_BATCH_SIZE))

    model = SpatialProbabilisticStacker(spatial_dim, n_models, quantiles).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=META_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # print(f"  Training Stacker on {DEVICE}...")

    for epoch in range(META_EPOCHS):
        model.train()
        perm = torch.randperm(n_samples, device=DEVICE)

        for i in range(n_batches):
            idx = perm[i * META_BATCH_SIZE: (i + 1) * META_BATCH_SIZE]
            optimizer.zero_grad()
            p_pred, _, q_pred = model(X_sp_tr_dev[idx], Base_p_tr_dev[idx])
            l_mse = nn.MSELoss()(p_pred, y_tr_dev[idx])
            l_pin = pinball_loss(q_pred, y_tr_dev[idx], quantiles)
            loss = l_mse + PINBALL_LAMBDA * l_pin
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp_pred, _, vq_pred = model(X_sp_val_dev, Base_p_val_dev)
            vl_mse = nn.MSELoss()(vp_pred, y_val_dev)
            vl_pin = pinball_loss(vq_pred, y_val_dev, quantiles)
            val_loss = (vl_mse + PINBALL_LAMBDA * vl_pin).item()

        scheduler.step(val_loss)

        # === [新增] 打印训练日志 ===
        if epoch % 10 == 0:
            print(f"    [Epoch {epoch}] Val Loss: {val_loss:.4f}")
        # =========================

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 15:
            break

    model.load_state_dict(best_model_state)
    return model


def inference_stacker_fast(model, X_spatial, Base_preds):
    model.eval()
    batch_size = META_BATCH_SIZE * 4
    n_samples = len(X_spatial)
    n_batches = int(np.ceil(n_samples / batch_size))

    all_p, all_w, all_q = [], [], []

    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            sp_x = torch.FloatTensor(X_spatial[start:end]).to(DEVICE)
            bp = torch.FloatTensor(Base_preds[start:end]).to(DEVICE)
            p, w, q = model(sp_x, bp)
            all_p.append(p.cpu().numpy())
            all_w.append(w.cpu().numpy())
            all_q.append(q.cpu().numpy())

    return np.vstack(all_p).flatten(), np.vstack(all_w), np.vstack(all_q)


# =====================================================================================
# Part 5) Ablation Sets
# =====================================================================================
MODEL_SETS = {
    "LGB": ["lgb"],
    "RF": ["rf"],
    "CB": ["cb"],
    "LGB+RF": ["lgb", "rf"],
    "LGB+CB": ["lgb", "cb"],
    "RF+CB": ["rf", "cb"],
    "LGB+RF+CB": ["lgb", "rf", "cb"],
}


def stack_preds(pred_dict, keys):
    return np.column_stack([pred_dict[k] for k in keys]).astype(np.float32)


def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    mb = float(np.mean(y_pred - y_true))  # Mean Bias

    # --- [修改点] 增加 range 输出 ---
    print(f"  {name} | RMSE={rmse:.4f} | R2={r2:.4f} | MAE={mae:.4f} | MB={mb:.4f} | range=[{y_pred.min():.3f}, {y_pred.max():.3f}]")

    return {"rmse": rmse, "r2": r2, "mae": mae, "mb": mb}


# =====================================================================================
# Main
# =====================================================================================
def main():
    t0 = time.time()
    print(f"\n===== GLOBAL | YEAR-SPLIT | SPATIAL-PROBABILISTIC STACKING (GPU) =====")

    # 1) Load
    print("Loading raw global data...")
    raw = np.load(DATA_PATH, allow_pickle=True)
    data = global_filter(raw)
    np.save(f"{BASE_PATH}/data/global_filtered.npy", data)
    print(f"Filtered global points: {len(data)}")

    # 2) X/y
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32)
    years = X[:, 2].astype(int)

    # 3) Outer Split
    (X_tr, y_tr, tr_yrs), (X_val, y_val, val_yrs), (X_test, y_test, test_yrs) = \
        year_based_split(X, y, years, train_ratio=0.8, val_ratio=0.1)

    print(f"\nSplit: Train={len(tr_yrs)}yrs, Val={len(val_yrs)}yrs, Test={len(test_yrs)}yrs")
    with open(f"{BASE_PATH}/results/year_split.pkl", "wb") as f:
        pickle.dump({"train_years": tr_yrs, "val_years": val_yrs, "test_years": test_yrs}, f)

    # 4) OOF Preds (Base Layer)
    oof_path = f"{BASE_PATH}/results/oof_base_preds.npz"
    if os.path.exists(oof_path):
        print("\nLoading cached OOF preds...")
        tmp = np.load(oof_path)
        oof = {k: tmp[k] for k in tmp}
    else:
        print("\nGenerating OOF preds (GroupKFold)...")
        oof = fit_predict_oof_group_year(X_tr, y_tr, X_tr[:, 2].astype(int), K=K_FOLDS)
        np.savez(oof_path, **oof)

    # 5) Train Full Base Models
    print("\nTraining FULL base models on TRAIN only...")
    # 定义模型保存路径
    lgb_path = f"{BASE_PATH}/models/lgb_model.pkl"
    rf_path = f"{BASE_PATH}/models/rf_model.joblib"
    cb_path = f"{BASE_PATH}/models/catboost_model.cbm"

    # 检查是否所有模型文件都存在
    if os.path.exists(lgb_path) and os.path.exists(rf_path) and os.path.exists(cb_path):
        print("  -> Found all saved Full Models. Loading directly (Skipping Retrain)...")

        # 1. Load LightGBM
        with open(lgb_path, "rb") as f:
            lgbm = pickle.load(f)

        # 2. Load RandomForest
        rf = joblib.load(rf_path)

        # 3. Load CatBoost (需要先实例化一个空对象)
        cb = CatBoostRegressor()
        cb.load_model(cb_path)

        full_models = (lgbm, rf, cb)
        print("  -> All Base Models Loaded Successfully.")

    else:
        print("  -> Saved models not found (or incomplete). Retraining FULL base models...")

        # 训练所有模型
        full_models = train_full_base_models_train_only(X_tr, y_tr)

        # === Saving Base Models ===
        print("  -> Saving new base models to disk...")
        with open(lgb_path, "wb") as f:
            pickle.dump(full_models[0], f)
        joblib.dump(full_models[1], rf_path)
        full_models[2].save_model(cb_path)
        print("  -> Save Complete.")

    full_models = train_full_base_models_train_only(X_tr, y_tr)

    # # === Saving Base Models ===
    # print("Saving base models to disk...")
    # with open(f"{BASE_PATH}/models/lgb_model.pkl", "wb") as f:
    #     pickle.dump(full_models[0], f)
    # joblib.dump(full_models[1], f"{BASE_PATH}/models/rf_model.joblib")
    # full_models[2].save_model(f"{BASE_PATH}/models/catboost_model.cbm")
    # ==========================

    # Predict on Val/Test
    val_preds = predict_base_models(full_models, X_val)
    test_preds = predict_base_models(full_models, X_test)

    # 6) Prepare Spatial Features for Stacker
    SPATIAL_IDXS = [0, 1, 4, 5, 6]  # Lat, Lon, Depth, Temp, Salt

    scaler = StandardScaler()
    X_sp_tr = scaler.fit_transform(X_tr[:, SPATIAL_IDXS])
    X_sp_val = scaler.transform(X_val[:, SPATIAL_IDXS])
    X_sp_test = scaler.transform(X_test[:, SPATIAL_IDXS])

    with open(f"{BASE_PATH}/models/spatial_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 7) Ablation Loop
    print("\n===== Running Spatial-Stacking Ablation (7 Sets) =====")
    all_results = {}

    for exp_name, keys in MODEL_SETS.items():
        print(f"\n>>> Experiment: {exp_name} {keys}")

        Z_tr = stack_preds(oof, keys)
        Z_val = stack_preds(val_preds, keys)
        Z_test = stack_preds(test_preds, keys)

        # Train Meta-Model (GPU)
        stacker = train_stacker_fast(X_sp_tr, Z_tr, y_tr,
                                     X_sp_val, Z_val, y_val)

        # =======================================================
        # [修改] 保存所有组合模型 (名称中带 + 号的实验)
        # =======================================================
        if "+" in exp_name:
            # 将 + 替换为 _ (例如 LGB+RF -> Stacker_LGB_RF.pth)
            safe_name = exp_name.replace("+", "_")
            save_path = f"{BASE_PATH}/models/Stacker_{safe_name}.pth"
            torch.save(stacker.state_dict(), save_path)
            print(f"    [Saved] Stacker model saved to {save_path}")
        # =======================================================

        # Inference
        p_val, w_val, q_val = inference_stacker_fast(stacker, X_sp_val, Z_val)
        p_test, w_test, q_test = inference_stacker_fast(stacker, X_sp_test, Z_test)

        # Metrics
        val_res = evaluate(y_val, p_val, "VAL ")
        test_res = evaluate(y_test, p_test, "TEST")

        # === [新增] 计算并打印平均权重和不确定性 ===
        avg_weights = np.mean(w_test, axis=0)
        avg_uncertainty = np.mean(q_test[:, 2] - q_test[:, 0])  # 95%分位数 - 5%分位数

        print(f"    -> Avg Weights: {np.round(avg_weights, 3)} (Keys: {keys})")
        print(f"    -> Avg Uncertainty Width: {avg_uncertainty:.4f}")
        print("-" * 60)

        # Store
        res_dict = {
            "exp": exp_name,
            "keys": keys,
            "val": val_res,
            "test": test_res,
            "weights_sample": w_test[:5].tolist(),
            "uncertainty_sample": q_test[:5].tolist()
        }
        all_results[exp_name] = res_dict

        np.savez(f"{BASE_PATH}/results/{exp_name}_preds.npz",
                 val_pred=p_val, val_weights=w_val,
                 test_pred=p_test, test_weights=w_test, test_quantiles=q_test)

    # Summary
    print("\n===== Summary (Sorted by TEST RMSE) =====")
    ranking = sorted(all_results.items(), key=lambda x: x[1]["test"]["rmse"])
    for name, res in ranking:
        print(f"{name:10s} | TEST RMSE={res['test']['rmse']:.4f}")

    with open(f"{BASE_PATH}/results/ALL_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print(f"\nDone. Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()