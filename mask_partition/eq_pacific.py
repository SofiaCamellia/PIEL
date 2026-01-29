# -*- coding: utf-8 -*-
"""
equatorial_pacific INTEGRATED PIPELINE:
Geographic Filter -> 6 Base Models -> Spatial Stacker -> PINN
------------------------------------------------------------
1. Filters IAP data for equatorial_pacific Ocean using shapefile.
2. Trains 6 Base Models (LGB, RF, CB, XGB, ERT, KNN) with GroupKFold OOF.
3. Trains Spatial-Probabilistic Stacker on OOF predictions.
4. Trains PINN using Stacker output + Physics constraints.
"""

import os
import time
import warnings
import pickle
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon
import gsw

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
TARGET_OCEAN_REGION = "Equatorial Pacific Ocean"  # <<<--- 正式名称

# The 'REGION' variable for file paths is now generated automatically.
REGION = TARGET_OCEAN_REGION.lower().replace(" ", "_").replace("_ocean", "") # <<<--- 自动生成简称 "equatorial_pacific"

# REGION = "equatorial_pacific"  # <--- 在这里定义区域名称，方便以后改为 "north_atlantic" 等
# =======================
# PATHS (请根据实际情况修改)
# =======================
# 原始数据路径
DATA_PATH = "/home/bingxing2/home/scx7l1f/IAP_TSDO.npy"
# 海洋多边形 Shapefile 路径 (用于筛选北冰洋)
SHAPEFILE_PATH = "/home/bingxing2/home/scx7l1f/rec/ne_10m_poly_shp/ne_10m_geography_marine_polys.shp"
# 输出保存路径
BASE_PATH = f"/home/bingxing2/home/scx7l1f/rec/mask_partition/{REGION}"

# 创建目录
os.makedirs(f"{BASE_PATH}/models", exist_ok=True)
os.makedirs(f"{BASE_PATH}/results", exist_ok=True)
os.makedirs(f"{BASE_PATH}/data", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 5


# =====================================================================================
# PART 1: Data Filtering (equatorial_pacific Region)
# =====================================================================================
def create_equatorial_pacific_mask(shapefile_path, target_region):
    """
    Loads marine polygons and creates a single geometric mask for the equatorial_pacific Ocean region.
    """
    print("Step 1: Creating precise equatorial_pacific geographic mask...")
    try:
        marine_gdf = gpd.read_file(shapefile_path, engine="pyogrio")
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return None

    ocean_mapping = {
        "Pacific Ocean": [
            "North Pacific Ocean", "South Pacific Ocean", "Tasman Sea", "Philippine Sea",
            "Yellow Sea", "East China Sea", "Bering Sea", "South China Sea",
            "Bismarck Sea", "Solomon Sea", "Taiwan Strait", "Halmahera Sea", "Samar Sea",
            "Visayan Sea", "Coral Sea", "Bohol Sea", "Gulf of Alaska", "Sea of Okhotsk",
            "Bering Sea", "Norton Sound", "Bristol Bay", "Gulf of Anadyr'"
        ]
    }
    name_to_ocean = {name.upper(): ocean for ocean, names in ocean_mapping.items() for name in names}
    marine_gdf['main_basin'] = marine_gdf['name'].str.upper().map(name_to_ocean)

    # --- NEW: Logic to handle different types of regions ---
    # This block decides whether to do a simple filter or a complex split.
    geometries_for_mask = []

    # Case 1: The region is one of the Pacific subdivisions that require splitting.
    if target_region in ["North Pacific Ocean", "Equatorial Pacific Ocean", "South Pacific Ocean"]:
        print("Applying complex Pacific Ocean splitting rules...")
        pacific_polygons = marine_gdf[marine_gdf['main_basin'] == 'Pacific Ocean'].copy()
        if pacific_polygons.empty:
            print("No Pacific Ocean polygons found.")
            return None

        # Define latitude zones for splitting
        north_zone = Polygon([(-180, 10), (180, 10), (180, 90), (-180, 90)])
        equatorial_zone = Polygon([(-180, -10), (180, -10), (180, 10), (-180, 10)])
        south_zone = Polygon([(-180, -90), (180, -90), (180, -10), (-180, -10)])

        for _, row in pacific_polygons.iterrows():
            geom, name = row.geometry, row['name']

            # Special Rule: Coral Sea is entirely South Pacific
            if name == 'Coral Sea':
                if target_region == 'South Pacific Ocean':
                    geometries_for_mask.append(geom)
                continue  # Move to next polygon

            # Special Rule: South China Sea excludes its equatorial part
            if name == 'South China Sea':
                if target_region == 'North Pacific Ocean':
                    north_intersection = geom.intersection(north_zone)
                    if not north_intersection.is_empty: geometries_for_mask.append(north_intersection)
                elif target_region == 'South Pacific Ocean':
                    south_intersection = geom.intersection(south_zone)
                    if not south_intersection.is_empty: geometries_for_mask.append(south_intersection)
                continue  # Move to next polygon

            # General Rule for all other Pacific seas
            if target_region == 'North Pacific Ocean':
                north_intersection = geom.intersection(north_zone)
                if not north_intersection.is_empty: geometries_for_mask.append(north_intersection)
            elif target_region == 'Equatorial Pacific Ocean':
                equatorial_intersection = geom.intersection(equatorial_zone)
                if not equatorial_intersection.is_empty: geometries_for_mask.append(equatorial_intersection)
            elif target_region == 'South Pacific Ocean':
                south_intersection = geom.intersection(south_zone)
                if not south_intersection.is_empty: geometries_for_mask.append(south_intersection)

    # Case 2: The region is a simple ocean basin (like the original logic).
    else:
        print(f"Applying simple filter for '{target_region}'...")
        regional_gdf = marine_gdf[marine_gdf['main_basin'] == target_region].copy()
        if not regional_gdf.empty:
            geometries_for_mask = list(regional_gdf.geometry)

    # --- Final Mask Generation ---
    if not geometries_for_mask:
        print(f"Warning: No geometries found for the target region '{target_region}'.")
        return None

    # Combine all collected geometries into a single mask
    final_mask = gpd.GeoDataFrame(geometry=geometries_for_mask, crs="EPSG:4326").unary_union
    print(f"Geographic mask for '{target_region}' created successfully!")
    return final_mask


def get_equatorial_pacific_data():
    # 1. Load Raw
    print(f"Loading raw data from {DATA_PATH}...")
    raw = np.load(DATA_PATH, allow_pickle=True)

    # 2. Basic Filter (Year/Depth)
    year = raw[:, 2]
    depth = raw[:, 4]
    mask = (year >= 1980) & (depth <= 2000)
    data = raw[mask]

    # 3. Geo Filter
    # equatorial_pacific_mask = create_equatorial_pacific_mask(SHAPEFILE_PATH)
    equatorial_pacific_mask = create_equatorial_pacific_mask(SHAPEFILE_PATH, TARGET_OCEAN_REGION)
    print(f"Filtering {len(data)} points by equatorial_pacific mask...")

    df = pd.DataFrame(data, columns=['lat', 'lon', 'year', 'month', 'depth', 'temp', 'salt', 'oxygen'])
    # GeoPandas requires -180 to 180
    df['lon_conv'] = df['lon'].apply(lambda x: x - 360 if x > 180 else x)

    geometry = [Point(xy) for xy in zip(df['lon_conv'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    mask_gdf = gpd.GeoDataFrame(geometry=[equatorial_pacific_mask], crs="EPSG:4326")

    joined = gpd.sjoin(gdf, mask_gdf, how="inner", predicate='within')
    final_data = joined.drop(columns=['geometry', 'index_right', 'lon_conv']).to_numpy()

    print(f"equatorial_pacific Data Count: {len(final_data)}")
    np.save(f"{BASE_PATH}/data//{REGION}_filtered.npy", final_data)
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
    # Split training fold into Train/EarlyStopping based on time
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
        "lgb": lambda: lgb.LGBMRegressor(boosting_type='gbdt',
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
        verbosity=-1 ),
        "rf": lambda: RandomForestRegressor(n_estimators=40,
        random_state=RANDOM_SEED,
        max_depth=20,
        min_samples_leaf=50,
        n_jobs=8,  # Keep optimization
        verbose=2,
        oob_score=False,
        max_features='sqrt'),
        "cb": lambda: CatBoostRegressor(iterations=3500,
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
        devices='0'),  # Use GPU if available
        "xgb": lambda: XGBRegressor(n_estimators=3500,
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
        verbosity=1),
        "ert": lambda: ExtraTreesRegressor(n_estimators=40,
        max_depth=20,
        min_samples_leaf=50,
        max_features='sqrt',
        n_jobs=8,
        random_state=RANDOM_SEED,
        verbose=2),
        "knn": lambda: KNeighborsRegressor(n_neighbors=40,
        weights='distance',
        algorithm='auto',
        leaf_size=30,
        p=2,
        n_jobs=8)
    }


def run_base_models(X_tr, y_tr, X_val, X_test, scaler_knn):
    """
    1. Generate OOF predictions on X_tr (for Stacker training)
    2. Train full models on X_tr
    3. Predict on X_val and X_test
    """
    print("\n>>> Step 2: Training 6 Base Models (OOF + Full)...")
    models = get_model_factories()
    model_names = list(models.keys())
    n_tr = len(X_tr)

    # 1. OOF Generation
    oof_preds = {k: np.zeros(n_tr, dtype=np.float32) for k in model_names}
    years_tr = X_tr[:, 2].astype(int)
    gkf = GroupKFold(n_splits=K_FOLDS)

    # Pre-scale for KNN
    X_tr_sc = scaler_knn.transform(X_tr)

    print(f"   Generating OOF ({K_FOLDS} folds)...")
    for fold, (idx_fit, idx_val) in enumerate(gkf.split(X_tr, y_tr, groups=years_tr), 1):
        print(f"     Fold {fold}/{K_FOLDS}...")
        X_f, y_f = X_tr[idx_fit], y_tr[idx_fit]
        X_v, y_v = X_tr[idx_val], y_tr[idx_val]  # Validation fold

        # Split for Early Stopping
        X_inn, y_inn, X_es, y_es = make_time_ordered_es_split(X_f, y_f)

        # --- Train Loop ---
        # LGB
        m = models['lgb']()
        m.fit(X_inn, y_inn, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_preds['lgb'][idx_val] = m.predict(X_v)

        # XGB
        m = models['xgb']()
        m.fit(X_inn, y_inn, eval_set=[(X_es, y_es)], verbose=False)
        oof_preds['xgb'][idx_val] = m.predict(X_v)

        # CB
        m = models['cb']()
        m.fit(X_inn, y_inn, eval_set=(X_es, y_es), early_stopping_rounds=50, verbose=False)
        oof_preds['cb'][idx_val] = m.predict(X_v)

        # RF (No ES)
        m = models['rf']()
        m.fit(X_f, y_f)
        oof_preds['rf'][idx_val] = m.predict(X_v)

        # ERT (No ES)
        m = models['ert']()
        m.fit(X_f, y_f)
        oof_preds['ert'][idx_val] = m.predict(X_v)

        # KNN (Scaled)
        m = models['knn']()
        m.fit(X_tr_sc[idx_fit], y_f)  # Use scaled fit data
        oof_preds['knn'][idx_val] = m.predict(X_tr_sc[idx_val])

    # 2. Full Training & Inference
    print("   Training Full Models for Inference...")
    val_preds = {}
    test_preds = {}

    X_val_sc = scaler_knn.transform(X_val)
    X_test_sc = scaler_knn.transform(X_test)

    # Split Train for ES
    X_inn, y_inn, X_es, y_es = make_time_ordered_es_split(X_tr, y_tr)

    for name in model_names:
        print(f"     Fitting {name}...")
        model = models[name]()

        if name in ['lgb']:
            model.fit(X_inn, y_inn, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(50, verbose=False)])
            val_preds[name] = model.predict(X_val)
            test_preds[name] = model.predict(X_test)
        elif name in ['xgb']:
            model.fit(X_inn, y_inn, eval_set=[(X_es, y_es)], verbose=False)
            val_preds[name] = model.predict(X_val)
            test_preds[name] = model.predict(X_test)
        elif name in ['cb']:
            model.fit(X_inn, y_inn, eval_set=(X_es, y_es), early_stopping_rounds=50, verbose=False)
            val_preds[name] = model.predict(X_val)
            test_preds[name] = model.predict(X_test)
        elif name == 'knn':
            model.fit(X_tr_sc, y_tr)
            val_preds[name] = model.predict(X_val_sc)
            test_preds[name] = model.predict(X_test_sc)
        else:  # RF, ERT
            model.fit(X_tr, y_tr)
            val_preds[name] = model.predict(X_val)
            test_preds[name] = model.predict(X_test)
            # [新增] 模型保存目录

        model_dir = f"{BASE_PATH}/models"
            # --- [新增] 保存基模型到硬盘 ---
        if name == 'lgb':
            with open(f"{model_dir}/lgb_model.pkl", "wb") as f:
                pickle.dump(model, f)
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


def run_stacker(X_sp_tr, Z_tr, y_tr, X_sp_val, Z_val, y_val, X_sp_test, Z_test):
    print("\n>>> Step 3: Training Spatial Stacker...")

    # Prepare Tensors
    t_x_tr = torch.FloatTensor(X_sp_tr).to(DEVICE)
    t_z_tr = torch.FloatTensor(Z_tr).to(DEVICE)
    t_y_tr = torch.FloatTensor(y_tr).view(-1, 1).to(DEVICE)

    t_x_val = torch.FloatTensor(X_sp_val).to(DEVICE)
    t_z_val = torch.FloatTensor(Z_val).to(DEVICE)
    t_y_val = torch.FloatTensor(y_val).view(-1, 1).to(DEVICE)

    # Model
    model = SpatialProbabilisticStacker(5, 6).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.005)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    best_state = None
    batch_size = 65536

    # Train Loop
    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(t_x_tr), device=DEVICE)

        for i in range(0, len(t_x_tr), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pp, _, qp = model(t_x_tr[idx], t_z_tr[idx])
            loss = nn.MSELoss()(pp, t_y_tr[idx]) + 0.2 * pinball_loss(qp, t_y_tr[idx], [0.05, 0.5, 0.95])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vpp, _, vqp = model(t_x_val, t_z_val)
            vloss = nn.MSELoss()(vpp, t_y_val) + 0.2 * pinball_loss(vqp, t_y_val, [0.05, 0.5, 0.95])

        sch.step(vloss)
        if vloss < best_loss:
            best_loss = vloss
            best_state = model.state_dict()

        if epoch % 20 == 0:
            print(f"   Stacker Epoch {epoch}: Val Loss {vloss.item():.4f}")

    # Load Best & Predict
    model.load_state_dict(best_state)
    torch.save(best_state, f"{BASE_PATH}/models/best_stacker.pth")

    # Infer All
    def infer(X, Z):
        model.eval()
        p_list = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                tx = torch.FloatTensor(X[i:i + batch_size]).to(DEVICE)
                tz = torch.FloatTensor(Z[i:i + batch_size]).to(DEVICE)
                p, _, _ = model(tx, tz)
                p_list.append(p.cpu().numpy())
        return np.vstack(p_list).flatten()

    p_stack_tr = infer(X_sp_tr, Z_tr)
    p_stack_val = infer(X_sp_val, Z_val)
    p_stack_test = infer(X_sp_test, Z_test)  # <--- TEST prediction

    return p_stack_tr, p_stack_val, p_stack_test, model


# =====================================================================================
# PART 4: PINN
# =====================================================================================
class AvgPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: Stacker Pred (1) + Lat, Depth, Temp, Salt (4)
        self.net = nn.Sequential(
            nn.Linear(5, 256), nn.BatchNorm1d(256), nn.Tanh(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.Tanh(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, stack_pred, phys_feats):
        x = torch.cat([stack_pred, phys_feats], dim=1)
        return self.net(x).squeeze()


def physical_constraint_loss(y_pred, features):
    # features: Lat, Depth, Temp, Salt
    lat, depth, temp, salt = features[:, 0], features[:, 1], features[:, 2], features[:, 3]

    # Approx Pressure (simplified from gsw for speed/diff)
    pressure = depth  # Simple approximation or use gsw pre-calculated

    # DO Saturation (Weiss-Doxy)
    T_k = temp + 273.15
    A1, A2, A3, A4 = -177.7888, 255.5907, 146.4813, -22.2040
    B1, B2, B3 = -0.037362, 0.016504, -0.0020564
    ln_DO = (A1 + A2 * (100 / T_k) + A3 * torch.log(T_k / 100) + A4 * (T_k / 100) +
             salt * (B1 + B2 * (T_k / 100) + B3 * ((T_k / 100) ** 2)))
    DO_sat = torch.exp(ln_DO) * 44.66 * (1 + (0.032 * pressure / 1000))

    # Constraints
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
        # Physics features: Lat(0), Depth(4), Temp(5), Salt(6)
        self.p_raw = torch.tensor(X_raw[:, [0, 4, 5, 6]], dtype=torch.float32)
        self.p_sc = torch.tensor(scaler.transform(X_raw[:, [0, 4, 5, 6]]), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return self.s[i], self.p_sc[i], self.p_raw[i], self.y[i]


def run_pinn(p_stack_tr, X_tr, y_tr, p_stack_val, X_val, y_val, p_stack_test, X_test, y_test):
    print("\n>>> Step 4: Training PINN...")

    # Scaler for PINN Inputs
    pinn_scaler = StandardScaler()
    pinn_scaler.fit(X_tr[:, [0, 4, 5, 6]])

    # Loaders
    bs = 65536
    tr_loader = DataLoader(PINNDataset(p_stack_tr, X_tr, y_tr, pinn_scaler), batch_size=bs, shuffle=True)
    val_loader = DataLoader(PINNDataset(p_stack_val, X_val, y_val, pinn_scaler), batch_size=bs)
    te_loader = DataLoader(PINNDataset(p_stack_test, X_test, y_test, pinn_scaler), batch_size=bs)

    # Train
    model = AvgPINN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.005)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5)
    scaler = GradScaler()

    w_phys, w_teach = 0.5, 0.4
    best_loss = float('inf')

    for epoch in range(50):
        model.train()
        t_loss = 0

        # Dynamic Weights
        if epoch > 0 and epoch % 5 == 0:
            w_phys = min(1.0, w_phys + 0.1)
            w_teach = max(0.1, w_teach - 0.05)

        for s, p_sc, p_raw, y in tr_loader:
            s, p_sc, p_raw, y = s.to(DEVICE), p_sc.to(DEVICE), p_raw.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()

            with autocast():
                pred = model(s, p_sc)
                l_mse = nn.MSELoss()(pred, y.squeeze())
                l_phy = physical_constraint_loss(pred, p_raw)
                l_fid = nn.MSELoss()(pred, s.squeeze())
                loss = l_mse + w_phys * l_phy + w_teach * l_fid

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            t_loss += loss.item()

        # Validation
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for s, p_sc, _, y in val_loader:
                s, p_sc, y = s.to(DEVICE), p_sc.to(DEVICE), y.to(DEVICE)
                with autocast():
                    pred = model(s, p_sc)
                    v_loss += nn.MSELoss()(pred, y.squeeze()).item()

        v_loss /= len(val_loader)
        sch.step(v_loss)
        print(f"   PINN Epoch {epoch}: Val MSE {v_loss:.4f} (Phys W: {w_phys:.1f})")

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), f"{BASE_PATH}/models/best_pinn.pth")

    # Final Prediction
    model.load_state_dict(torch.load(f"{BASE_PATH}/models/best_pinn.pth"))
    model.eval()
    preds = []
    with torch.no_grad():
        for s, p_sc, _, _ in te_loader:
            s, p_sc = s.to(DEVICE), p_sc.to(DEVICE)
            preds.append(model(s, p_sc).float().cpu().numpy())

    return np.concatenate(preds)


# =====================================================================================
# Main
# =====================================================================================
def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mb = np.mean(y_pred - y_true)
    print(f"{name} | RMSE: {rmse:.4f} | R2: {r2:.4f} | MAE: {mae:.4f} | MB: {mb:.4f}")
    print(f"预测值范围: {y_pred.min():.2f} ~ {y_pred.max():.2f}")
    return rmse


def main():
    t0 = time.time()
    print("===== equatorial_pacific INTEGRATED PIPELINE STARTED =====")

    # 1. Data
    data = get_equatorial_pacific_data()
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32)
    years = X[:, 2].astype(int)

    (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = year_based_split(X, y, years)
    print(f"Split: Tr:{len(X_tr)}, Val:{len(X_val)}, Te:{len(X_test)}")

    # 2. Base Models
    # Scale for KNN inside the function
    scaler_knn = StandardScaler()
    scaler_knn.fit(X_tr)  # Fit on global train for consistency

    oof_preds, val_preds, test_preds = run_base_models(X_tr, y_tr, X_val, X_test, scaler_knn)
    # === [新增] 保存 OOF 和 验证/测试集预测结果 ===
    print(f"   Saving OOF and Base Preds to {BASE_PATH}/results/ ...")
    np.savez(f"{BASE_PATH}/results/oof_base_preds_ALL6.npz", **oof_preds)
    np.savez(f"{BASE_PATH}/results/val_base_preds_ALL6.npz", **val_preds)
    np.savez(f"{BASE_PATH}/results/test_base_preds_ALL6.npz", **test_preds)

    # 3. Stacker
    # Spatial Features: Lat(0), Lon(1), Depth(4), Temp(5), Salt(6)
    SP_IDXS = [0, 1, 4, 5, 6]
    sp_scaler = StandardScaler()
    X_sp_tr = sp_scaler.fit_transform(X_tr[:, SP_IDXS])
    X_sp_val = sp_scaler.transform(X_val[:, SP_IDXS])
    X_sp_test = sp_scaler.transform(X_test[:, SP_IDXS])

    # Stack inputs (Column stack 6 models)
    Z_tr = np.column_stack([oof_preds[k] for k in oof_preds])
    Z_val = np.column_stack([val_preds[k] for k in val_preds])
    Z_test = np.column_stack([test_preds[k] for k in test_preds])

    p_stack_tr, p_stack_val, p_stack_test, stacker_model = run_stacker(
        X_sp_tr, Z_tr, y_tr, X_sp_val, Z_val, y_val, X_sp_test, Z_test
    )
    evaluate(y_test, p_stack_test, "Stacker (Test)")

    # 4. PINN
    p_pinn_test = run_pinn(
        p_stack_tr, X_tr, y_tr,
        p_stack_val, X_val, y_val,
        p_stack_test, X_test, y_test
    )

    # 5. Final Metrics
    print("\n===== FINAL RESULTS =====")
    evaluate(y_test, p_stack_test, "Stacker Only")
    evaluate(y_test, p_pinn_test, "PINN (Final)")

    np.savez(f"{BASE_PATH}/results/final_predictions.npz",
             stacker=p_stack_test, pinn=p_pinn_test, y_true=y_test)

    print(f"Total Time: {(time.time() - t0) / 60:.1f} mins")


if __name__ == "__main__":
    main()