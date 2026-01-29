# 最后选用的是谱聚类13

import matplotlib
matplotlib.use('TkAgg')  # 如在无界面环境报错可取消注释

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20  # 标题稍微大一点


def _pick_var(ds, candidates):
    """从一组候选名字里挑第一个存在于 ds.variables 的变量名。"""
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _shift_lon_0_360_to_m180_180(lon_1d, data_2d, lon_dim=-1):
    """
    若 lon 为 0..360，转换到 -180..180 并按经度排序，同时重排 data_2d 对应的经度维。
    lon_1d: (nlon,)
    data_2d: (..., nlon) 这里假设经度维是最后一维（或通过 lon_dim 指定）
    """
    lon_new = ((lon_1d + 180) % 360) - 180
    order = np.argsort(lon_new)
    lon_sorted = lon_new[order]

    data_sorted = np.take(data_2d, order, axis=lon_dim)
    return lon_sorted, data_sorted


def main(nc_path="ocean_partition_mask_kmeans.nc", out_png=None):
    ds = xr.open_dataset(nc_path)

    # 1) 变量名尽量自动识别（你文件里是 lat/lon/partition 的话会直接命中）
    part_name = _pick_var(ds, ["partition", "PARTITION", "mask", "region", "basin"])
    if part_name is None:
        raise KeyError("找不到 partition 变量：请把文件里真正的变量名替换到候选列表中。")

    lat_name = _pick_var(ds, ["lat", "latitude", "LAT", "nav_lat", "y"])
    lon_name = _pick_var(ds, ["lon", "longitude", "LON", "nav_lon", "x"])
    if lat_name is None or lon_name is None:
        raise KeyError("找不到 lat/lon 变量：请检查文件变量名（ncdump -h）并修改候选列表。")

    p = ds[part_name]
    lat = ds[lat_name]
    lon = ds[lon_name]

    # 转成 numpy
    pvals = p.values
    # 把 0 设为缺测：不绘制
    pvals = pvals.astype(float)  # 确保能放 NaN
    pvals[pvals == 0] = np.nan
    # pvals[pvals == 4] = np.nan
    # pvals[pvals == 10] = np.nan
    # pvals[pvals == ] = np.nan
    latvals = lat.values
    lonvals = lon.values



    # 2) 处理经纬度网格：支持 1D 或 2D
    if latvals.ndim == 1 and lonvals.ndim == 1:
        # 可选：如果 lon 是 0..360，转成 -180..180 并重排 partition
        if np.nanmin(lonvals) >= 0 and np.nanmax(lonvals) > 180:
            # 确认 partition 的经度维确实对应 lon 的长度（常见为最后一维）
            if pvals.shape[-1] == lonvals.shape[0]:
                lonvals, pvals = _shift_lon_0_360_to_m180_180(lonvals, pvals, lon_dim=-1)
            else:
                # 如果不是最后一维，你需要按实际维度改 lon_dim
                pass

        Lon2d, Lat2d = np.meshgrid(lonvals, latvals)
    else:
        # 2D 经纬度（比如 curvilinear grid）
        Lon2d, Lat2d = lonvals, latvals

    regions_to_remove = [
        (30, 45, 63, 69),  # Box 1: 白海
        (26, 42, 40, 47),  # Box 2: 黑海
        (46, 56, 36, 48),  # Box 3: 里海
        # (20, 30, 58, 62),  # 可选：波罗的海
        (-120, -60, 50, 70),
        (-55,-48,45,52),
        (-60,-50,63.5,70)
    ]

    for (min_lon, max_lon, min_lat, max_lat) in regions_to_remove:
        # 此时 Lon2d 和 Lat2d 已经存在了，可以安全使用
        mask_region = (Lon2d >= min_lon) & (Lon2d <= max_lon) & \
                      (Lat2d >= min_lat) & (Lat2d <= max_lat)

        # 修改 pvals (原始数据)，这样后面的步骤会自动忽略它们
        pvals[mask_region] = np.nan

    # 3) 获取实际有哪些 partition 值（去掉 NaN）
    flat = pvals.ravel()
    if np.issubdtype(flat.dtype, np.floating):
        flat = flat[np.isfinite(flat)]
    ids = np.unique(flat).astype(int)
    ids = np.sort(ids)

    # 4) 把“原始 partition id”映射到 0..n-1，方便做离散色标（颜色等距）
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    mapped = np.full_like(pvals, fill_value=np.nan, dtype=float)

    # 用循环赋值，兼容各种 dtype
    for pid, idx in id_to_idx.items():
        mapped[pvals == pid] = idx

    nclass = len(ids)
    if nclass == 0:
        raise ValueError("partition 没有有效值（可能全是缺测）。")



    # 5) 离散 colormap + norm（每一类一个颜色块）
    # cmap = plt.cm.get_cmap("tab20", nclass) if nclass <= 20 else plt.cm.get_cmap("hsv", nclass)
    from matplotlib.colors import ListedColormap

    # colors = list(plt.get_cmap("Dark2").colors) + list(plt.get_cmap("Set2").colors)
    # cmap = ListedColormap(colors[:nclass])

    nclass = 13
    base = plt.get_cmap("tab20").colors  # 20 个离散颜色

    # 先取偶数位(更“主色”)，再补奇数位(更“浅色”)，保证前 13 个尽量差异大
    idx = list(range(0, 20, 2)) + list(range(1, 20, 2))
    colors = [base[i] for i in idx[:nclass]]

    cmap = ListedColormap(colors)

    bounds = np.arange(-0.5, nclass + 0.5, 1)
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    # 6) 开始用 cartopy 画图
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=proj)
    ax.set_global()

    # 地理要素
    ax.add_feature(cfeature.LAND, zorder=0, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.6)

    # 分区图（分类/离散）
    im = ax.pcolormesh(
        Lon2d, Lat2d, mapped,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto"
    )

    # 网格线（可关）
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # colorbar：刻度显示“原始 partition id”
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)
    cbar.set_ticks(np.arange(nclass))
    cbar.set_ticklabels([str(i) for i in ids])
    cbar.set_label("partition id")

    ax.set_title(f"Partition categorical map ({part_name})  |  classes={nclass}")

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_png}")

    plt.show()


if __name__ == "__main__":
    # main("ocean_partition_mask_kmeans.nc", out_png="partition_map.png") #kmeans的结果不好，不用
    main("ocean_pujulei_mask.nc", out_png="partition_map.png") #聚类数13