import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def main():
    # ================= 配置路径与参数 =================
    # 输入数据目录 (EN数据存放处)
    data_dir = "/home/bingxing2/home/scx7l1f/processed"

    # GLODAP数据路径
    glodap_path = "/home/bingxing2/home/scx7l1f/processed/glodap_oxygen_array_all.npy"

    # 最终输出路径
    output_path = "/home/bingxing2/home/scx7l1f/processed/global_matched_en_glodap.npy"

    # 年份范围 (1980 - 2025)
    years = range(1980, 2026)

    # ================= 第一步：加载 GLODAP 数据 =================
    print(f"正在加载 GLODAP 数据: {glodap_path}")
    if not os.path.exists(glodap_path):
        print(f"错误: GLODAP 文件不存在: {glodap_path}")
        return

    glodap_data = np.load(glodap_path)
    # GLODAP 列定义: lat(0), lon(1), year(2), month(3), day(4), depth(5), oxygen(6), pressure(7)
    glodap_columns = ['lat', 'lon', 'year', 'month', 'day', 'depth', 'oxygen', 'pressure']
    glodap_df = pd.DataFrame(glodap_data, columns=glodap_columns)

    # 优化：为了加速，将 year 和 month 转为整数类型（如果不是的话）
    glodap_df['year'] = glodap_df['year'].astype(int)
    glodap_df['month'] = glodap_df['month'].astype(int)

    print(f"GLODAP 数据加载完成，形状: {glodap_df.shape}")

    # 用于收集最终匹配结果的列表
    all_matched_records = []

    # ================= 第二步：循环处理 EN 数据并匹配 =================
    print("开始处理 EN 数据并进行匹配...")

    for year in tqdm(years, desc="处理年份"):
        # 1. 构建当年文件名
        file_path = os.path.join(data_dir, f"processed_data_{year}.npy")

        # 如果文件不存在则跳过
        if not os.path.exists(file_path):
            continue

        # 检查 GLODAP 中是否有这一年的数据，如果没有，无需加载 EN 数据，直接跳过
        glodap_year_df = glodap_df[glodap_df['year'] == year]
        if glodap_year_df.empty:
            continue

        try:
            # 2. 加载当年的 EN 数据
            # EN 列定义: LATITUDE(0), LONGITUDE(1), YEAR(2), MONTH(3), DEPH_CORRECTED(4), TEMP(5), PSAL_CORRECTED(6)
            en_data_raw = np.load(file_path, allow_pickle=True)

            # 3. 筛选 EN 数据 (Filter)
            # 筛选条件：
            # - 全球数据 (不限制经度)
            # - 深度 < 2000
            # - 温度 -2.5 ~ 40.0
            # - 盐度 2.0 ~ 42.0
            filter_mask = (
                    (en_data_raw[:, 5] >= -2.5) & (en_data_raw[:, 5] <= 40.0) &  # TEMP
                    (en_data_raw[:, 6] >= 2.0) & (en_data_raw[:, 6] <= 42.0) &  # PSAL
                    (en_data_raw[:, 4] < 2000)  # DEPTH
            )

            en_data_filtered = en_data_raw[filter_mask]

            if len(en_data_filtered) == 0:
                continue

            # 转换为 DataFrame 方便后续匹配操作
            en_columns = ['lat', 'lon', 'year', 'month', 'depth', 'temp', 'sal']
            en_df = pd.DataFrame(en_data_filtered, columns=en_columns)

            # 4. 开始匹配 (Match)
            # 此时 en_df 和 glodap_year_df 都是同一年份的数据

            # 遍历该年存在的月份
            common_months = set(en_df['month'].unique()) & set(glodap_year_df['month'].unique())

            for month in common_months:
                en_month_df = en_df[en_df['month'] == month]
                glodap_month_df = glodap_year_df[glodap_year_df['month'] == month]

                # 遍历 GLODAP 数据寻找匹配的 EN 点
                # 使用 itertuples 提高遍历速度
                for g_row in glodap_month_df.itertuples(index=False):
                    # g_row 属性: lat, lon, year, month, day, depth, oxygen, pressure

                    # 4.1 深度完全匹配
                    # 注意：EN数据 depth 是第4列(index 4)，在df中名为 'depth'
                    # 筛选同深度的 EN 点
                    depth_candidates = en_month_df[en_month_df['depth'] == g_row.depth]

                    if depth_candidates.empty:
                        continue

                    # 4.2 经纬度距离筛选 (阈值 0.001)
                    # 计算绝对距离
                    lat_diff = np.abs(depth_candidates['lat'] - g_row.lat)
                    lon_diff = np.abs(depth_candidates['lon'] - g_row.lon)

                    # 找出满足条件的索引
                    match_mask = (lat_diff < 0.001) & (lon_diff < 0.001)
                    matched_candidates = depth_candidates[match_mask]

                    if not matched_candidates.empty:
                        best_match = None

                        # 4.3 冲突解决：如果有多个匹配，选距离最近的
                        if len(matched_candidates) > 1:
                            # 重新获取对应的 diff 进行求和
                            total_diff = lat_diff[match_mask] + lon_diff[match_mask]
                            best_idx = total_diff.idxmin()
                            best_match = matched_candidates.loc[best_idx]
                        else:
                            best_match = matched_candidates.iloc[0]

                        # 4.4 拼接数据
                        # 顺序: [EN 7列] + [GLODAP 8列]
                        # EN: lat, lon, year, month, depth, temp, sal
                        # GLODAP: lat, lon, year, month, day, depth, oxygen, pressure

                        # 提取 EN 值
                        en_values = [
                            best_match['lat'], best_match['lon'], best_match['year'],
                            best_match['month'], best_match['depth'], best_match['temp'],
                            best_match['sal']
                        ]

                        # 提取 GLODAP 值
                        glodap_values = [
                            g_row.lat, g_row.lon, g_row.year, g_row.month,
                            g_row.day, g_row.depth, g_row.oxygen, g_row.pressure
                        ]

                        # 合并并添加到结果列表
                        all_matched_records.append(en_values + glodap_values)

        except Exception as e:
            print(f"处理年份 {year} 时出错: {e}")
            continue

    # ================= 第三步：保存结果 =================
    if all_matched_records:
        print("正在转换并保存数据...")
        final_data = np.array(all_matched_records)

        print(f"总计匹配数据点数量: {len(final_data)}")
        print(f"数据列数: {final_data.shape[1]} (预期 15 列)")

        # 保存
        np.save(output_path, final_data)
        print(f"已保存至: {output_path}")

        # 打印列名参考
        print("保存的数据列顺序如下:")
        print("Indices 0-6 (EN): lat, lon, year, month, depth, temp, sal")
        print("Indices 7-14 (GLODAP): lat, lon, year, month, day, depth, oxygen, pressure")

    else:
        print("未找到任何匹配的数据点。")


if __name__ == "__main__":
    main()