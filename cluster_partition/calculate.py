import numpy as np

# 以 Class 分类数据为例 (从你的数据汇总表中提取)
data = {
    # "ClassID": (RMSE, Count)
    "Class1": (9.3822, 10519270),
    "Class2": (10.4502, 27228),
    "Class3": (6.6585, 3442432),
    # "Class4": (35.9202, 165048),
    "Class5": (10.2473, 4623530),
    "Class6": (21.7704, 311543),
    "Class7": (9.5249, 2243179),
    "Class8": (8.1049, 2988949),
    "Class9": (7.1310, 4488154),
    "Class10": (7.9126, 1945065),
    "Class11": (7.5054, 5384974),
    "Class12": (8.3843, 8383088),
    "Class13": (10.2086, 1442009),
}


def calculate_global_rmse(metrics_dict):
    total_squared_error = 0
    total_count = 0

    for name, (rmse, count) in metrics_dict.items():
        total_squared_error += (rmse ** 2) * count
        total_count += count

    global_rmse = np.sqrt(total_squared_error / total_count)
    return global_rmse, total_count


# 计算结果
global_stacker_rmse, total_samples = calculate_global_rmse(data)
print(f"全球加权平均 RMSE: {global_stacker_rmse:.4f}")
print(f"总样本量: {total_samples}")

# 全球加权平均 RMSE: 8.7512
# 总样本量: 45799421