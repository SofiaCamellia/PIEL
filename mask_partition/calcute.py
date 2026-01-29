import numpy as np

# 数据格式: { "海域名称": (RMSE_Stacker, RMSE_PINN, Test_Count) }
ocean_data = {
    "Equatorial Pacific": (10.5237, 10.3589, 978945),
    "North Atlantic": (9.4060, 9.2739, 1757390),
    "Indian Ocean": (8.0363, 9.1877, 1856344),
    "North Pacific": (9.8155, 9.5496, 1880733),
    "South Atlantic": (7.4741, 7.0743, 1350783),
    "South Pacific": (8.3656, 8.6489, 1245662),
    "Southern Ocean": (6.9683, 7.3951, 793251),
    "Equatorial Atlantic": (8.6930, 8.9007, 260212),
    "Arctic Ocean": (8.0003, 7.8568, 440789)
}


def calculate_weighted_rmse(data_dict):
    total_n = sum(item[2] for item in data_dict.values())

    stacker_se = sum((item[0] ** 2) * item[2] for item in data_dict.values())
    pinn_se = sum((item[1] ** 2) * item[2] for item in data_dict.values())

    global_stacker = np.sqrt(stacker_se / total_n)
    global_pinn = np.sqrt(pinn_se / total_n)

    return global_stacker, global_pinn, total_n


s_rmse, p_rmse, total_samples = calculate_weighted_rmse(ocean_data)

print(f"Total Test Samples: {total_samples}")
print(f"Global Weighted Stacker RMSE: {s_rmse:.4f}")
print(f"Global Weighted PINN RMSE: {p_rmse:.4f}")

# Total Test Samples: 10564109
# Global Weighted Stacker RMSE: 8.7781
# Global Weighted PINN RMSE: 8.8979