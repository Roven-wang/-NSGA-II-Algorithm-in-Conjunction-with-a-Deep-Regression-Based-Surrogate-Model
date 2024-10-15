import pandas as pd
import numpy as np
from scipy.stats import linregress

# 从Excel文件读取数据
data = pd.read_excel(r"F:/desktop/NSE.xlsx", header=None)

# 提取观测值列
observed_data = data.iloc[:, 0]

# 提取模拟数据列（从第2列开始）
simulated_data = data.iloc[:, 1:]

# 初始化结果表格
result_df = pd.DataFrame(index=range(1, simulated_data.shape[1] + 1),
                          columns=["NSE", "R²", "KGE", "PBIAS", "bR²", "MNS", "RSR", "SSQR", "RMSE" ,"D" , "LOGE"])


# 计算 NSE
def calculate_nse(observed, simulated):
    # 计算观测值的平均值
    mean_observed = np.mean(observed)

    # 计算 NSE
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

# 计算KGE
def calculate_kge(observed, simulated):
    # 计算相关性系数
    correlation = np.corrcoef(observed, simulated)[0, 1]

    # 计算偏差系数
    bias = np.mean(simulated) / np.mean(observed)

    # 计算变异系数
    std_observed = np.std(observed)
    std_simulated = np.std(simulated)

    # 计算KGE
    kge = 1 - np.sqrt((correlation - 1) ** 2 + (bias - 1) ** 2 + (std_simulated / std_observed - 1) ** 2)
    return kge

#计算PBIAS
def calculate_pbias(observed, simulated):
    pbias = ((np.sum(observed - simulated) / np.sum(observed)) * 100)
    return pbias

#计算LOGE
def calculate_loge(observed, simulated):
    loge = np.sqrt(np.sum((np.log10(observed / simulated)) ** 2) / len(observed))
    return loge

# 循环计算每一列模拟数据的性能指标
for i in range(simulated_data.shape[1]):
    simulated_series = simulated_data.iloc[:, i]

    # 计算NSE
    nse_value = calculate_nse(observed_data, simulated_series)

    # 计算 R²
    mean_observed = np.mean(observed_data)
    mean_simulated = np.mean(simulated_series)
    total_variance = sum((x - mean_observed) ** 2 for x in observed_data) * sum(
        (y - mean_simulated) ** 2 for y in simulated_series)
    residual_variance = sum(
        (z - mean_observed) * (q - mean_simulated) for z, q in zip(observed_data, simulated_series)) ** 2
    r_squared = residual_variance / total_variance

    # 计算 KGE
    kge_value = calculate_kge(observed_data, simulated_series)

    # 计算 PBIAS
    pbias_value = calculate_pbias(observed_data, simulated_series)

    # 计算 bR²
    correlation_coefficient = np.corrcoef(observed_data, simulated_series)[0, 1]
    result = linregress(simulated_series, observed_data)
    slope_b = result.slope
    if abs(slope_b) <= 1:
        br2 = abs(slope_b) * correlation_coefficient ** 2
    else:
        br2 = abs(slope_b) ** (-1) * correlation_coefficient ** 2

    # 计算 MNS
    numerator = np.sum(np.abs(np.array(observed_data) - np.array(simulated_series)))
    denominator = np.sum(np.abs(np.array(observed_data) - mean_observed))
    mns = 1 - (numerator / denominator)

    # 计算 RSR
    rmse = np.sqrt(((simulated_series - observed_data) ** 2).mean())
    std_dev_observed = np.std(observed_data)
    rsr = rmse / std_dev_observed

    # 计算 SSQR
    sorted_simulated = np.sort(simulated_series)[::-1]
    sorted_observed = np.sort(observed_data)[::-1]
    ssqr = np.sum((sorted_simulated - sorted_observed) ** 2) / len(observed_data)

    # 计算RMSE
    rmse = np.sqrt(((simulated_series - observed_data) ** 2).mean())

    # 计算一致性d
    numerator = np.sum((observed_data - simulated_series) ** 2)
    denominator = np.sum((np.abs(observed_data - mean_observed) + np.abs(simulated_series - mean_observed)) ** 2)

    consistency_d = 1 - (numerator / denominator)

    # 计算LOGE
    loge_value = calculate_loge(observed_data, simulated_series)

    # 将结果存入结果表格
    result_df.loc[i+1] = [nse_value, r_squared, kge_value, pbias_value, br2, mns, rsr, ssqr, rmse, consistency_d, loge_value]

# 将结果保存到Excel文件
result_df.to_excel("F:\desktop\指标1.xlsx", index=False)
