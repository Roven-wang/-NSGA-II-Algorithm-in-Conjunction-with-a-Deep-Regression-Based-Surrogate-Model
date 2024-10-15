import numpy as np
import pandas as pd

# 从Excel文件中读取数据
file_path = r'F:\desktop\evolution_results.xlsx'
data = pd.read_excel(file_path)

# 定义统计量（如均值）
def statistic(x):
    return np.mean(x)

# 非重叠块引导采样函数
def nbb_resample(data, block_size):
    n = len(data)
    num_blocks = n // block_size
    # 创建非重叠块
    blocks = [data[i * block_size:(i + 1) * block_size] for i in range(num_blocks)]
    # 从非重叠块中进行重采样
    sampled_blocks = np.random.choice(num_blocks, num_blocks, replace=True)
    resampled_data = np.concatenate([blocks[i] for i in sampled_blocks])
    return resampled_data

# 计算单列数据的Bootstrap标准误和Jackknife-after-Bootstrap标准误
def compute_errors(data_column, B=500, block_size=20):
    # 生成Bootstrap样本
    bootstrap_samples = np.array([nbb_resample(data_column, block_size) for _ in range(B)])
    bootstrap_statistics = np.array([statistic(sample) for sample in bootstrap_samples])

    # 计算Bootstrap标准误
    se_boot = np.std(bootstrap_statistics, ddof=1)

    # Jackknife-after-Bootstrap
    data_array = data_column.to_numpy()  # 将数据列转换为Numpy数组
    n = len(data_array)
    jackknife_statistics = np.array([
        statistic(np.delete(data_array, i)) for i in range(n)
    ])
    se_jack = np.sqrt((n - 1) / n * np.sum((jackknife_statistics - np.mean(jackknife_statistics)) ** 2))

    return se_boot, se_jack

# 批量处理所有列
results = {}
for column in data.columns:
    se_boot, se_jack = compute_errors(data[column])
    results[column] = {'Bootstrap标准误': se_boot, 'Jackknife-after-Bootstrap标准误': se_jack}

# 打印结果
for column, errors in results.items():
    '''print(f"{column}:")
    print(f"  Bootstrap标准误: {errors['Bootstrap标准误']}")
    print(f"  Jackknife-after-Bootstrap标准误: {errors['Jackknife-after-Bootstrap标准误']}")'''
    print(f" {errors['Bootstrap标准误']}",f"{errors['Jackknife-after-Bootstrap标准误']}")
