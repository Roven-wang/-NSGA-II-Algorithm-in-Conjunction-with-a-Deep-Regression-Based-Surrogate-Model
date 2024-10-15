import numpy as np
import pandas as pd
from scipy.stats import rankdata

def normalize(matrix, indicator_types):
    """
    对原始矩阵进行正向化处理

    Parameters:
    - matrix: 原始矩阵
    - indicator_types: 指标类型列表，包含每个指标的类型，如['max', 'min', 'range', 'mid', ...]

    Returns:
    - normalized_matrix: 正向化后的矩阵
    """
    normalized_matrix = np.copy(matrix)

    for j, indicator_type in enumerate(indicator_types):
        if indicator_type == 'max':
            # 不需要额外处理，已经是极大型指标
            pass
        elif indicator_type == 'min':
            # 对于极小型指标，进行转化
            max_value = np.max(matrix[:, j])
            normalized_matrix[:, j] = max_value - matrix[:, j]
        elif indicator_type == 'range':
            # 范围型指标，进行线性标准化
            min_value = np.min(matrix[:, j])
            max_value = np.max(matrix[:, j])
            normalized_matrix[:, j] = (matrix[:, j] - min_value) / (max_value - min_value)
        elif indicator_type == 'mid':
            # 0为中间值最佳值，需要根据需要修改
            normalized_matrix[:, j] = 1 - np.abs(matrix[:, j] - 0) / np.max(np.abs(matrix[:, j] - 0))
        # 其他类型的指标可以根据需要进行处理

    return normalized_matrix

def normalize_minmax(matrix):
    """
    将矩阵的每个列标准化到 [0, 1] 区间，处理负值

    Parameters:
    - matrix: 原始矩阵

    Returns:
    - matrix: 标准化后的矩阵
    """
    min_value = np.min(matrix, axis=0)
    max_value = np.max(matrix, axis=0)
    matrix = (matrix - min_value) / (max_value - min_value + 1e-10)  # 避免除以零
    return matrix


def standardize(normalized_matrix):
    # 对矩阵进行标准化
    squared_sum = np.sum(normalized_matrix**2, axis=0)
    # 处理元素为0的情况，将0替换为1
    squared_sum[squared_sum == 0] = 1
    standardized_matrix = normalized_matrix / np.sqrt(squared_sum)

    return standardized_matrix


def replace_zeros(matrix):
    # 将矩阵中的0替换为1
    matrix = np.where(matrix == 0, 1, matrix)
    return matrix


def entropy(matrix):
    # 处理空值
    matrix = np.nan_to_num(matrix)
    # 计算熵值
    normalized_matrix = matrix / (np.sum(matrix, axis=0))

    # 将接近零的数值替换为一个很小的正数，避免log(0)问题
    normalized_matrix = np.where(normalized_matrix == 0, 1e-10, normalized_matrix)

    entropy_values = -np.sum(normalized_matrix * np.log(normalized_matrix), axis=0) / np.log(len(matrix))

    return entropy_values



def dissimilarity_coefficient(entropy_values):
    # 计算差异系数
    return 1 - entropy_values

def entropy_weight(entropy_values):
    """
    计算熵权

    Parameters:
    - entropy_values: 每个指标的熵值

    Returns:
    - entropy_weights: 每个指标的熵权
    """
    dissimilarity_coefficients = 1 - entropy_values
    total_dissimilarity = np.sum(dissimilarity_coefficients)
    entropy_weights = dissimilarity_coefficients / total_dissimilarity
    return entropy_weights

def topsis(matrix, entropy_weights):
    """
    进行TOPSIS算法

    Parameters:
    - matrix: 正向化后的矩阵
    - entropy_weights: 每个指标的熵权

    Returns:
    - rankings: 评价对象的排名
    """
    normalized_matrix = matrix * entropy_weights

    # 计算正理想解和负理想解
    ideal_positive = np.max(normalized_matrix, axis=0)
    ideal_negative = np.min(normalized_matrix, axis=0)

    # 计算到理想解和负理想解的距离
    distance_to_positive = np.sqrt(np.sum((normalized_matrix - ideal_positive)**2 * entropy_weights, axis=1))
    distance_to_negative = np.sqrt(np.sum((ideal_negative - normalized_matrix)**2 * entropy_weights, axis=1))

    # 计算综合得分
    performance_score = distance_to_negative / (distance_to_negative + distance_to_positive)

    # 对得分进行排名
    rankings = np.argsort(performance_score)[::-1] + 1  # 从高到低排名

    return rankings


# 读取 Excel 文件
excel_path = r'F:\desktop\TOPSIS.xlsx'  # 替换为你的 Excel 文件路径
df = pd.read_excel(excel_path, header=None)

# 评价指标类型列表，这里示例为 3 个指标，具体根据实际情况修改
indicator_types = ['max','max','max','mid','max','max','min','min','max','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min','min']
# 提取矩阵数据
original_matrix = df.values

# 正向化处理
normalized_matrix = normalize(original_matrix, indicator_types)

normalized_matrix = normalize_minmax(normalized_matrix)
# 标准化处理
standardized_matrix = standardize(normalized_matrix)

# 计算熵值
entropy_values = entropy(standardized_matrix)

# 计算差异系数
dissimilarity_coefficients = dissimilarity_coefficient(entropy_values)

# 计算熵权
entropy_weights = entropy_weight(entropy_values)


# 计算到理想解和负理想解的距离
ideal_positive = np.max(standardized_matrix, axis=0)
ideal_negative = np.min(standardized_matrix, axis=0)
distance_to_positive = np.sqrt(np.sum((standardized_matrix - ideal_positive)**2 * entropy_weights, axis=1))
distance_to_negative = np.sqrt(np.sum((ideal_negative - standardized_matrix)**2 * entropy_weights, axis=1))

# 计算相对接近度
relative_closeness = distance_to_negative / (distance_to_negative + distance_to_positive)

# 使用rankdata函数获取排名
ranking = rankdata(relative_closeness,method='max')
rankings= len(ranking) + 1 - ranking

# 输出结果
print("正向化矩阵：")
print(normalized_matrix)
print("标准化矩阵：")
print(standardized_matrix)
print("熵值：", entropy_values)
print("差异系数：", dissimilarity_coefficients)
print("熵权：", entropy_weights)
print("正理想解：",ideal_positive)
print("负理想解：",ideal_negative)
print("到正理想解的距离：", distance_to_positive)
print("到负理想解的距离：", distance_to_negative)
print("相对接近度：", relative_closeness)
print("评价对象排名：", rankings)








