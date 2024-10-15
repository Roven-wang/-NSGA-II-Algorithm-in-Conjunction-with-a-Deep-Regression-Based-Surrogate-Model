import pandas as pd

# 读取数据
data = pd.read_csv(r'F:\desktop\训练数据.csv')  # 替换为你的数据文件路径

# 删除缺失值所在的行
data.dropna(inplace=True)

# 删除异常值所在的行
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = ((data < lower_bound) | (data > upper_bound)).any(axis=1)
data = data[~outliers]

# 保存处理后的数据集
data.to_csv(r'F:\desktop\训练数据2.csv', index=False)
