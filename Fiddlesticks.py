import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

# 设置英文字体为Times New Roman或其他支持英文的字体
font_path = r'C:\Windows\Fonts\times.ttf'  # Times New Roman的字体文件路径
font = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font.get_name()


# 读取Excel文件
file_path = r'F:\desktop\提琴图.xlsx'
data = pd.read_excel(file_path)

# 将数据转换为适合seaborn的长格式
data_long = data.melt(var_name='指标', value_name='值')

# 设置调色板（颜色）
palette = sns.color_palette("husl", 10)  # 你可以选择其他调色板，如 "muted", "bright", "dark" 等

# 创建提琴图
plt.figure(figsize=(12, 8))
sns.violinplot(x='指标', y='值', data=data_long, palette=palette)
plt.title('10个指标的提琴图')
plt.xlabel('指标')
plt.ylabel('值')
plt.show()
