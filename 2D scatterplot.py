import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

# 设置英文字体为Times New Roman或其他支持英文的字体
font_path = r'C:\Windows\Fonts\times.ttf'  # Times New Roman的字体文件路径
font = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font.get_name()

# 从Excel文件中读取数据
file_path = r'F:\desktop\pareto_front.xlsx'  # 请替换为你的文件路径
sheet_name = 'Sheet1'  # 请替换为你的工作表名称
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 假设Excel文件中有两列数据，分别为'NSE'和'D'
x = data['BR2']
y = data['D']

# 绘制第一个点（特殊颜色）
plt.figure(figsize=(7, 6))

# 绘制其余点（原本的颜色）
plt.scatter(x[1:], y[1:], color='#007DD9', s=70, marker='o', edgecolors='none')
plt.scatter(x[0], y[0], color='#FF0000', s=100, marker='o', edgecolors='none', label='First Point')  # 红色


# 设置坐标轴标签
plt.xlabel('bR$^2$', fontsize=22, labelpad=7, fontstyle='italic')
plt.ylabel('D', fontsize=22, labelpad=7, fontstyle='italic')
plt.grid(False)

# 调整刻度和布局
plt.tick_params(axis='x', pad=3, labelsize=20)
plt.tick_params(axis='y', pad=3, labelsize=20)
plt.tight_layout(pad=1.0, w_pad=0.8, h_pad=0.8)

# 保存图像并显示
plt.savefig(r'F:\desktop\散点1.png', dpi=1000)
plt.show()
