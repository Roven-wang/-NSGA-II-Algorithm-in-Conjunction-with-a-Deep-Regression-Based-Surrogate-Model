import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

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

# 假设Excel文件中有三列数据，分别为'NSE'、'BR2'和'D'
x = data['NSE']
y = data['BR2']
z = data['D']

# 绘制三维散点图
fig = plt.figure(figsize=(18, 18))
ax = fig.add_subplot(111, projection='3d')

# 绘制第一个点（特殊颜色）
ax.scatter(x[0], y[0], z[0], color='#FF0000', s=550, marker='o', edgecolors='none', alpha=0.9, label='First Point')  # 红色

# 绘制其余点（原本的颜色）
ax.scatter(x[1:], y[1:], z[1:], color='#007DD9', s=450, marker='o', edgecolors='#007DD9', alpha=0.6, depthshade=True)

# 设置坐标轴标签
ax.set_xlabel('NSE', fontsize=40, labelpad=52, fontstyle='italic')
ax.set_ylabel('bR$^2$', fontsize=40, labelpad=52, fontstyle='italic')
ax.set_zlabel('D', fontsize=40, labelpad=52, fontstyle='italic')

# 设置图像偏移的角度
ax.view_init(elev=31, azim=49)

# 调整刻度和布局
plt.tick_params(axis='x', pad=10, labelsize=40)
plt.tick_params(axis='y', pad=10, labelsize=40)
plt.tick_params(axis='z', pad=25, labelsize=40)

# 设置网格线的格式
ax.xaxis._axinfo["grid"].update(color='#3b5387', linestyle='--', linewidth=0.5)
ax.yaxis._axinfo["grid"].update(color='#3b5387', linestyle='--', linewidth=0.5)
ax.zaxis._axinfo["grid"].update(color='#3b5387', linestyle='--', linewidth=0.5)

# 设置背景颜色为空白
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

plt.savefig(r'F:\desktop\散点4.png', dpi=1000)
plt.show()
