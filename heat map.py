import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap

# 设置英文字体为Times New Roman或其他支持英文的字体
font_path = r'C:\Windows\Fonts\times.ttf'  # Times New Roman的字体文件路径
font = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font.get_name()

# 读取Excel文件
# 请将 'your_file.xlsx' 替换为你的文件路径
df = pd.read_excel(r'F:\desktop\热图.xlsx', index_col=0)

# 生成热图
plt.figure(figsize=(7.5, 6))  # 调整图形大小
custom_cmap = LinearSegmentedColormap.from_list("my_cmap", ["#dadad8", "#1477d2"])
heatmap = sns.heatmap(df, annot=False, cmap=custom_cmap, fmt=".2f", annot_kws={"size": 15})  # 调整颜色条图例字体大小

# 设置颜色条刻度字体大小
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)  # 设置颜色条刻度字体大小

plt.xticks(fontsize=18)  # 设置x轴标签字体大小
plt.yticks(fontsize=18)  # 设置y轴标签字体大小
plt.ylabel('')
#plt.tight_layout(pad=1.0, w_pad=0.8, h_pad=0.8)
# 保存图形
plt.savefig(r'F:\desktop\热图8.png', dpi=1000)
plt.show()
