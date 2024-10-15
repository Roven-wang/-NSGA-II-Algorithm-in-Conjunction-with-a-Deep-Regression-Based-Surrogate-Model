import os
import pandas as pd

# 文件夹路径
folder_path = r'F:\desktop\NSGA选参3'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):  # 仅处理Excel文件
        file_path = os.path.join(folder_path, filename)

        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 假设第一列为要删除的列，第二列为需要拆分的数据列
        if len(df.columns) >= 2:
            # 删除第一列
            df = df.iloc[:, 1]

            # 将第二列数据转换为字符串类型
            df = df.astype(str)

            # 拆分第二列中的数据
            new_df = df.str.extract(r'\[(.*?),(.*?),(.*?)\]')

            # 转换为数值类型并取相反数
            new_df = new_df.apply(pd.to_numeric, errors='coerce')
            new_df = -new_df  # 取相反数

            # 设置新列名
            new_df.columns = ['Col1', 'Col2', 'Col3']

            # 保存修改后的Excel文件，覆盖原文件
            new_file_path = os.path.join(folder_path, 'processed_' + filename)
            new_df.to_excel(new_file_path, index=False)

print("所有文件已处理完毕。")
