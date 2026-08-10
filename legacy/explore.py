"""
数据探索
查看原始数据的基本情况
"""

import pandas as pd
import os

# 文件路径
file_path = 'UserBehavior.csv'

if not os.path.exists(file_path):
    print(f"错误：找不到文件 {file_path}")
    exit()

# 定义列名
column_names = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']

# 读取前10万条数据
df = pd.read_csv(file_path, nrows=100000, names=column_names, header=None)

# 查看数据形状
print(f"数据行数：{df.shape[0]}")
print(f"数据列数：{df.shape[1]}")

# 查看前5行
print("\n前5行数据：")
print(df.head())

# 查看数据类型
print("\n数据类型：")
print(df.dtypes)

# 行为类型分布
print("\n行为类型分布：")
print(df['behavior_type'].value_counts())

# 用户数、商品数、类目数
print(f"\n用户数：{df['user_id'].nunique()}")
print(f"商品数：{df['item_id'].nunique()}")
print(f"类目数：{df['category_id'].nunique()}")

# 时间范围
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
print(f"时间范围：{df['datetime'].min()} 到 {df['datetime'].max()}")

# 保存采样数据
# df.to_csv('user_behavior_sample.csv', index=False)
# print("\n已保存采样数据：user_behavior_sample.csv")