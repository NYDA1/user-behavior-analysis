"""
数据类型优化
降低内存占用
"""

import pandas as pd

# 读取采样数据
df = pd.read_csv('user_behavior_sample.csv')

print(f"优化前内存：{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# 整数类型优化
for col in ['user_id', 'item_id', 'category_id']:
    col_min = df[col].min()
    col_max = df[col].max()
    if col_max < 2**15:
        df[col] = df[col].astype('int16')
    elif col_max < 2**31:
        df[col] = df[col].astype('int32')
    else:
        df[col] = df[col].astype('int64')

# 行为类型转为category
df['behavior_type'] = df['behavior_type'].astype('category')

# 时间戳转为datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

print(f"优化后内存：{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# 保存优化后的数据
df.to_csv('user_behavior_optimized.csv', index=False)
print("已保存：user_behavior_optimized.csv")