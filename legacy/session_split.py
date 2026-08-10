"""
会话分割
将用户行为序列按30分钟时间窗口切分为独立会话
"""

import pandas as pd

# 读取优化后的数据
df = pd.read_csv('user_behavior_optimized.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 按用户和时间排序
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# 计算相邻行为时间差（分钟）
df['time_diff'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60

# 标记新会话（时间差 > 30分钟 或 第一条记录）
df['new_session'] = (df['time_diff'] > 30) | (df['time_diff'].isna())

# 生成会话ID
df['session_id'] = df.groupby('user_id')['new_session'].cumsum()
df['session_id'] = df['user_id'].astype(str) + '_' + df['session_id'].astype(str).str.zfill(3)

# 输出统计结果
total_sessions = df['session_id'].nunique()
avg_len = len(df) / total_sessions

print(f"总会话数：{total_sessions}")
print(f"平均会话长度：{avg_len:.2f} 个行为")

# 会话长度分布
session_lengths = df.groupby('session_id').size()
print(f"最短会话：{session_lengths.min()} 个行为")
print(f"中位数：{session_lengths.median():.0f} 个行为")
print(f"最长会话：{session_lengths.max()} 个行为")

# 保存结果
df.to_csv('user_behavior_with_sessions.csv', index=False)
print("已保存：user_behavior_with_sessions.csv")