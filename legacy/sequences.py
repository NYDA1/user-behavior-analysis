"""
行为序列构建
为每个会话生成行为序列，并压缩连续重复行为
"""

import pandas as pd

# 读取带会话ID的数据
df = pd.read_csv('user_behavior_with_sessions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 构建行为序列
sequences = []

for session_id, group in df.groupby('session_id'):
    group = group.sort_values('timestamp')
    behaviors = group['behavior_type'].tolist()

    # 压缩连续重复行为
    compressed = []
    for b in behaviors:
        if not compressed or b != compressed[-1]:
            compressed.append(b)

    sequences.append({
        'session_id': session_id,
        'user_id': group['user_id'].iloc[0],
        'has_buy': 'buy' in behaviors,
        'has_cart': 'cart' in behaviors,
        'has_fav': 'fav' in behaviors,
        'raw_sequence': '→'.join(behaviors),
        'compressed_sequence': '→'.join(compressed),
        'raw_length': len(behaviors),
        'compressed_length': len(compressed)
    })

sequences_df = pd.DataFrame(sequences)

# 统计信息
print(f"总会话数：{len(sequences_df)}")
print(f"有购买的会话：{sequences_df['has_buy'].sum()}")
print(f"有加购的会话：{sequences_df['has_cart'].sum()}")
print(f"有收藏的会话：{sequences_df['has_fav'].sum()}")

# 高频序列
print("\n高频行为序列（前10）：")
top_sequences = sequences_df['compressed_sequence'].value_counts().head(10)
for seq, count in top_sequences.items():
    print(f"  {seq}: {count}")

# 保存结果
sequences_df.to_csv('session_sequences.csv', index=False)
print("\n已保存：session_sequences.csv")