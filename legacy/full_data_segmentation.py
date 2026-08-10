"""
全量数据用户分群统计
计算高价值用户和流失用户的数量及占比
"""

import pandas as pd

column_names = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']

# 统计全量用户的点击和购买情况
users_with_pv = set()
user_buy_count = {}

chunk_size = 500000
for chunk in pd.read_csv('UserBehavior.csv', chunksize=chunk_size,
                         names=column_names, header=None,
                         usecols=['user_id', 'behavior_type']):

    # 有点击的用户
    pv_users = chunk[chunk['behavior_type'] == 'pv']['user_id'].unique()
    users_with_pv.update(pv_users)

    # 统计每个用户的购买次数
    buy_counts = chunk[chunk['behavior_type'] == 'buy']['user_id'].value_counts().to_dict()
    for uid, count in buy_counts.items():
        user_buy_count[uid] = user_buy_count.get(uid, 0) + count

users_with_buy = set(user_buy_count.keys())

# 高价值用户（购买次数 >= 3）
high_value_users = [uid for uid, count in user_buy_count.items() if count >= 3]

# 流失用户（有点击但无购买）
loss_users = users_with_pv - users_with_buy

total_pv = len(users_with_pv)
total_high_value = len(high_value_users)
total_loss = len(loss_users)

print(f"有点击的用户数：{total_pv:,}")
print(f"高价值用户数（购买>=3次）：{total_high_value:,} ({total_high_value / total_pv * 100:.2f}%)")
print(f"流失用户数（有点击无购买）：{total_loss:,} ({total_loss / total_pv * 100:.2f}%)")

# 购买次数分布
print("\n购买次数分布：")
buy_dist = {}
for count in user_buy_count.values():
    buy_dist[count] = buy_dist.get(count, 0) + 1

for i in range(1, 11):
    if i in buy_dist:
        print(f"  购买 {i} 次：{buy_dist[i]:,} 人")