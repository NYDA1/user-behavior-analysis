"""
用户行为特征对比
对比转化用户与流失用户的行为特征
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('user_behavior_with_sessions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 用户分类
users_buy = df[df['behavior_type'] == 'buy']['user_id'].unique()
users_pv = df[df['behavior_type'] == 'pv']['user_id'].unique()
users_loss = [u for u in users_pv if u not in users_buy]

print(f"转化用户数：{len(users_buy)}")
print(f"流失用户数：{len(users_loss)}")

# 平均会话长度
user_avg_len = {}
for uid in df['user_id'].unique():
    user_sessions = df[df['user_id'] == uid].groupby('session_id').size()
    if len(user_sessions) > 0:
        user_avg_len[uid] = user_sessions.mean()

len_buy = np.mean([user_avg_len[u] for u in users_buy if u in user_avg_len])
len_loss = np.mean([user_avg_len[u] for u in users_loss if u in user_avg_len])
print(f"\n平均会话长度：转化用户 {len_buy:.1f} 个行为，流失用户 {len_loss:.1f} 个行为")

# 人均加购次数
cart_cnt = df[df['behavior_type'] == 'cart'].groupby('user_id').size().to_dict()
cart_buy = np.mean([cart_cnt.get(u, 0) for u in users_buy])
cart_loss = np.mean([cart_cnt.get(u, 0) for u in users_loss])
print(f"人均加购次数：转化用户 {cart_buy:.1f} 次，流失用户 {cart_loss:.1f} 次")

# 人均收藏次数
fav_cnt = df[df['behavior_type'] == 'fav'].groupby('user_id').size().to_dict()
fav_buy = np.mean([fav_cnt.get(u, 0) for u in users_buy])
fav_loss = np.mean([fav_cnt.get(u, 0) for u in users_loss])
print(f"人均收藏次数：转化用户 {fav_buy:.1f} 次，流失用户 {fav_loss:.1f} 次")

# 平均会话时长
sess_dur = {}
for sid, group in df.groupby('session_id'):
    duration = (group['timestamp'].max() - group['timestamp'].min()).total_seconds() / 60
    sess_dur[sid] = duration

user_dur = {}
for uid in df['user_id'].unique():
    user_sids = df[df['user_id'] == uid]['session_id'].unique()
    durations = [sess_dur[s] for s in user_sids if s in sess_dur]
    if durations:
        user_dur[uid] = np.mean(durations)

dur_buy = np.mean([user_dur[u] for u in users_buy if u in user_dur])
dur_loss = np.mean([user_dur[u] for u in users_loss if u in user_dur])
print(f"平均会话时长：转化用户 {dur_buy:.1f} 分钟，流失用户 {dur_loss:.1f} 分钟")

# 绘制对比柱状图
metrics = ['平均会话长度\n(个行为)', '人均加购次数\n(次)', '人均收藏次数\n(次)']
buy_values = [len_buy, cart_buy, fav_buy]
loss_values = [len_loss, cart_loss, fav_loss]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, buy_values, width, label='转化用户', color='#2ca02c')
bars2 = ax.bar(x + width/2, loss_values, width, label='流失用户', color='#d62728')

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

ax.set_title('转化用户与流失用户行为特征对比')
ax.set_ylabel('数值')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.tight_layout()
plt.savefig('user_feature_compare.png', dpi=300)
print("\n已保存：user_feature_compare.png")