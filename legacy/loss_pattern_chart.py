"""
流失路径分布图
绘制流失路径模式的饼图和柱状图
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取行为序列数据
sequences_df = pd.read_csv('session_sequences.csv')

# 筛选流失会话
no_buy = sequences_df[~sequences_df['has_buy']]

# 按规则分类
pattern_pv = no_buy[no_buy['compressed_sequence'] == 'pv']
pattern_cart = no_buy[no_buy['compressed_sequence'].str.contains('cart', na=False)]
pattern_fav = no_buy[no_buy['compressed_sequence'].str.contains('fav', na=False) &
                      ~no_buy['compressed_sequence'].str.contains('cart', na=False)]

p1 = len(pattern_pv)
p2 = len(pattern_cart)
p3 = len(pattern_fav)
total = len(no_buy)

print(f"总流失会话数：{total}")
print(f"模式一（纯点击）：{p1} ({p1/total*100:.1f}%)")
print(f"模式二（加购后流失）：{p2} ({p2/total*100:.1f}%)")
print(f"模式三（收藏后流失）：{p3} ({p3/total*100:.1f}%)")

# 绘制饼图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

labels = ['纯点击无交互', '加购后流失', '收藏后流失']
sizes = [p1, p2, p3]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax1.set_title('流失路径模式分布（饼图）')

# 绘制柱状图
bars = ax2.bar(labels, sizes, color=colors, edgecolor='white')
ax2.set_ylabel('会话数量')
ax2.set_title('流失路径模式分布（柱状图）')
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, sizes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{val}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('loss_pattern_distribution.png', dpi=300)
print("\n已保存：loss_pattern_distribution.png")