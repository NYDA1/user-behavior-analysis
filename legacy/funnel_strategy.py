"""
漏斗分析与策略输出
计算用户层级和会话层级的转化漏斗
"""

import pandas as pd
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('user_behavior_with_sessions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
sequences_df = pd.read_csv('session_sequences.csv')

# 用户层级转化统计
users_pv = df[df['behavior_type'] == 'pv']['user_id'].nunique()
users_cart = df[df['behavior_type'] == 'cart']['user_id'].nunique()
users_buy = df[df['behavior_type'] == 'buy']['user_id'].nunique()

print("用户层级转化漏斗")
print(f"点击用户数：{users_pv}")
print(f"加购用户数：{users_cart}（转化率：{users_cart/users_pv*100:.2f}%）")
print(f"购买用户数：{users_buy}（转化率：{users_buy/users_pv*100:.2f}%）")

# 会话层级转化统计
sessions_pv = len(sequences_df)
sessions_cart = sequences_df['has_cart'].sum()
sessions_buy = sequences_df['has_buy'].sum()

print("\n会话层级转化漏斗")
print(f"点击会话数：{sessions_pv}")
print(f"加购会话数：{sessions_cart}（转化率：{sessions_cart/sessions_pv*100:.2f}%）")
print(f"购买会话数：{sessions_buy}（转化率：{sessions_buy/sessions_pv*100:.2f}%）")

print(f"\n点击→加购流失率：{(1 - sessions_cart/sessions_pv)*100:.1f}%")
print(f"加购→购买流失率：{(1 - sessions_buy/sessions_cart)*100:.1f}%")

# 流失路径统计
no_buy = sequences_df[~sequences_df['has_buy']]
total_loss = len(no_buy)

pattern_pv = no_buy[no_buy['compressed_sequence'] == 'pv']
pattern_cart = no_buy[no_buy['compressed_sequence'].str.contains('cart', na=False)]
pattern_fav = no_buy[no_buy['compressed_sequence'].str.contains('fav', na=False) &
                      ~no_buy['compressed_sequence'].str.contains('cart', na=False)]

print("\n流失路径模式统计")
print(f"模式一（纯点击无交互）：{len(pattern_pv)}（{len(pattern_pv)/total_loss*100:.1f}%）")
print(f"模式二（加购后流失）：{len(pattern_cart)}（{len(pattern_cart)/total_loss*100:.1f}%）")
print(f"模式三（收藏后流失）：{len(pattern_fav)}（{len(pattern_fav)/total_loss*100:.1f}%）")

# 绘制漏斗图
stages = ['点击用户', '加购用户', '购买用户']
values = [users_pv, users_cart, users_buy]

plt.figure(figsize=(8, 5))
bars = plt.barh(stages, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

for i, (bar, val) in enumerate(zip(bars, values)):
    plt.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
             str(val), va='center', ha='left')
    if i > 0:
        rate = val / values[0] * 100
        plt.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                 f'{rate:.1f}%', va='center', ha='center', color='white')

plt.title('用户转化漏斗')
plt.xlabel('用户数')
plt.tight_layout()
plt.savefig('conversion_funnel.png', dpi=300)

print("\n已保存：conversion_funnel.png")