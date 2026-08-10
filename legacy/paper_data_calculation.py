"""
核心数据汇总
汇总所有核心统计指标
"""

import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('user_behavior_with_sessions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
sequences_df = pd.read_csv('session_sequences.csv')

# 用户分类
users_buy = df[df['behavior_type'] == 'buy']['user_id'].unique()
users_pv = df[df['behavior_type'] == 'pv']['user_id'].unique()
users_loss = [u for u in users_pv if u not in users_buy]

print("=" * 50)
print("核心数据")
print("=" * 50)

# 基础统计
print(f"\n总记录数：{len(df):,}")
print(f"总会话数：{df['session_id'].nunique():,}")
print(f"转化用户数：{len(users_buy)}")
print(f"流失用户数：{len(users_loss)}")

# 会话层级转化
sessions_pv = len(sequences_df)
sessions_cart = sequences_df['has_cart'].sum()
sessions_buy = sequences_df['has_buy'].sum()

print(f"\n点击会话数：{sessions_pv}")
print(f"加购会话数：{sessions_cart}（转化率：{sessions_cart/sessions_pv*100:.2f}%）")
print(f"购买会话数：{sessions_buy}（转化率：{sessions_buy/sessions_pv*100:.2f}%）")
print(f"点击→加购流失率：{(1 - sessions_cart/sessions_pv)*100:.1f}%")
print(f"加购→购买流失率：{(1 - sessions_buy/sessions_cart)*100:.1f}%")

# 流失路径
no_buy = sequences_df[~sequences_df['has_buy']]
total_loss = len(no_buy)

pattern_pv = len(no_buy[no_buy['compressed_sequence'] == 'pv'])
pattern_cart = len(no_buy[no_buy['compressed_sequence'].str.contains('cart', na=False)])
pattern_fav = len(no_buy[no_buy['compressed_sequence'].str.contains('fav', na=False) &
                           ~no_buy['compressed_sequence'].str.contains('cart', na=False)])

print(f"\n流失路径模式：")
print(f"  模式一（纯点击）：{pattern_pv}（{pattern_pv/total_loss*100:.1f}%）")
print(f"  模式二（加购后流失）：{pattern_cart}（{pattern_cart/total_loss*100:.1f}%）")
print(f"  模式三（收藏后流失）：{pattern_fav}（{pattern_fav/total_loss*100:.1f}%）")