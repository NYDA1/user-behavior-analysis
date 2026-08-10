"""
样本代表性验证
对比全量数据与采样数据的关键指标
"""

import pandas as pd

column_names = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']

# 统计全量数据
total_records = 0
total_pv = 0
total_cart = 0
total_fav = 0
total_buy = 0

chunk_size = 500000
for chunk in pd.read_csv('UserBehavior.csv', chunksize=chunk_size,
                          names=column_names, header=None):
    total_records += len(chunk)
    total_pv += (chunk['behavior_type'] == 'pv').sum()
    total_cart += (chunk['behavior_type'] == 'cart').sum()
    total_fav += (chunk['behavior_type'] == 'fav').sum()
    total_buy += (chunk['behavior_type'] == 'buy').sum()

pct_pv = total_pv / total_records * 100
pct_cart = total_cart / total_records * 100
pct_fav = total_fav / total_records * 100
pct_buy = total_buy / total_records * 100
conv_rate = total_buy / total_pv * 100

# 统计采样数据
df_sample = pd.read_csv('UserBehavior.csv', nrows=100000,
                          names=column_names, header=None)

sample_records = len(df_sample)
sample_pv = (df_sample['behavior_type'] == 'pv').sum()
sample_cart = (df_sample['behavior_type'] == 'cart').sum()
sample_fav = (df_sample['behavior_type'] == 'fav').sum()
sample_buy = (df_sample['behavior_type'] == 'buy').sum()

sample_pct_pv = sample_pv / sample_records * 100
sample_pct_cart = sample_cart / sample_records * 100
sample_pct_fav = sample_fav / sample_records * 100
sample_pct_buy = sample_buy / sample_records * 100
sample_conv_rate = sample_buy / sample_pv * 100

# 输出对比结果
print(f"{'指标':<12} {'全量数据':<12} {'采样数据':<12} {'差异值'}")
print("-" * 48)
print(f"{'点击占比':<12} {pct_pv:<11.2f}% {sample_pct_pv:<11.2f}% {abs(pct_pv - sample_pct_pv):<8.2f}%")
print(f"{'加购占比':<12} {pct_cart:<11.2f}% {sample_pct_cart:<11.2f}% {abs(pct_cart - sample_pct_cart):<8.2f}%")
print(f"{'收藏占比':<12} {pct_fav:<11.2f}% {sample_pct_fav:<11.2f}% {abs(pct_fav - sample_pct_fav):<8.2f}%")
print(f"{'购买占比':<12} {pct_buy:<11.2f}% {sample_pct_buy:<11.2f}% {abs(pct_buy - sample_pct_buy):<8.2f}%")
print(f"{'点击转化率':<12} {conv_rate:<11.2f}% {sample_conv_rate:<11.2f}% {abs(conv_rate - sample_conv_rate):<8.2f}%")