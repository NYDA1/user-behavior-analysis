"""
桑基图绘制
可视化用户行为流转路径
"""

import pandas as pd
import plotly.graph_objects as go

# 读取行为序列数据
sequences_df = pd.read_csv('session_sequences.csv')

# 统计状态转移
def count_transitions(seq):
    states = seq.split('→')
    return [(states[i], states[i+1]) for i in range(len(states)-1)]

all_links = []
for seq in sequences_df['compressed_sequence']:
    all_links.extend(count_transitions(seq))

# 聚合转移次数
link_counts = {}
for src, tgt in all_links:
    key = (src, tgt)
    link_counts[key] = link_counts.get(key, 0) + 1

# 定义节点
nodes = ['pv', 'cart', 'fav', 'buy']
node_map = {n: i for i, n in enumerate(nodes)}

# 构建链接
sources = []
targets = []
values = []

for (src, tgt), val in link_counts.items():
    if src in node_map and tgt in node_map:
        sources.append(node_map[src])
        targets.append(node_map[tgt])
        values.append(val)

# 绘制桑基图
fig = go.Figure(data=[go.Sankey(
    node=dict(
        label=nodes,
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values
    )
)])

fig.update_layout(title="用户行为路径桑基图", width=800, height=600)
fig.write_html("user_path_sankey.html")

print("已保存：user_path_sankey.html")
print(f"节点数：{len(nodes)}")
print(f"链接数：{len(sources)}")