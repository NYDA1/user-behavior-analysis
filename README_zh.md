# 电商用户行为分析

基于**淘宝 UserBehavior 数据集**（约 1 亿条行为记录、100 万用户）的端到端数据分析项目：
以流式方式处理 **3.4 GB** 原始数据并随机抽取 **5,000 个完整用户**，对用户行为进行
会话化切分，分析转化漏斗与流失路径，训练**流失预测模型**（严格的防泄漏评估），
并以**交互式 Streamlit 仪表盘**呈现全部结果。

> 本项目基于作者的本科毕业设计（原始论文脚本归档在 [`legacy/`](legacy/README.md)）。

## 项目亮点

- **流式大数据处理** — 两遍分块扫描 3.4 GB CSV，峰值内存约 100 MB；可复现的
  5,000 用户随机抽样（`seed=42`）
- **全向量化 ETL** — 30 分钟会话切分、行为序列压缩、20+ 用户级特征全部用
  `groupby` 聚合实现（无逐用户 Python 循环）
- **漏斗与流失路径分析** — 用户级/会话级转化漏斗、三类流失模式（纯浏览 /
  加购后流失 / 仅收藏）、桑基图行为流转
- **防泄漏的流失预测** — 逻辑回归、随机森林、XGBoost，类别不平衡处理，
  采用**两种切分方案**（按用户随机切分 + 严格时间切分：第 1-7 天特征 → 第 8-10 天标签）
- **交互式仪表盘** — 6 个板块的 Streamlit 应用，只读取流水线产物（不重复计算）
- **完全可复现** — 一键流水线（`python run_all.py`）、固定随机种子、核心逻辑单元测试

## 数据集

**淘宝 UserBehavior**（阿里天池）：移动电商平台的用户行为日志，每条记录一次交互：

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | int | 用户 id（匿名化） |
| `item_id` | int | 商品 id（匿名化） |
| `category_id` | int | 类目 id（匿名化） |
| `behavior_type` | str | `pv` 浏览 · `cart` 加购 · `fav` 收藏 · `buy` 购买 |
| `timestamp` | int | Unix 时间戳 |

全量数据为 **约 1 亿条事件、100 万用户**，周期 10 天（2017-11-24 → 2017-12-03）。
流水线抽样 **5,000 个完整用户**（约 50 万条事件），在保证交互性的同时保持代表性
（pv ≈ 89%、buy ≈ 2%，与公开数据分布一致）。

> 运行前请将原始 CSV 下载到 `data/raw/UserBehavior.csv`（该文件已加入 .gitignore）。

## 流水线

```
data/raw/UserBehavior.csv  (3.4 GB, 约 1 亿行)
        │
        ▼ 01_sample_users.py   两遍流式抽样：5,000 个完整用户
data/processed/user_sample.parquet  （约 50 万条事件）
        │
        ▼ 02_build_features.py 向量化 ETL
        ├─ events.parquet      事件 + session_id（30 分钟无操作切分）
        ├─ sessions.parquet    压缩行为序列、标志位、时长
        └─ user_features.parquet  每个用户 20+ 特征 + 是否购买标签
        │
        ├─ ▼ 03_analysis.py    漏斗 · 流失路径 · 桑基图 · 特征对比
        │     └─ output/charts/*  output/metrics/*.json
        │
        └─ ▼ 04_churn_model.py  LR / RandomForest / XGBoost，组切分与时间切分
              └─ output/model/*  + data/processed/model_predictions.parquet
        │
        ▼ dashboard/app.py      Streamlit — 只读取以上产物
```

## 关键结果

> 以下数字由当前流水线在 5,000 用户样本上生成（重新运行 `python run_all.py` 即可复现）。

### 转化漏斗（会话级）

| 阶段 | 会话数 | 相对 pv 转化率 | 阶段转化率 |
|---|---|---|---|
| pv | 84,251 | 100% | — |
| cart | 15,958 | 18.9% | pv → cart: 18.9% |
| buy | 7,905 | 9.4% | cart → buy: 49.5% |

*用户级：69.2% 的用户完成购买；加过购的用户中 92.6% 最终购买。*

### 流失路径模式（未购买会话，n = 76,346）

| 模式 | 会话数 | 占比 |
|---|---|---|
| 纯浏览（无任何交互） | 55,851 | 73.2% |
| 加购后流失 | 14,271 | 18.7% |
| 仅收藏 | 6,224 | 8.2% |

### 流失预测（按用户随机切分，测试集）

| 模型 | ROC-AUC | PR-AUC | F1@0.5 | MCC | Precision@top-10% |
|---|---|---|---|---|---|
| Logistic Regression | 0.718 | 0.825 | 0.731 | 0.326 | 0.860 |
| RandomForest | 0.686 | 0.814 | 0.825 | 0.264 | 0.870 |
| XGBoost | 0.712 | 0.828 | 0.774 | 0.296 | 0.890 |

*测试集购买用户占比（prevalence）：68.9%。Precision@top-10% = 0.89 表示
模型最有把握的 10% 用户中有 89% 完成购买——比基线高出 29 个百分点。*

### 流失预测（严格时间切分）

第 1-7 天特征预测第 8-10 天标签（窗口不重叠）。标签窗口购买占比：**38.6%**。

| 模型 | ROC-AUC | PR-AUC | Precision@top-10% |
|---|---|---|---|
| Logistic Regression | 0.571 | 0.466 | 0.570 |
| RandomForest | 0.530 | 0.409 | 0.440 |
| XGBoost | 0.526 | 0.417 | 0.460 |

预测未来窗口更难——但头部 10% 用户的精确率仍比 38.6% 基线提升 47%。
运行 `python scripts/04_churn_model.py --split time` 重新生成
（仪表盘显示最后一次运行的切分方案）。

## 建模细节

- **任务**：仅基于购买前的行为，预测用户是否会完成购买。
- **特征（21 个）**：行为计数与比率（`cart_rate`、`fav_rate`）、会话结构
  （`n_sessions`、平均/中位会话长度、会话时长）、活跃度（`n_active_days`、
  行为节奏）、多样性（`n_categories`、`n_items`、头部类目占比）、时段偏好
  （晚间占比）。标签单独计算，并有断言阻止任何购买相关列进入特征矩阵。
- **类别不平衡**：仅约 9% 的用户购买；用 `class_weight` / `scale_pos_weight`
  处理，并以 **PR-AUC**（而非仅 ROC-AUC）和 top-10% 精确率相对 prevalence
  基线汇报。
- **防泄漏评估**（两种方案均提供）：
  - `--split group`：按用户分层随机切分——每行一个用户，训练/测试用户不重叠。
  - `--split time`：特征取**第 1-7 天**，标签取**第 8-10 天**。特征窗口与
    标签窗口不重叠，模型无法利用事后信息——比随机切分更强的论证。

## 目录结构

```
├── run_all.py                  一键流水线（01 → 04）
├── pyproject.toml              src-layout 包定义（pip install -e .）
├── src/user_behavior_analysis/ 公共逻辑：config, io, sessions,
│                                sequences, features, funnel, plots
├── scripts/                    01_sample_users · 02_build_features ·
│                               03_analysis · 04_churn_model
├── dashboard/app.py            Streamlit 仪表盘
├── tests/                      会话切分、序列、流失模式、防泄漏守卫的单元测试
├── data/                       [gitignored] 原始数据与中间产物
├── output/                     [gitignored] 图表、指标 JSON、模型输出
└── legacy/                     论文原版脚本（存档）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install -e .

# 2. 放置数据集（已 gitignore）
#    下载 UserBehavior.csv -> data/raw/UserBehavior.csv

# 3. 运行完整流水线
python run_all.py

# 4. 启动仪表盘
streamlit run dashboard/app.py

# 5. 运行测试
python -m pytest
```

可选：`python scripts/04_churn_model.py --split time` 运行时间切分评估
（仪表盘显示最后一次运行的切分方案）。

## 可复现性

- `SEED = 42` — 抽样、训练/测试切分、所有模型共用同一随机种子。
- `SAMPLE_SIZE = 5000`、`SESSION_GAP_MINUTES = 30` 及流失模式定义位于
  `src/user_behavior_analysis/config.py`。
- 每个流水线步骤幂等；`python run_all.py` 可从零重新生成全部产物。

## 技术栈

Python 3.11+ · pandas · numpy · matplotlib · plotly · scikit-learn · XGBoost ·
Streamlit · pytest

## 许可

MIT
