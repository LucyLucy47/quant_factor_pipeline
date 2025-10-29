# 量化高频因子挖掘与模型训练 Pipeline

本仓库提供一个完整的量化因子挖掘与模型训练框架，适用于基于 Level‑2 高频数据和日频因子库（如 Alpha158）构建股票预测模型。其核心功能包括：

* **GPU 高频特征生成**：使用 `quant_factor_pipeline.py` 中的 `GPUFeatureGenerator` 将高频订单簿数据转换为掩码–基础特征–聚合算子的组合特征，支持 sum、mean、std、max、min 等统计指标，并自动利用 GPU 加速（如可用）。
* **特征筛选**：同时支持基于信息系数 (IC) 的线性筛选（连续二分法）和基于 LightGBM 特征重要度的非线性筛选，可捕捉因子间的复杂组合关系。
* **多音级筛选**：新增“polyphonic”筛选模式，对时间序列划分为多个阶段，在每个阶段计算因子 t‑统计量并结合平均绝对 t 值与标准差衡量稳定性，通过 K‑means 聚类在各簇中按综合得分筛选因子，最终保留至少 5000 个表现各异且稳定的候选因子。
* **多频率融合**：可将筛选后的高频因子与日频因子库拼接，形成完整的特征矩阵，提高模型的稳健性。
* **多算法训练**：内置 LightGBM、XGBoost 和图卷积网络 (GNN) 三种回归模型，均支持 GPU 加速训练；用户可根据需要选择不同算法进行比较。
* **新增经典算法**：在最新版本中，增加了随机森林（Random Forest）、岭回归（Ridge Regression）和支持向量回归（SVR），并扩展支持弹性网（ElasticNet）、Lasso 和多层感知机回归（MLPRegressor）等传统机器学习模型；这些模型均内置简单的超参数搜索以提升预测效果。
* **GitHub Actions 自动化**：在 `.github/workflows/daily_pipeline.yml` 中配置了一个每天触发的工作流，自动运行因子挖掘代理并执行训练脚本，实现数据和模型的持续更新。

## 目录结构

```
quant_factor_pipeline.py        # 主脚本：加载数据、生成特征、筛选因子、融合日频数据并训练模型
.github/
  workflows/
    daily_pipeline.yml          # GitHub Actions 工作流，定时运行挖掘和训练
requirements.txt                # Python 依赖列表
README.md                       # 项目说明
```

## 安装依赖

建议使用 Python 3.8+（示例为 3.10）。执行以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

如需在 GPU 上运行 LightGBM/XGBoost/GNN，请确保已安装相应的 CUDA 环境，并使用支持 GPU 的包版本。

## 数据准备

1. **高频数据**：将 Level‑2 高频数据按照日期保存为 `YYYYMMDD.csv`，每个文件包含以下字段（可根据实际情况调整）：
   - `datetime`：时间戳，精确到秒或毫秒；
   - `code`：股票代码；
   - `bid_price1`、`ask_price1`、`bid_volume1`、`ask_volume1`：一级报价与挂单数量；
   - `last_price`：成交价；
   - `trade_volume`：成交量；
   - `order_imbal`：订单簿失衡度等。

2. **日频因子库（可选）**：若有 Alpha158 等日频因子，可以同样按日期保存为 `YYYYMMDD.csv` 并放置在单独目录中。脚本会自动对齐 `(date, code)` 索引并融合。

3. 在 `quant_factor_pipeline.py` 中的 `run_pipeline` 调用时，通过 `DATA_DIR_HF` 和 `DATA_DIR_DAILY` 指定上述数据目录。

## 使用方法

### 手动运行

编辑 `quant_factor_pipeline.py` 末尾的示例参数，使其指向正确的数据目录和日期区间，然后直接运行：

```bash
python quant_factor_pipeline.py
```

### 单独训练与回测

仓库还提供 `train_and_backtest.py` 脚本，用于在已有的特征数据与收益数据上执行模型训练并进行简单的日内多头回测。该脚本支持与 `run_pipeline` 相同的算法选项（包括 `lightgbm`、`xgboost`、`gnn`、`random_forest`、`ridge`、`svr`、`elastic_net`、`lasso`、`mlp`），并对每种传统机器学习模型（随机森林、岭/弹性网回归、Lasso、支持向量回归和多层感知机）自动进行简单的超参数搜索。

示例用法：

```bash
python train_and_backtest.py \
  --features_path output/selected_features.csv \
  --returns_path output/returns.csv \
  --train_start 2016-01-01 --train_end 2019-12-31 \
  --valid_start 2020-01-01 --valid_end 2021-12-31 \
  --test_start 2022-01-01 --test_end 2024-12-31 \
  --algorithm ridge \
  --topk 20
```

脚本会输出回测摘要并将每天的策略收益保存至 `backtest_results.csv`。请注意，该回测框架仅为演示用途，未考虑交易成本、滑点或仓位控制等因素。

你也可以在其他脚本中导入 `run_pipeline`、`train_model` 等函数进行自定义调用。例如：

```python
from quant_factor_pipeline import run_pipeline

preds = run_pipeline(
    data_dir_hf='/path/to/high_freq',
    data_dir_daily='/path/to/daily_factors',
    start_date='2016-01-01',
    end_date='2024-12-31',
    train_range=('2016-01-01','2019-12-31'),
    valid_range=('2020-01-01','2021-12-31'),
    test_range=('2022-01-01','2024-12-31'),
    base_features=['bid_price1','ask_price1','bid_volume1','ask_volume1','last_price','trade_volume','order_imbal'],
    screening='linear',  # 也可以设为 'nonlinear' 或 'polyphonic'
    algorithm='lightgbm'  # 也可以设为 'xgboost', 'gnn', 'random_forest', 'ridge', 'svr', 'elastic_net', 'lasso', 'mlp'
)
```

### 自动化运行

仓库包含的 GitHub Actions 工作流 `daily_pipeline.yml` 会每天按设定时间自动运行挖掘代理和训练脚本。要启用自动化：

1. 确保仓库已经启用 GitHub Actions，并且您在仓库或组织设置中允许 GitHub Actions 运行。
2. 根据实际需要修改 `daily_pipeline.yml` 中的 `cron` 表达式、Python 版本和依赖安装步骤。
3. 将数据目录路径及脚本参数在工作流或脚本中配置好（例如使用环境变量或修改脚本默认值）。

一旦配置完成，GitHub Actions 会自动拉取仓库、安装依赖、运行 `daily_feature_agent.py` 更新因子，并调用 `quant_factor_pipeline.py` 进行训练和预测。您可以在 Actions 页查看运行记录和日志。

## 注意事项

* 本项目的代码仅供研究和学习使用，不构成任何投资建议；
* 在提交到公开仓库之前，请确保不含有任何敏感或私有数据；
* 如需使用 GPU，请提前配置好 CUDA 驱动，并安装支持 GPU 的 PyTorch、LightGBM、XGBoost 等版本。

欢迎根据自身需求修改或扩展本框架，例如接入更多因子、改进模型结构、增加超参数调优等。