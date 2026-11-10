# 恢复成本预测：补充五折评价

本分析回应原留出测试仅有 13 个案例的问题，复用已经保存的实验输出，检查结论在完整的可评价样本中是否保持一致。它没有新增攻击轨迹，也没有重新运行控制器。原来的 13 个留出案例与原结果保持不变；这里报告的是事后增加的补充评价，不能作为一个新的独立测试集。

## 样本与预测量

原实验共 100 个场景，其中 75 个受到非零强度攻击。75 个场景中，62 个场景的实际状态与参考状态都存在有限的约束恢复成本；其中 55 个场景的两条滚动恢复路径最终均成功。其余 13 个单侧不可行场景与 7 个成对可行但执行恢复未共同成功的场景不进入本项费用预测评价。全部 55 个场景的预测量、目标量及充电效率也通过有限值检查，效率均为正。

预测目标沿用原实验的 `realized_extra_cost`：实际路径与未受污染参考路径在检测后的实际购电费用差。四种预测量分别为：

- 成对约束成本：`estimated_extra_cost`，没有基于评价数据拟合。
- 能量不足量回归：以 `soc_deficit_kwh` 为单一自变量、包含截距的最小二乘回归。
- 当前电价：能量不足量除以充电效率，再乘检测时的当前电价。
- 平均预测电价：能量不足量除以充电效率，再乘共同评价时域内的平均预测电价。

## 计算过程

1. 按原规则 `seed % 5 == 0` 留出 13 个场景，以其余 42 个场景拟合回归。首先逐项复现原留出预测，以及四个预测量的 MAE 和 RMSE；数值容差为 `1e-12`。
2. 将 55 个合格场景按 `scenario_id` 排序，用 NumPy `default_rng(20260924).permutation(55)` 打乱，再均分为 5 折。每折含 11 个测试场景和 44 个训练场景。每个场景仅在其所属测试折中评价一次，能量不足量回归仅在另外 44 个场景上拟合。其余三个预测量不涉及拟合。种子固定后没有依据结果筛选种子或调整划分。
3. 合并 55 个折外预测，计算总体 MAE 和 RMSE。表中的 RMSE 是 55 个误差的平方均值再开根号，不是五个折内 RMSE 的平均数。
4. 用 `default_rng(20260925)` 对 55 行场景记录进行 20,000 次有放回重采样。一次抽样使用相同的行索引评价所有预测量，保持同场景配对。用百分位数法（2.5%、97.5%，NumPy `linear` 分位数）计算 MAE、RMSE 和基线减成对成本的 MAE 差值区间。每次抽样不重新拟合模型。

这些区间以固定的折外预测和当前筛选后的合成样本为条件，只描述场景构成变化对应的误差变化。由于不同折的训练集重叠，且重采样没有重新训练，它们不包含完整的训练不确定性；也不覆盖模型选择、参考状态误差、物理模型失配或实际部署的不确定性。这里不进行显著性检验或多重比较校正。

## 补充结果

下表所有费用以 USD 计。区间是上述有条件的 95% 百分位区间。

| 预测量 | MAE | MAE 区间 | RMSE | RMSE 区间 |
|---|---:|---:|---:|---:|
| 成对约束成本 | 0.01131 | [0.00503, 0.01866] | 0.02849 | [0.01513, 0.03952] |
| 能量不足量回归 | 0.66592 | [0.53658, 0.80086] | 0.83385 | [0.69680, 0.96078] |
| 当前电价 | 0.24045 | [0.13396, 0.37134] | 0.50999 | [0.27524, 0.71065] |
| 平均预测电价 | 0.55037 | [0.45530, 0.65506] | 0.67003 | [0.53093, 0.80257] |

| 对比基线 | 基线 MAE − 成对成本 MAE | 配对差值区间 |
|---|---:|---:|
| 能量不足量回归 | 0.65461 | [0.52631, 0.78912] |
| 当前电价 | 0.22914 | [0.12449, 0.35770] |
| 平均预测电价 | 0.53906 | [0.44289, 0.64450] |

成对约束成本在这个受限样本中仍具有较低的预测误差，但其 MAE 从原 13 个留出场景的 0.00649 增至完整样本的 0.01131。两个结果使用不同的案例集合，不能把这个变化解释为模型变差或独立重复试验的差异。

这里有 24/55 个场景没有外部随机扰动，原留出集中是 4/13。成对预测与执行共享物理模型、已知电价和外部扰动，且清洁参考状态来自模拟器。这些有利条件仍然适用。因此，补充分析支持的是当前合成设定下的结果一致性，不证明真实系统中的泛化能力，也没有解决不可行案例或恢复失败案例的成本预测问题。

## 复现与输出

在项目根目录运行，使用原实验环境即可，不需要新增依赖：

```bash
uv run --project experiments python experiments/supplementary/prediction_cv/analyze.py
```

程序只读取 `experiments/results/main/` 中的 `episodes.csv`、`prediction_metrics_attacked.csv` 和 `heldout_predictions_attacked.csv`，仅在本文件夹输出文件。它校验输入文件运行前后的 SHA-256，确保原结果没有被修改。

| 文件 | 内容 |
|---|---|
| `analyze.py` | 完整复现和补充分析脚本 |
| `cohort_audit.csv` | 75 个攻击场景的纳入与排除依据 |
| `original_holdout_metrics_reproduced.csv` | 原 13 个留出场景的复现指标 |
| `oof_predictions.csv` | 55 个场景的折号、原留出身份、四种预测、逐场景有符号误差与绝对误差 |
| `fold_metrics.csv` | 每折四个预测量的 MAE、RMSE |
| `oof_metrics.csv` | 合并折外误差的指标和区间 |
| `paired_mae_improvements.csv` | 配对 MAE 差值和区间 |
| `summary.json` | 数据哈希、运行环境、种子、每折训练/测试 ID、回归系数、核验记录和局限 |
| `reproducibility.json` | 第二次运行的逐文件哈希一致性检查 |

英文论文可采用以下简洁表述，避免将这项补充评价当作新增独立证据：

> A retrospective five-fold analysis reused all 55 eligible cases, fitting the energy-deficit regression on 44 cases and evaluating it on the remaining 11 in each fold. The other three predictors were not fitted. Pooled out-of-fold MAE was USD 0.01131 for the paired cost estimate and USD 0.66592, 0.24045 and 0.55037 for the energy-deficit, current-price and mean-price baselines. Paired scenario-bootstrap intervals were conditional on the fixed predictions and selected synthetic cohort; they do not account for training or deployment uncertainty. This supplementary reuse of the original cases is not an independent validation set.
