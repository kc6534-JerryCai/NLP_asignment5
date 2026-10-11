# Mid-term equity basket 检验

生成时间：2026-10-10T19:36:43.008247+00:00

## 设计

- 收益保持原项目口径：复权开盘至收盘股票收益减去同区间 SPY 收益。
- 全样本篮子优先使用股票层面全局 BH-FDR q<0.05 的股票；严格版本进一步要求 5 日移动区块 bootstrap 区间排除 0。
- long 版本等权持有成分股；directional 版本按原股票回归 beta 的符号调整方向后等权。
- 每个 signal-specific 篮子只检验用于选择它的同一主题和指标。union 篮子作为广义 mid-term basket 检验全部实际存在的 18 个指标。
- 全样本篮子使用了同一时期的显著性结果选股，显著性属于事后构造，不能解释为独立确认。

## 全样本探索结果

- 构建 31 个篮子，共 49 个有效篮子检验。
- 在本轮篮子检验家族内统一 BH-FDR 后，q<0.05 为 28 个；其中同时通过篮子层面区块 bootstrap 为 24 个。

| Basket | Topic/index | Components | N | Effect bps/1SD | raw p | q | bootstrap |
|---|---|---:|---:|---:|---:|---:|---|
| BOOTSTRAP_SIGNAL_DIRECTIONAL__T2_democrat_stance | T2 democrat_stance | 21 | 39 | +46.47 | 6.276e-23 | 3.075e-21 | yes |
| BOOTSTRAP_TOP20_DIRECTIONAL__T2_democrat_stance | T2 democrat_stance | 20 | 39 | +47.13 | 5.11e-22 | 1.252e-20 | yes |
| BOOTSTRAP_SIGNAL_DIRECTIONAL__T4_democrat_stance | T4 democrat_stance | 62 | 31 | +57.40 | 3.128e-21 | 3.832e-20 | yes |
| BOOTSTRAP_TOP20_DIRECTIONAL__T4_democrat_stance | T4 democrat_stance | 20 | 31 | +66.82 | 2.642e-21 | 3.832e-20 | yes |
| BOOTSTRAP_SIGNAL_DIRECTIONAL__T4_topic_sentiment | T4 topic_sentiment | 10 | 53 | +52.20 | 7.028e-20 | 6.888e-19 | yes |
| BOOTSTRAP_TOP20_DIRECTIONAL__T3_topic_sentiment | T3 topic_sentiment | 20 | 50 | +47.71 | 2.649e-18 | 2.163e-17 | yes |
| BOOTSTRAP_SIGNAL_DIRECTIONAL__T3_topic_sentiment | T3 topic_sentiment | 24 | 50 | +45.22 | 4.174e-18 | 2.922e-17 | yes |
| BOOTSTRAP_SIGNAL_DIRECTIONAL__T5_topic_sentiment | T5 topic_sentiment | 19 | 50 | +57.21 | 1.047e-16 | 6.412e-16 | yes |
| BOOTSTRAP_SIGNAL_DIRECTIONAL__T3_republican_stance | T3 republican_stance | 7 | 38 | +78.64 | 8.149e-14 | 4.437e-13 | yes |
| BOOTSTRAP_SIGNAL_LONG__T2_democrat_stance | T2 democrat_stance | 21 | 39 | -29.10 | 1.084e-13 | 5.31e-13 | yes |
| BOOTSTRAP_TOP20_LONG__T2_democrat_stance | T2 democrat_stance | 20 | 39 | -28.89 | 2.193e-12 | 9.77e-12 | yes |
| BOOTSTRAP_SIGNAL_DIRECTIONAL__T4_republican_stance | T4 republican_stance | 6 | 43 | +69.84 | 1.631e-11 | 6.662e-11 | yes |
| BOOTSTRAP_SIGNAL_LONG__T3_republican_stance | T3 republican_stance | 7 | 38 | +67.71 | 3.111e-11 | 1.173e-10 | yes |
| BOOTSTRAP_SIGNAL_DIRECTIONAL__T5_republican_stance | T5 republican_stance | 4 | 35 | +103.65 | 3.192e-09 | 1.066e-08 | yes |
| FDR_SIGNAL_LONG__T4_democrat_stance | T4 democrat_stance | 147 | 31 | -29.02 | 3.264e-09 | 1.066e-08 | no |
| BOOTSTRAP_SIGNAL_DIRECTIONAL__T2_topic_sentiment | T2 topic_sentiment | 4 | 59 | +60.85 | 2.265e-08 | 6.936e-08 | yes |
| FDR_SIGNAL_LONG__T2_democrat_stance | T2 democrat_stance | 120 | 39 | -12.69 | 1.693e-07 | 4.879e-07 | no |
| BOOTSTRAP_TOP20_LONG__T3_topic_sentiment | T3 topic_sentiment | 20 | 50 | +26.49 | 3.581e-07 | 9.749e-07 | yes |
| FDR_UNION_LONG | T4 democrat_stance | 299 | 31 | -14.77 | 4.545e-07 | 1.172e-06 | no |
| BOOTSTRAP_SIGNAL_DIRECTIONAL__T2_republican_stance | T2 republican_stance | 4 | 51 | +46.25 | 1.242e-06 | 2.767e-06 | yes |
| BOOTSTRAP_SIGNAL_LONG__T2_republican_stance | T2 republican_stance | 4 | 51 | -46.25 | 1.242e-06 | 2.767e-06 | yes |
| BOOTSTRAP_SIGNAL_LONG__T3_topic_sentiment | T3 topic_sentiment | 24 | 50 | +23.87 | 1.138e-06 | 2.767e-06 | yes |
| BOOTSTRAP_SIGNAL_LONG__T4_topic_sentiment | T4 topic_sentiment | 10 | 53 | -21.26 | 7.571e-06 | 1.613e-05 | yes |
| BOOTSTRAP_TOP20_LONG__T4_democrat_stance | T4 democrat_stance | 20 | 31 | -13.33 | 2.972e-05 | 6.068e-05 | yes |
| FDR_SIGNAL_LONG__T4_topic_sentiment | T4 topic_sentiment | 11 | 53 | -15.86 | 0.0001673 | 0.000328 | yes |

## 时间顺序留出检查

- 成分选择期截至 2026-09-02；验证期从 2026-09-03 开始。验证期收益没有参与成分选择。
- early-FDR 使用选择期内按单一信号跨股票校正后的 q<0.05；early-top20 只使用选择期 HAC p 值最小的 20 只股票。
- 留出期有效检验 34 个，留出检验家族 BH-FDR q<0.05 为 5 个。由于验证期很短，最低仅要求 8 个有效信号日，全部结果仍属探索性。

| Basket | Topic/index | Components | N | Effect bps/1SD | raw p | q | bootstrap |
|---|---|---:|---:|---:|---:|---:|---|
| EARLY_FDR_DIRECTIONAL__T4_topic_sentiment | T4 topic_sentiment | 15 | 22 | -25.42 | 9.355e-06 | 0.0002051 | yes |
| EARLY_TOP20_DIRECTIONAL__T4_topic_sentiment | T4 topic_sentiment | 20 | 22 | -26.52 | 1.276e-05 | 0.0002051 | yes |
| EARLY_TOP20_DIRECTIONAL__T5_topic_sentiment | T5 topic_sentiment | 20 | 20 | -19.13 | 1.81e-05 | 0.0002051 | yes |
| EARLY_TOP20_DIRECTIONAL__T2_democrat_stance | T2 democrat_stance | 20 | 14 | -11.67 | 0.0001297 | 0.001102 | no |
| EARLY_TOP20_LONG__T2_democrat_stance | T2 democrat_stance | 20 | 14 | -22.56 | 0.000604 | 0.004107 | no |
| EARLY_FDR_LONG__T2_democrat_stance | T2 democrat_stance | 625 | 14 | -14.73 | 0.01492 | 0.08454 | yes |
| EARLY_FDR_DIRECTIONAL__T5_topic_sentiment | T5 topic_sentiment | 60 | 20 | -13.23 | 0.02414 | 0.1026 | yes |
| EARLY_TOP20_LONG__T4_republican_stance | T4 republican_stance | 20 | 17 | +23.76 | 0.02155 | 0.1026 | yes |
| EARLY_FDR_LONG__T1_topic_sentiment | T1 topic_sentiment | 8 | 13 | -28.31 | 0.04012 | 0.1516 | no |
| EARLY_TOP20_LONG__T1_topic_sentiment | T1 topic_sentiment | 20 | 13 | -25.00 | 0.09461 | 0.3217 | no |
| EARLY_FDR_DIRECTIONAL__T3_republican_stance | T3 republican_stance | 142 | 18 | -17.25 | 0.1387 | 0.4288 | no |
| EARLY_FDR_LONG__T3_topic_sentiment | T3 topic_sentiment | 48 | 21 | +8.27 | 0.1543 | 0.4373 | no |
| EARLY_FDR_LONG__T4_topic_sentiment | T4 topic_sentiment | 15 | 22 | -4.93 | 0.1722 | 0.4503 | no |
| EARLY_TOP20_DIRECTIONAL__T3_topic_sentiment | T3 topic_sentiment | 20 | 21 | -7.20 | 0.2495 | 0.6059 | no |
| EARLY_TOP20_LONG__T2_republican_stance | T2 republican_stance | 20 | 20 | -7.79 | 0.3061 | 0.6939 | no |
| EARLY_FDR_DIRECTIONAL__T1_topic_sentiment | T1 topic_sentiment | 8 | 13 | -6.13 | 0.4719 | 0.8342 | no |
| EARLY_FDR_DIRECTIONAL__T5_republican_stance | T5 republican_stance | 37 | 15 | -10.62 | 0.5152 | 0.8342 | no |
| EARLY_FDR_LONG__T5_republican_stance | T5 republican_stance | 37 | 15 | -14.91 | 0.472 | 0.8342 | no |
| EARLY_TOP20_DIRECTIONAL__T2_republican_stance | T2 republican_stance | 20 | 20 | +5.21 | 0.5062 | 0.8342 | no |
| EARLY_TOP20_DIRECTIONAL__T4_republican_stance | T4 republican_stance | 20 | 17 | +4.94 | 0.4211 | 0.8342 | no |
| EARLY_TOP20_LONG__T3_republican_stance | T3 republican_stance | 20 | 18 | +11.66 | 0.3936 | 0.8342 | no |
| EARLY_FDR_DIRECTIONAL__T2_democrat_stance | T2 democrat_stance | 625 | 14 | -0.54 | 0.6136 | 0.9483 | no |
| EARLY_FDR_LONG__T5_topic_sentiment | T5 topic_sentiment | 60 | 20 | +3.40 | 0.694 | 0.9509 | no |
| EARLY_TOP20_DIRECTIONAL__T5_republican_stance | T5 republican_stance | 20 | 15 | -6.10 | 0.6992 | 0.9509 | no |
| EARLY_TOP20_LONG__T3_topic_sentiment | T3 topic_sentiment | 20 | 21 | +3.05 | 0.6766 | 0.9509 | no |

## 解释限制

1. 全样本显著股票篮子具有明确的数据挖掘和 winner's curse；其很强的 p 值不能当作新的独立证据。
2. 留出期只有约四成交易日，稀疏新闻指标的有效 N 更少；无显著结果不等于不存在关系。
3. directional 篮子包含按历史符号做多或做空的统计因子，不是考虑手续费、借券、滑点和容量后的可交易策略。
4. Guardian 文本呈现的情绪和党派态度不代表选民偏好；相关关系也不构成因果关系。

验证状态：`pass`。
