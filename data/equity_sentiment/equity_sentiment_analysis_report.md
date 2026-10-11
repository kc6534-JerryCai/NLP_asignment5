# Guardian 政策情绪与美国股票市场调整后收益：探索性分析报告

生成时间：2026-10-10T19:23:26.773793+00:00

## 设计与覆盖

- 研究窗口：2026-07-10 至 2026-10-09；使用目标交易日开盘前固定新闻窗口。
- 股票池在读取任何收益关联前冻结为 1000 只；成功下载 999 只，其中可检验 998 只；下载失败 1 只，另有 1 只行情样本不足。
- 主指数为 6 个主题 × 3 类实际指数（议题情绪、民主党态度、共和党态度）= 18 条信号；关注度单列补充，不当作情绪。
- 主检验有效 9976 个，全部有效原始 p 值在同一个股票×指数家族中统一做 BH-FDR。HAC 固定 maxlags=5；区块 bootstrap 长度 5 个完整交易日、500 次。
- 收益为复权开盘至收盘股票收益减去相同区间 SPY 收益。直接相减隐含 beta=1，预设映射股票另做估计市场 beta 与行业 ETF 控制。

## 主结果

共有 346 个股票—指数关联在全局 BH-FDR q<0.05 后保留。

| Ticker | Topic | Index | N | Effect (bps / 1 SD) | raw p | global q |
|---|---:|---|---:|---:|---:|---:|
| WCC | T2 | Democratic stance | 39 | -75.80 | 1.644e-13 | 1.64e-09 |
| ESS | T4 | Democratic stance | 31 | -50.85 | 1.707e-12 | 8.516e-09 |
| USFD | T4 | Democratic stance | 31 | -83.88 | 9.999e-12 | 3.325e-08 |
| ITT | T2 | Democratic stance | 39 | -52.71 | 9.999e-11 | 2.228e-07 |
| LNG | T4 | Democratic stance | 31 | -61.53 | 1.117e-10 | 2.228e-07 |
| CPT | T4 | Democratic stance | 31 | -29.32 | 2.198e-10 | 3.655e-07 |
| MRNA | T2 | Democratic stance | 39 | 156.63 | 2.853e-10 | 3.836e-07 |
| PFGC | T4 | Democratic stance | 31 | -53.08 | 3.076e-10 | 3.836e-07 |
| M | T4 | Democratic stance | 31 | 78.22 | 3.623e-10 | 4.016e-07 |
| IVZ | T4 | Democratic stance | 31 | 82.57 | 5.285e-10 | 5.272e-07 |
| AJG | T4 | Democratic stance | 31 | -83.32 | 1.05e-09 | 9.524e-07 |
| TXN | T4 | Democratic stance | 31 | -71.63 | 1.269e-09 | 9.739e-07 |
| BSY | T2 | Democratic stance | 39 | -69.38 | 1.195e-09 | 9.739e-07 |
| DT | T2 | Democratic stance | 39 | -56.86 | 1.503e-09 | 1.071e-06 |
| APP | T2 | Democratic stance | 39 | 67.80 | 2.623e-09 | 1.636e-06 |
| MSI | T2 | Democratic stance | 39 | -52.75 | 2.524e-09 | 1.636e-06 |
| CMG | T4 | Democratic stance | 31 | -122.57 | 3.128e-09 | 1.835e-06 |
| JCI | T4 | Democratic stance | 31 | -48.89 | 3.698e-09 | 1.978e-06 |
| ROK | T2 | Democratic stance | 39 | -48.28 | 3.768e-09 | 1.978e-06 |
| SYY | T4 | Democratic stance | 31 | -50.67 | 5.259e-09 | 2.623e-06 |
| ARW | T2 | Democratic stance | 39 | -68.00 | 6.35e-09 | 2.987e-06 |
| ERIE | T4 | Democratic stance | 31 | -210.76 | 6.586e-09 | 2.987e-06 |
| TTC | T4 | Democratic stance | 31 | 41.20 | 7.953e-09 | 3.449e-06 |
| PCG | T2 | Democratic stance | 39 | 61.01 | 1.026e-08 | 4.264e-06 |
| CASY | T4 | Democratic stance | 31 | -94.01 | 1.268e-08 | 5.059e-06 |
| CAT | T2 | Democratic stance | 39 | -69.66 | 1.395e-08 | 5.352e-06 |
| HWM | T2 | Democratic stance | 39 | -70.43 | 1.54e-08 | 5.689e-06 |
| WMB | T2 | Democratic stance | 39 | -41.36 | 1.65e-08 | 5.878e-06 |
| TWLO | T4 | Democratic stance | 31 | -112.84 | 1.976e-08 | 6.798e-06 |
| EQH | T2 | Democratic stance | 39 | -39.23 | 2.522e-08 | 8.385e-06 |
| POOL | T4 | Democratic stance | 31 | 81.79 | 2.906e-08 | 9.352e-06 |
| AMG | T4 | Democratic stance | 31 | 58.06 | 3.63e-08 | 1.132e-05 |
| QXO | T4 | Democratic stance | 31 | 128.86 | 3.796e-08 | 1.147e-05 |
| MS | T2 | Democratic stance | 39 | 24.77 | 3.959e-08 | 1.162e-05 |
| EXC | T4 | Democratic stance | 31 | -43.90 | 4.464e-08 | 1.272e-05 |
| DGX | T4 | Democratic stance | 31 | -46.84 | 5.03e-08 | 1.394e-05 |
| ELS | T2 | Democratic stance | 39 | -35.84 | 5.585e-08 | 1.506e-05 |
| FTI | T4 | Democratic stance | 31 | -89.96 | 6.486e-08 | 1.703e-05 |
| FDS | T4 | Democratic stance | 31 | -60.03 | 6.995e-08 | 1.789e-05 |
| ACGL | T4 | Democratic stance | 31 | -39.56 | 7.289e-08 | 1.818e-05 |
| REXR | T4 | Democratic stance | 31 | -49.95 | 1.053e-07 | 2.561e-05 |
| THG | T4 | Democratic stance | 31 | -41.15 | 1.1e-07 | 2.563e-05 |
| IBM | T4 | Democratic stance | 31 | -81.42 | 1.105e-07 | 2.563e-05 |
| BRX | T4 | Democratic stance | 31 | -24.23 | 1.168e-07 | 2.649e-05 |
| DOW | T2 | Democratic stance | 39 | -43.93 | 1.264e-07 | 2.803e-05 |
| JHX | T4 | Democratic stance | 31 | 49.76 | 1.536e-07 | 3.287e-05 |
| XPO | T4 | Democratic stance | 31 | -69.03 | 1.549e-07 | 3.287e-05 |
| INTU | T4 | Democratic stance | 31 | -85.19 | 2.237e-07 | 4.649e-05 |
| SMCI | T2 | Democratic stance | 39 | 107.88 | 2.602e-07 | 5.298e-05 |
| CINF | T2 | Democratic stance | 39 | -31.09 | 2.937e-07 | 5.859e-05 |
| ODFL | T4 | Democratic stance | 31 | -47.31 | 3.098e-07 | 6.059e-05 |
| CW | T2 | Democratic stance | 39 | -62.58 | 3.286e-07 | 6.303e-05 |
| AFG | T2 | Democratic stance | 39 | -20.76 | 3.393e-07 | 6.387e-05 |
| WAT | T3 | Topic sentiment | 50 | 62.97 | 4.769e-07 | 8.81e-05 |
| PRI | T2 | Democratic stance | 39 | -29.47 | 5.757e-07 | 0.0001044 |
| CPNG | T4 | Democratic stance | 31 | -57.48 | 6.343e-07 | 0.0001111 |
| ECHO | T4 | Democratic stance | 31 | 51.96 | 6.351e-07 | 0.0001111 |
| RYAN | T2 | Democratic stance | 39 | -58.09 | 6.564e-07 | 0.0001129 |
| WMB | T4 | Democratic stance | 31 | -41.43 | 6.799e-07 | 0.0001139 |
| KIM | T4 | Democratic stance | 31 | -28.85 | 6.853e-07 | 0.0001139 |
| RDDT | T2 | Democratic stance | 39 | 97.16 | 7.684e-07 | 0.0001257 |
| WAB | T2 | Democratic stance | 39 | -45.27 | 8.288e-07 | 0.0001334 |
| INGM | T4 | Democratic stance | 31 | -61.16 | 8.598e-07 | 0.0001362 |
| LEN | T2 | Democratic stance | 39 | 43.39 | 1.053e-06 | 0.0001642 |
| ECG | T2 | Democratic stance | 39 | -75.63 | 1.113e-06 | 0.0001709 |
| GL | T2 | Democratic stance | 39 | -24.14 | 1.207e-06 | 0.0001824 |
| HPQ | T4 | Democratic stance | 31 | -97.95 | 1.355e-06 | 0.0002018 |
| AEP | T4 | Democratic stance | 31 | -28.07 | 1.668e-06 | 0.0002426 |
| ISRG | T4 | Democratic stance | 31 | 59.98 | 1.678e-06 | 0.0002426 |
| TW | T2 | Democratic stance | 39 | -27.91 | 2.097e-06 | 0.0002959 |
| LNC | T2 | Democratic stance | 39 | -27.42 | 2.106e-06 | 0.0002959 |
| LYB | T2 | Democratic stance | 39 | -39.48 | 2.462e-06 | 0.0003319 |
| RLI | T4 | Democratic stance | 31 | -62.24 | 2.45e-06 | 0.0003319 |
| ENTG | T2 | Democratic stance | 39 | 48.09 | 2.437e-06 | 0.0003319 |
| TNL | T3 | Topic sentiment | 50 | 54.49 | 3.418e-06 | 0.0004546 |
| HEI-A | T4 | Democratic stance | 31 | 37.02 | 3.472e-06 | 0.0004558 |
| PEN | T2 | Democratic stance | 39 | -9.93 | 3.616e-06 | 0.0004685 |
| IR | T2 | Democratic stance | 39 | -42.03 | 3.688e-06 | 0.0004716 |
| NXT | T2 | Democratic stance | 39 | 60.42 | 4.351e-06 | 0.0005494 |
| DBX | T2 | Democratic stance | 39 | -53.01 | 4.655e-06 | 0.0005804 |
| LHX | T4 | Democratic stance | 31 | -35.07 | 5e-06 | 0.0006158 |
| EMN | T4 | Democratic stance | 31 | 33.36 | 5.798e-06 | 0.0007013 |
| CDNS | T4 | Democratic stance | 31 | -85.58 | 5.835e-06 | 0.0007013 |
| SUI | T4 | Democratic stance | 31 | -45.79 | 6.178e-06 | 0.0007337 |
| RNR | T4 | Democratic stance | 31 | -49.42 | 6.507e-06 | 0.0007548 |
| ACGL | T2 | Democratic stance | 39 | -21.91 | 6.487e-06 | 0.0007548 |
| ENPH | T2 | Democratic stance | 39 | 66.10 | 6.832e-06 | 0.0007834 |
| SWKS | T2 | Democratic stance | 39 | 85.83 | 7.257e-06 | 0.0008227 |
| LW | T4 | Democratic stance | 31 | -61.30 | 9.07e-06 | 0.001011 |
| PFE | T2 | Democratic stance | 39 | 27.29 | 9.12e-06 | 0.001011 |
| BMRN | T2 | Democratic stance | 39 | -37.42 | 9.272e-06 | 0.001016 |
| MLI | T4 | Democratic stance | 31 | -31.33 | 9.487e-06 | 0.001029 |
| DRS | T2 | Democratic stance | 39 | -42.03 | 9.96e-06 | 0.001068 |
| DUK | T4 | Democratic stance | 31 | -38.62 | 1.093e-05 | 0.00116 |
| KR | T4 | Democratic stance | 31 | -54.41 | 1.227e-05 | 0.001288 |
| IOT | T4 | Democratic stance | 31 | -98.03 | 1.301e-05 | 0.001348 |
| MSGS | T4 | Democratic stance | 31 | -41.67 | 1.311e-05 | 0.001348 |
| COIN | T5 | Republican stance | 35 | 154.24 | 1.332e-05 | 0.001356 |
| ADI | T4 | Democratic stance | 31 | -36.41 | 1.36e-05 | 0.00137 |
| CINF | T4 | Democratic stance | 31 | -69.58 | 1.401e-05 | 0.001397 |
| JLL | T4 | Democratic stance | 31 | 77.55 | 1.428e-05 | 0.001411 |
| WMS | T3 | Topic sentiment | 50 | 34.11 | 1.471e-05 | 0.001438 |
| MUSA | T4 | Democratic stance | 31 | -65.07 | 1.497e-05 | 0.001449 |
| SO | T4 | Democratic stance | 31 | -27.61 | 1.546e-05 | 0.001483 |
| DELL | T2 | Democratic stance | 39 | 51.15 | 1.575e-05 | 0.001496 |
| CSL | T4 | Democratic stance | 31 | 46.88 | 1.663e-05 | 0.001565 |
| MSFT | T4 | Democratic stance | 31 | 34.37 | 1.801e-05 | 0.00168 |
| EIX | T2 | Democratic stance | 39 | 42.32 | 1.889e-05 | 0.001745 |
| VLO | T4 | Democratic stance | 31 | -72.85 | 2.028e-05 | 0.001856 |
| CMS | T4 | Democratic stance | 31 | -41.89 | 2.164e-05 | 0.001962 |
| OKTA | T4 | Democratic stance | 31 | -90.79 | 2.238e-05 | 0.001994 |
| KNX | T4 | Democratic stance | 31 | -46.95 | 2.239e-05 | 0.001994 |
| STZ | T2 | Democratic stance | 39 | 56.97 | 2.3e-05 | 0.002013 |
| EXR | T4 | Democratic stance | 31 | -24.43 | 2.289e-05 | 0.002013 |
| PSX | T4 | Democratic stance | 31 | -74.49 | 2.544e-05 | 0.002207 |
| AMZN | T4 | Democratic stance | 31 | 36.71 | 2.621e-05 | 0.002254 |
| NOC | T4 | Democratic stance | 31 | -31.48 | 2.664e-05 | 0.002272 |
| JAN | T4 | Topic sentiment | 53 | 38.19 | 2.713e-05 | 0.002294 |
| SHOP | T2 | Democratic stance | 39 | 37.23 | 2.909e-05 | 0.002439 |
| PNFP | T2 | Democratic stance | 39 | -43.11 | 3.061e-05 | 0.002545 |
| HII | T5 | Topic sentiment | 50 | -40.60 | 3.148e-05 | 0.002595 |
| BBWI | T4 | Democratic stance | 31 | 63.30 | 3.203e-05 | 0.002619 |
| KMI | T2 | Democratic stance | 39 | -25.45 | 3.232e-05 | 0.002621 |
| EMR | T4 | Topic sentiment | 53 | 42.97 | 3.318e-05 | 0.002669 |
| TLN | T2 | Democratic stance | 39 | 78.95 | 3.555e-05 | 0.002837 |
| GPC | T4 | Democratic stance | 31 | -33.52 | 3.835e-05 | 0.003036 |
| SSNC | T4 | Democratic stance | 31 | 31.91 | 3.968e-05 | 0.003076 |
| ATO | T2 | Democratic stance | 39 | -14.83 | 3.949e-05 | 0.003076 |
| AMP | T2 | Democratic stance | 39 | -29.68 | 3.978e-05 | 0.003076 |
| ORLY | T4 | Democratic stance | 31 | -70.04 | 4.191e-05 | 0.003216 |
| BRX | T4 | Topic sentiment | 53 | 19.26 | 4.459e-05 | 0.003396 |
| FRT | T4 | Topic sentiment | 53 | 16.50 | 4.549e-05 | 0.003412 |
| JAN | T4 | Republican stance | 43 | -63.54 | 4.532e-05 | 0.003412 |
| AEE | T4 | Democratic stance | 31 | -19.66 | 4.651e-05 | 0.003463 |
| TRGP | T3 | Topic sentiment | 50 | -39.02 | 4.96e-05 | 0.003665 |
| LSTR | T4 | Democratic stance | 31 | -41.30 | 5.012e-05 | 0.003676 |
| CDW | T4 | Democratic stance | 31 | -106.81 | 5.066e-05 | 0.003688 |
| ASND | T2 | Democratic stance | 39 | 38.76 | 5.101e-05 | 0.003688 |
| KTOS | T5 | Topic sentiment | 50 | -97.38 | 5.167e-05 | 0.003708 |
| CAG | T4 | Democratic stance | 31 | 50.56 | 5.41e-05 | 0.003855 |
| COLD | T4 | Democratic stance | 31 | -29.80 | 5.597e-05 | 0.00396 |
| MELI | T4 | Democratic stance | 31 | -49.16 | 5.961e-05 | 0.004188 |
| CHTR | T2 | Democratic stance | 39 | -65.64 | 6.281e-05 | 0.004382 |
| DIS | T3 | Topic sentiment | 50 | 34.72 | 6.534e-05 | 0.004527 |
| WELL | T4 | Democratic stance | 31 | -41.90 | 6.903e-05 | 0.00475 |
| CCL | T3 | Topic sentiment | 50 | 57.80 | 7.112e-05 | 0.004859 |
| RRX | T5 | Topic sentiment | 50 | 49.26 | 8.12e-05 | 0.00551 |
| RVMD | T2 | Democratic stance | 39 | 38.14 | 8.316e-05 | 0.005605 |
| CHD | T2 | Democratic stance | 39 | -12.33 | 8.417e-05 | 0.005636 |
| AHR | T4 | Democratic stance | 31 | -21.72 | 8.718e-05 | 0.005782 |
| NET | T2 | Democratic stance | 39 | -107.30 | 8.751e-05 | 0.005782 |
| DRS | T4 | Democratic stance | 31 | -44.12 | 8.852e-05 | 0.00581 |
| OGE | T4 | Democratic stance | 31 | -24.31 | 9.118e-05 | 0.005945 |
| GL | T4 | Democratic stance | 31 | 34.30 | 9.235e-05 | 0.005982 |
| EQH | T4 | Democratic stance | 31 | 62.54 | 9.443e-05 | 0.006077 |
| CAVA | T4 | Democratic stance | 31 | -58.15 | 9.847e-05 | 0.006282 |
| RS | T3 | Topic sentiment | 50 | 34.17 | 0.0001012 | 0.006282 |
| CCC | T5 | Topic sentiment | 50 | -73.82 | 0.0001005 | 0.006282 |
| TPL | T2 | Democratic stance | 39 | -104.68 | 0.0001004 | 0.006282 |
| CEG | T2 | Democratic stance | 39 | 51.26 | 0.0001014 | 0.006282 |
| GTLB | T5 | Republican stance | 35 | -90.29 | 9.973e-05 | 0.006282 |
| FERG | T4 | Democratic stance | 31 | -45.25 | 0.0001035 | 0.006372 |
| DAL | T3 | Topic sentiment | 50 | 40.52 | 0.0001045 | 0.006397 |
| FIGR | T4 | Topic sentiment | 53 | -144.14 | 0.0001093 | 0.006648 |
| SNX | T4 | Democratic stance | 31 | -57.76 | 0.0001137 | 0.006811 |
| KKR | T4 | Democratic stance | 31 | 53.96 | 0.0001134 | 0.006811 |
| KD | T2 | Democratic stance | 39 | -43.24 | 0.000114 | 0.006811 |
| CRM | T4 | Democratic stance | 31 | -73.50 | 0.0001197 | 0.007068 |
| VTRS | T2 | Democratic stance | 39 | -21.15 | 0.0001191 | 0.007068 |
| AVT | T2 | Democratic stance | 39 | -39.27 | 0.0001239 | 0.007268 |
| SARO | T2 | Democratic stance | 39 | -51.74 | 0.0001251 | 0.007292 |
| JNJ | T4 | Democratic stance | 31 | -38.72 | 0.0001257 | 0.007292 |
| CRL | T3 | Topic sentiment | 50 | 73.48 | 0.0001265 | 0.007294 |
| PSA | T4 | Democratic stance | 31 | -38.06 | 0.0001272 | 0.007294 |
| BJ | T2 | Democratic stance | 39 | -28.58 | 0.0001338 | 0.007548 |
| BPOP | T2 | Democratic stance | 39 | -18.17 | 0.0001339 | 0.007548 |
| CRL | T4 | Democratic stance | 31 | -62.38 | 0.0001324 | 0.007548 |
| MSI | T4 | Democratic stance | 31 | -34.72 | 0.0001406 | 0.007878 |
| WEC | T4 | Democratic stance | 31 | -26.77 | 0.0001445 | 0.008056 |
| MAR | T2 | Democratic stance | 39 | -32.98 | 0.0001468 | 0.008136 |
| OWL | T2 | Democratic stance | 39 | 47.54 | 0.0001559 | 0.008545 |
| DT | T4 | Democratic stance | 31 | -46.56 | 0.0001554 | 0.008545 |
| PPL | T4 | Democratic stance | 31 | -22.46 | 0.0001699 | 0.009262 |
| JAN | T4 | Democratic stance | 31 | -66.57 | 0.0001819 | 0.009809 |
| FE | T4 | Democratic stance | 31 | -19.92 | 0.0001817 | 0.009809 |
| FOXA | T2 | Democratic stance | 39 | 44.48 | 0.0001861 | 0.009984 |
| FNB | T4 | Democratic stance | 31 | 18.57 | 0.0001887 | 0.01007 |
| ASTS | T3 | Republican stance | 38 | 135.12 | 0.0001903 | 0.0101 |
| EVRG | T4 | Democratic stance | 31 | -16.08 | 0.0001965 | 0.01032 |
| SNOW | T2 | Topic sentiment | 59 | 82.37 | 0.0001954 | 0.01032 |
| ETN | T2 | Democratic stance | 39 | -25.79 | 0.0001985 | 0.01037 |
| ROL | T4 | Democratic stance | 31 | -29.01 | 0.0002032 | 0.01056 |
| KNSL | T4 | Democratic stance | 31 | -72.82 | 0.0002045 | 0.01057 |
| GTLB | T4 | Democratic stance | 31 | -66.14 | 0.0002086 | 0.01068 |
| FISV | T4 | Democratic stance | 31 | -43.00 | 0.0002087 | 0.01068 |
| BAM | T4 | Democratic stance | 31 | 59.62 | 0.0002154 | 0.01096 |
| TEAM | T3 | Topic sentiment | 50 | -87.85 | 0.000217 | 0.01099 |
| RBLX | T2 | Democratic stance | 39 | 90.26 | 0.0002197 | 0.01101 |
| CUBE | T4 | Democratic stance | 31 | -18.35 | 0.0002189 | 0.01101 |
| MSTR | T2 | Democratic stance | 39 | -67.75 | 0.0002299 | 0.01145 |
| PGR | T4 | Democratic stance | 31 | -44.87 | 0.0002306 | 0.01145 |
| MBGL | T4 | Republican stance | 43 | 98.83 | 0.0002343 | 0.01157 |
| SNX | T2 | Democratic stance | 39 | -51.28 | 0.0002365 | 0.01162 |
| AYI | T4 | Democratic stance | 31 | 27.17 | 0.000243 | 0.01177 |
| O | T4 | Democratic stance | 31 | -32.27 | 0.0002426 | 0.01177 |
| PRU | T4 | Republican stance | 43 | -33.35 | 0.0002415 | 0.01177 |
| NFLX | T2 | Democratic stance | 39 | 22.31 | 0.0002605 | 0.01256 |
| CAVA | T3 | Republican stance | 38 | 128.23 | 0.0002634 | 0.01257 |
| RJF | T2 | Democratic stance | 39 | -24.82 | 0.0002624 | 0.01257 |
| CLX | T2 | Democratic stance | 39 | -32.23 | 0.0002698 | 0.01282 |
| SHW | T4 | Democratic stance | 31 | 22.58 | 0.0002752 | 0.01301 |
| RF | T2 | Republican stance | 51 | -39.93 | 0.0002804 | 0.0132 |
| ESAB | T2 | Democratic stance | 39 | -74.20 | 0.0002846 | 0.01333 |
| SON | T5 | Topic sentiment | 50 | 42.34 | 0.0002944 | 0.01366 |
| NEM | T4 | Topic sentiment | 53 | -29.31 | 0.000294 | 0.01366 |
| EG | T4 | Democratic stance | 31 | -43.83 | 0.0003071 | 0.01418 |
| UHS | T4 | Topic sentiment | 53 | -75.25 | 0.0003085 | 0.01418 |
| EEFT | T2 | Democratic stance | 39 | -27.46 | 0.0003151 | 0.01442 |
| BRO | T4 | Democratic stance | 31 | -32.98 | 0.0003192 | 0.01446 |
| RGA | T5 | Topic sentiment | 50 | 36.88 | 0.0003214 | 0.01446 |
| AXS | T2 | Democratic stance | 39 | -16.87 | 0.0003217 | 0.01446 |
| SMCI | T5 | Topic sentiment | 50 | -75.19 | 0.0003178 | 0.01446 |
| OXY | T4 | Democratic stance | 31 | -43.69 | 0.0003294 | 0.01474 |
| ADC | T2 | Democratic stance | 39 | -22.03 | 0.0003364 | 0.01498 |
| LECO | T3 | Topic sentiment | 50 | 49.58 | 0.0003453 | 0.01531 |
| MRSH | T5 | Topic sentiment | 50 | -24.92 | 0.0003613 | 0.01595 |
| COF | T2 | Republican stance | 51 | -46.61 | 0.000365 | 0.01604 |
| FHN | T2 | Democratic stance | 39 | -26.64 | 0.000377 | 0.0165 |
| AA | T4 | Republican stance | 43 | 92.77 | 0.0003795 | 0.01653 |
| FWONA | T2 | Democratic stance | 39 | 39.79 | 0.0003853 | 0.01671 |
| ALK | T4 | Democratic stance | 31 | 49.26 | 0.0003872 | 0.01672 |
| DOW | T4 | Democratic stance | 31 | -50.80 | 0.0003932 | 0.0169 |
| ST | T3 | Topic sentiment | 50 | 61.16 | 0.0003947 | 0.0169 |
| EQIX | T2 | Democratic stance | 39 | -27.17 | 0.0003965 | 0.0169 |
| GOOGL | T3 | Topic sentiment | 50 | 30.52 | 0.0004002 | 0.01699 |
| DAR | T4 | Democratic stance | 31 | -54.68 | 0.0004096 | 0.01731 |
| C | T2 | Democratic stance | 39 | 20.14 | 0.0004154 | 0.01748 |
| GEO | T2 | Democratic stance | 39 | 36.23 | 0.000423 | 0.01773 |
| LFUS | T2 | Democratic stance | 39 | -33.24 | 0.0004304 | 0.01796 |
| BOKF | T2 | Democratic stance | 39 | -21.98 | 0.0004346 | 0.01806 |
| LH | T4 | Democratic stance | 31 | -49.19 | 0.0004633 | 0.0191 |
| DTM | T2 | Democratic stance | 39 | -24.72 | 0.0004629 | 0.0191 |
| HR | T2 | Democratic stance | 39 | -26.52 | 0.0004667 | 0.01916 |
| CBRE | T4 | Democratic stance | 31 | 108.95 | 0.0004765 | 0.01948 |
| ALGN | T5 | Topic sentiment | 50 | 44.10 | 0.0004806 | 0.01957 |
| APLD | T2 | Democratic stance | 39 | -48.07 | 0.0004963 | 0.02013 |
| EHC | T4 | Topic sentiment | 53 | -40.66 | 0.0005014 | 0.02025 |
| RJF | T3 | Topic sentiment | 50 | 32.02 | 0.0005095 | 0.02033 |
| CR | T4 | Democratic stance | 31 | 45.38 | 0.000509 | 0.02033 |
| GAP | T2 | Democratic stance | 39 | 24.31 | 0.0005079 | 0.02033 |
| MORN | T2 | Democratic stance | 39 | 27.38 | 0.0005301 | 0.02107 |
| VSNT | T4 | Democratic stance | 31 | 39.66 | 0.0005327 | 0.02109 |
| FAST | T4 | Democratic stance | 31 | -27.52 | 0.0005448 | 0.02142 |
| IRM | T4 | Republican stance | 43 | -63.87 | 0.0005454 | 0.02142 |
| CCI | T2 | Democratic stance | 39 | 26.04 | 0.0005656 | 0.02213 |
| MET | T2 | Democratic stance | 39 | -36.04 | 0.0005761 | 0.02245 |
| FIG | T2 | Democratic stance | 39 | -85.33 | 0.0005814 | 0.02257 |
| RS | T5 | Topic sentiment | 50 | 35.90 | 0.0005941 | 0.02297 |
| MBGL | T4 | Democratic stance | 31 | 86.59 | 0.0006009 | 0.02315 |
| IESC | T2 | Democratic stance | 39 | -108.47 | 0.0006131 | 0.02353 |
| BIIB | T5 | Topic sentiment | 50 | 56.19 | 0.0006192 | 0.02367 |
| BRO | T5 | Topic sentiment | 50 | -45.26 | 0.0006216 | 0.02367 |
| VNO | T2 | Democratic stance | 39 | 24.30 | 0.0006282 | 0.02383 |
| DPZ | T2 | Democratic stance | 39 | -37.00 | 0.0006402 | 0.02419 |
| AIG | T4 | Democratic stance | 31 | -36.69 | 0.0006428 | 0.0242 |
| ABBV | T2 | Democratic stance | 39 | 16.04 | 0.0006537 | 0.02452 |
| NEU | T4 | Topic sentiment | 53 | 37.91 | 0.0006637 | 0.02471 |
| WFRD | T2 | Democratic stance | 39 | -69.59 | 0.0006617 | 0.02471 |
| LNC | T4 | Topic sentiment | 53 | 38.05 | 0.0006919 | 0.02538 |
| VRTX | T3 | Topic sentiment | 50 | 29.03 | 0.0006915 | 0.02538 |
| BLK | T3 | Republican stance | 38 | -38.24 | 0.0006853 | 0.02538 |
| FTNT | T5 | Republican stance | 35 | -91.54 | 0.000688 | 0.02538 |
| W | T4 | Democratic stance | 31 | 64.93 | 0.0006948 | 0.02539 |
| DTE | T4 | Democratic stance | 31 | -21.41 | 0.0006993 | 0.02546 |
| ORI | T4 | Democratic stance | 31 | -30.07 | 0.0007291 | 0.02645 |
| ED | T4 | Democratic stance | 31 | -28.59 | 0.0007386 | 0.0265 |
| MPWR | T4 | Democratic stance | 31 | -119.58 | 0.0007386 | 0.0265 |
| SYY | T3 | Republican stance | 38 | 42.67 | 0.0007358 | 0.0265 |
| MLI | T2 | Democratic stance | 39 | -57.80 | 0.0007454 | 0.0266 |
| NOV | T2 | Democratic stance | 39 | -37.71 | 0.0007492 | 0.0266 |
| ELAN | T5 | Topic sentiment | 50 | 71.70 | 0.0007485 | 0.0266 |
| SNPS | T3 | Republican stance | 38 | 82.68 | 0.000753 | 0.02664 |
| DUOL | T2 | Democratic stance | 39 | 93.55 | 0.0007698 | 0.02714 |
| TEL | T4 | Democratic stance | 31 | -42.02 | 0.0007733 | 0.02716 |
| DBX | T2 | Topic sentiment | 59 | 86.39 | 0.0007814 | 0.02735 |
| LW | T5 | Topic sentiment | 50 | -68.35 | 0.0008102 | 0.02826 |
| CUBE | T3 | Topic sentiment | 50 | -23.40 | 0.0008206 | 0.02852 |
| AOS | T3 | Topic sentiment | 50 | 27.31 | 0.0008357 | 0.02895 |
| VRSN | T5 | Republican stance | 35 | -78.52 | 0.0008402 | 0.029 |
| PCG | T3 | Topic sentiment | 50 | -61.91 | 0.0008647 | 0.02975 |
| THC | T4 | Topic sentiment | 53 | -77.94 | 0.0008701 | 0.02983 |
| SYF | T2 | Republican stance | 51 | -54.65 | 0.0008744 | 0.02987 |
| JNJ | T2 | Democratic stance | 39 | 16.92 | 0.0008836 | 0.03008 |
| NFG | T2 | Topic sentiment | 59 | -36.87 | 0.0008955 | 0.03039 |
| CAH | T2 | Democratic stance | 39 | -20.25 | 0.0009128 | 0.03076 |
| RMD | T2 | Democratic stance | 39 | 20.88 | 0.0009114 | 0.03076 |
| DECK | T4 | Democratic stance | 31 | -24.97 | 0.0009213 | 0.03095 |
| CB | T4 | Democratic stance | 31 | -42.28 | 0.0009358 | 0.03133 |
| GEV | T2 | Democratic stance | 39 | -39.60 | 0.0009569 | 0.03193 |
| CDNS | T3 | Republican stance | 38 | 82.13 | 0.0009872 | 0.03283 |
| LHX | T2 | Democratic stance | 39 | -28.63 | 0.0009993 | 0.03312 |
| TRU | T2 | Democratic stance | 39 | -36.74 | 0.001019 | 0.03366 |
| JLL | T2 | Democratic stance | 39 | -22.11 | 0.001024 | 0.0337 |
| FTNT | T4 | Democratic stance | 31 | -58.85 | 0.001047 | 0.03437 |
| NUE | T5 | Topic sentiment | 50 | 61.82 | 0.001058 | 0.03462 |
| INSM | T2 | Democratic stance | 39 | -41.91 | 0.001076 | 0.03504 |
| LBTYA | T4 | Republican stance | 43 | 66.69 | 0.001082 | 0.03504 |
| WWD | T2 | Democratic stance | 39 | -36.85 | 0.00108 | 0.03504 |
| VTRS | T3 | Topic sentiment | 50 | 61.65 | 0.00109 | 0.03508 |
| OC | T5 | Topic sentiment | 50 | 40.81 | 0.001088 | 0.03508 |
| MFP | T2 | Democratic stance | 39 | -98.10 | 0.001103 | 0.03539 |
| AYI | T3 | Topic sentiment | 50 | 58.40 | 0.00111 | 0.0355 |
| NOW | T4 | Democratic stance | 31 | -48.66 | 0.001128 | 0.03595 |
| CPAY | T2 | Democratic stance | 39 | -33.42 | 0.001139 | 0.03619 |
| GSAT | T3 | Topic sentiment | 50 | -11.02 | 0.001157 | 0.03664 |
| GOOGL | T4 | Democratic stance | 31 | 60.29 | 0.00117 | 0.03689 |
| FLS | T3 | Republican stance | 38 | 41.43 | 0.001172 | 0.03689 |
| FANG | T4 | Democratic stance | 31 | -36.76 | 0.00118 | 0.03691 |
| ILMN | T4 | Democratic stance | 31 | -94.26 | 0.001179 | 0.03691 |
| BROS | T2 | Democratic stance | 39 | -23.32 | 0.00119 | 0.0371 |
| ARMK | T4 | Democratic stance | 31 | -42.33 | 0.001195 | 0.03713 |
| VIRT | T5 | Topic sentiment | 50 | -108.85 | 0.001238 | 0.03835 |
| MANH | T4 | Democratic stance | 31 | -87.54 | 0.001244 | 0.03842 |
| CRDO | T2 | Democratic stance | 39 | 66.34 | 0.00125 | 0.03848 |
| A | T4 | Democratic stance | 31 | -43.24 | 0.00127 | 0.03897 |
| NKE | T3 | Topic sentiment | 50 | 44.49 | 0.001314 | 0.04022 |
| AA | T2 | Democratic stance | 39 | 48.58 | 0.001321 | 0.0403 |
| STLD | T5 | Topic sentiment | 50 | 71.78 | 0.001342 | 0.04082 |
| INGM | T2 | Democratic stance | 39 | -32.93 | 0.001347 | 0.04084 |
| TRGP | T4 | Democratic stance | 31 | -31.42 | 0.001383 | 0.0418 |
| DDOG | T4 | Democratic stance | 31 | -83.44 | 0.001399 | 0.04216 |
| TIGO | T4 | Democratic stance | 31 | -48.86 | 0.001407 | 0.04227 |
| APD | T4 | Democratic stance | 31 | -36.41 | 0.001418 | 0.04249 |
| COKE | T3 | Topic sentiment | 50 | 42.69 | 0.001468 | 0.04386 |
| LNG | T2 | Democratic stance | 39 | -58.18 | 0.00153 | 0.04556 |
| MDGL | T4 | Democratic stance | 31 | 69.45 | 0.001545 | 0.04587 |
| ADC | T4 | Democratic stance | 31 | -38.95 | 0.001552 | 0.04593 |
| W | T2 | Democratic stance | 39 | 33.63 | 0.00156 | 0.04603 |
| CSL | T5 | Topic sentiment | 50 | 41.90 | 0.001595 | 0.04695 |
| NFG | T3 | Topic sentiment | 50 | -32.96 | 0.001618 | 0.04746 |
| NEU | T2 | Democratic stance | 39 | -22.09 | 0.001627 | 0.04759 |
| HEI-A | T2 | Republican stance | 51 | -43.81 | 0.001655 | 0.04827 |
| AMT | T2 | Topic sentiment | 59 | -37.76 | 0.001705 | 0.04938 |
| HCA | T2 | Democratic stance | 39 | 24.30 | 0.001698 | 0.04938 |
| SHOP | T4 | Democratic stance | 31 | -85.56 | 0.001708 | 0.04938 |
| RNG | T4 | Democratic stance | 31 | -81.37 | 0.001718 | 0.04954 |

- q<0.05：346；0.05≤q<0.10：130；原始 p<0.05 但 q≥0.10：759。
- q<0.05 的 346 个结果中，161 个的 5 日移动区块 bootstrap 95% 区间也排除 0；其余结果对连续交易日共同冲击和稀疏信号的处理不稳，不能称为稳健发现。
- 效应量是情绪指数增加一个样本标准差对应的市场调整后开盘—收盘收益基点数；保留原始方向，不把统计显著等同于经济或因果重要性。
- 广泛显著结果主要集中在 T2 与 T4 的民主党态度。现有标注中民主党态度几乎没有 `oppose`，许多窗口是 0，少数窗口为正；因此这些回归更接近“少数出现正面民主党表达的日期”与其他日期的比较，容易同时反映共同市场日冲击。

## 信号覆盖

| Topic | Index | Valid pre-open trading-day windows |
|---:|---|---:|
| T0 | Democratic stance | 10 |
| T0 | Republican stance | 11 |
| T0 | Topic sentiment | 23 |
| T1 | Democratic stance | 14 |
| T1 | Republican stance | 18 |
| T1 | Topic sentiment | 29 |
| T2 | Democratic stance | 39 |
| T2 | Republican stance | 51 |
| T2 | Topic sentiment | 59 |
| T3 | Democratic stance | 15 |
| T3 | Republican stance | 38 |
| T3 | Topic sentiment | 50 |
| T4 | Democratic stance | 31 |
| T4 | Republican stance | 43 |
| T4 | Topic sentiment | 53 |
| T5 | Democratic stance | 22 |
| T5 | Republican stance | 35 |
| T5 | Topic sentiment | 50 |

## 稳健性、联合模型与异常值

- 预设业务映射共 126 个股票—主题—指数稳健性模型。方向在市场控制后保持的可估计记录数：70；加入预设行业 ETF 后保持：60。
- `predetermined_robustness_results.csv` 同时保留完整样本、市场控制、行业控制以及 Cook 距离/极端收益敏感性，不因删除某日后显著而替换主结果。
- 联合模型来自分析前保存的多主题业务映射；样本交集不足时按预设门槛跳过，不强行估计。

## 样本外预测

- 全部预设单指数模型均保留；至少 15 次预测的可评估模型 4986 个，其中 RMSE 同时低于零收益与历史均值基准 565 个。
- 负的样本外 R² 原样保留。这里是预测检验，不是可成交交易收益；未构造手续费、点差、滑点或做空假设下的盈利曲线。
- 若只展示全样本 q<0.05 后再挑出的模型，会产生事后选择；相关明细虽保留，但不当作独立样本外确认。

## 固定主题篮子

- 六个篮子均由分析前业务映射固定并等权汇总，逐日记录有效成分数；没有使用事后显著股票重新组篮子。
- 篮子有效检验 10 个，单独 BH-FDR 后 q<0.05 为 2 个。篮子样本外可评估模型 5 个。

## 解释限制

1. 仅约三个月、约六十多个交易日；稀疏的 T0/T1 和党派态度信号使很多模型不足 30 个配对。
2. 指数来自 Guardian 政策报道中呈现的评价与党派态度，不代表全体选民或客观政策状态；媒体选题和标注分布均可能带来偏差。
3. IWB、S&P 500 和 Nasdaq-100 名单来自 2026-10 附近的可公开快照而非 7 月 10 日历史成分，存在生存者偏差；IWB 是 Russell 1000 的透明代理而非授权指数成分文件。
4. 大范围 1,000 股票筛选是探索性的；即使 FDR 校正也不能消除所有数据挖掘、行业共同冲击和测量误差。
5. HAC 与区块 bootstrap 在短样本和不规则缺失下仍是近似推断；相关、条件相关或样本外预测表现均不构成因果证据。
6. 既有 Democratic Sweep 不显著结果保留在 `data/sweep_analysis/`，本模块没有修改其方法或输出。

验证状态：`pass`。详细检查见 `equity_sentiment_validation_report.json`。
