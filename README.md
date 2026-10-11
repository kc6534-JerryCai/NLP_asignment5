# Guardian 美国中期选举新闻采集与 LDA 主题分析

项目包含两个独立阶段：Guardian 新闻采集与基于已采集语料的 LDA 主题分析。本阶段不调用 DeepSeek，也不执行情绪打分、股票分析或选举概率回归。

## 官方 API 约束

程序使用 Guardian Open Platform 的 `https://content.guardianapis.com/search` 端点：

- 认证参数为 `api-key`；密钥只从 `GUARDIAN_API_KEY` 环境变量或本地 `.env` 读取。
- 使用 `show-fields=headline,trailText,standfirst,byline,bodyText` 请求标题、摘要、作者和全文。`bodyText` 为空时保留空正文并设置 `body_missing=true`，不会用摘要代替全文。
- 使用 `from-date`、`to-date`（均包含端点日期）、`page`、`page-size` 和 `order-by=oldest`；每页最多 200 条。
- 开发者访问当前上限为每秒 1 次、每天 500 次。程序将请求间隔设为至少 1.05 秒，默认本地日上限 475 次，并对 429、5xx 和超时做有限退避重试。

参考：

- https://open-platform.theguardian.com/documentation/
- https://open-platform.theguardian.com/access/

## 配置

需要 Python 3.9 或更高版本，无第三方依赖。

PowerShell：

```powershell
Copy-Item .env.example .env
# 然后仅在本机编辑 .env：GUARDIAN_API_KEY=你的密钥
python fetch_guardian_news.py --smoke-test
```

也可只为当前 PowerShell 会话设置：

```powershell
$env:GUARDIAN_API_KEY = "你的密钥"
```

不要把密钥粘贴到聊天、代码、日志或输出文件中。`.env` 已被 `.gitignore` 排除。

## 运行

默认日期为 2026-07-10 至 2026-10-09（含起止日）：

```powershell
python fetch_guardian_news.py
```

自定义日期：

```powershell
python fetch_guardian_news.py --from-date 2026-08-01 --to-date 2026-08-31
```

建议首次先执行 `--smoke-test`。它以每页 2 条请求最多两页，检查正文、UTC/纽约时区转换和分页，不写入正式采集进度。正式采集默认按 7 天拆分日期区间，每个区间分别执行七组查询。

## 分页、请求预算与断点续采

每成功保存一页后，程序会原子更新 `data/collection_progress.json`，并将已获取记录增量写入被 Git 忽略的 `data/.state/records.jsonl`。再次执行相同命令会跳过已完成任务并从 `next_page` 继续。

本地请求台账 `data/.state/request_ledger.json` 记录 UTC 日期内由本程序发出的每一次尝试（包括重试）。正式采集进度还维护累计请求尝试数，断点续采后汇总不会把先前调用遗漏。达到默认日保护线或单次运行预算时，程序保存现有成果，将剩余任务列入汇总中的 `incomplete_tasks`，并以非零状态退出。下一次直接运行相同命令即可继续。

可调整但不建议放宽到超过账户额度：

```powershell
python fetch_guardian_news.py --run-request-budget 300 --daily-request-limit 475
```

若日期、查询、分段天数或页大小改变，旧进度不能混用。使用 `--fresh` 会把旧进度和增量记录移动到带时间戳的归档目录，而不是删除：

```powershell
python fetch_guardian_news.py --from-date 2026-09-01 --to-date 2026-10-09 --fresh
```

## 查询和基础筛选

查询围绕中期选举、国会选举、民主党、共和党、Congress、House 和 Senate，且用美国党派、选举或机构语境约束容易误命中的词。没有以通胀、移民、医疗等预设政策议题限定搜索。

全部去重后的查询命中保存在 `guardian_news.*`；基础筛选把记录标为：

- `relevant`：明确出现中期选举，或同时满足选举与美国政党/联邦机构语境。
- `uncertain`：语境部分满足，应人工抽查。
- `likely_irrelevant`：基础规则认为上下文不足，但仍保留在完整数据中。

`guardian_news_filtered.*` 只包含 `relevant` 和 `uncertain`。这不是模型分类，不应用作最终相关性标签。

## 输出

- `data/raw/<日期区间+查询>/page_*.json`：每页原始 API 响应及不含密钥的请求元数据。
- `data/guardian_news.jsonl`、`data/guardian_news.csv`：全部去重后的查询命中。
- `data/guardian_news_filtered.jsonl`、`data/guardian_news_filtered.csv`：基础筛选结果。
- `data/collection_progress.json`：区间、查询、页码、状态和已获取 article_id。
- `data/collection_summary.json`：覆盖范围、请求/记录统计、逐日数量与正文缺失率、未完成任务和固定随机种子的约 10 篇抽样。
- `data/smoke_test_report.json`：小规模验证报告。

CSV 使用带 BOM 的 UTF-8（`utf-8-sig`），便于 Excel 正确识别。JSONL 与 CSV 字段一致；CSV 中的 `matched_queries` 用 JSON 数组字符串表示。

去重先合并相同 `article_id`，再合并规范化 URL，并汇总所有命中的查询表达式。完全相同或正文五词片段重叠度极高的同日文章只增加 `duplicate_body` 与 `duplicate_of_article_id` 标记，不会因报道同一事件而被自动删除。

## LDA 分析环境

LDA 阶段使用单独的 Conda 环境 `assignment5_guardian_lda`。`.venv` 是 Python 标准库创建的另一种项目内虚拟环境；本项目没有使用它，避免和 Conda 环境混用。

```powershell
conda env create -f environment.yml
conda activate assignment5_guardian_lda
```

环境包含 Python 3.12、spaCy、`en_core_web_sm`、Gensim、pandas、SciPy 和 Matplotlib。若选择用 pip 安装 `requirements.txt`，还需显式安装 spaCy 英文模型：

```powershell
python -m spacy download en_core_web_sm
```

## LDA 运行方式

配置位于 `lda_config.yaml`。分析脚本优先且仅读取 `data/guardian_news.jsonl`，不会再合并同批 CSV。完整运行：

```powershell
conda activate assignment5_guardian_lda
python analyze_guardian_lda.py --stage all
```

也可分阶段运行并在检查候选模型后调整 `selected_k` 与人工主题名：

```powershell
python analyze_guardian_lda.py --stage selection
python analyze_guardian_lda.py --stage candidates
python analyze_guardian_lda.py --stage final
```

- `selection`：数据审计、上下文相关性筛选、人工抽查清单和排除原因。
- `candidates`：spaCy 词形还原、Gensim 双词短语、词典/词袋以及 K=5/8/10 候选模型。
- `final`：最终主题概率、主题关注度、两个额外随机种子的稳定性检查、图表和模型重载验证。

每篇文章是一个 document，标题不重复加入正文。主模型使用 `doc2bow` 词频而非 TF-IDF；少于 30 个有效 tokens、过滤后空词袋、正文缺失、明确无关或重复正文的记录不进入训练。原始采集数据不会被改写。

## LDA 输出

表格和报告位于 `data/lda/`，包括：

- `sample_selection_report.json`、`manual_review_articles.csv`、`uncertain_articles.csv`、`excluded_articles.csv`
- `model_comparison.csv`、`candidate_topic_diagnostics.csv`
- `topic_keywords.csv`、`topic_representative_articles.csv`、`topic_labels.csv`
- `article_topic_probabilities.csv`、`overall_topic_attention.csv`
- `daily_topic_attention.csv`、`daily_topic_attention_7d.csv`
- `topic_stability.csv`、`validation_report.json`、`lda_analysis_report.md`

模型、词典、词袋、短语模型和训练文档索引位于 `models/lda/`；三类 PNG 图位于 `figures/lda/`。CSV 均使用适合 Excel 打开的 UTF-8 BOM。每日关注度按纽约当地日期计算，无有效文章的日期保留缺失值，不填成零；7 日均值仅用于展示。

本分析是对完整三个月 Guardian 新闻样本的事后描述，不能解释为全体美国选民的议题偏好，也不是无前视的实时预测模型。主题名称是依据关键词和代表文章提出的人工解释，仍需人工确认。

## 实质政策议题模型

为避免竞选程序、候选人动态和政治机构词汇挤占主题，项目另提供实质政策分析脚本 `analyze_guardian_policy_lda.py` 及配置 `policy_lda_config.yaml`。主模型只使用 `policy_substance` 记录；`mixed`（政策与竞选并存）记录单独保留，供敏感性分析。该筛选不会修改原始 Guardian 数据。

```powershell
conda activate assignment5_guardian_lda
python analyze_guardian_policy_lda.py --stage all
```

也可分阶段执行 `selection`、`candidates` 和 `final`。政策模型尝试 K=4、5、6、7、8，并依据 coherence、关键词重叠、代表文章可解释性和两个额外随机种子的稳定性综合选择 K。`ai` 被明确保留，避免短词长度规则误删。

- 数据表、审核清单和验证报告：`data/lda_policy/`
- LDA 模型、词典、语料和短语模型：`models/lda_policy/`
- coherence、总体关注度和每日关注度图：`figures/lda_policy/`

政策关注度是 Guardian 政策报道样本内的平均主题概率，不代表全体美国选民的偏好。完整三个月语料用于事后描述；如做样本外预测，应另用早期数据拟合并冻结预处理、词典和模型。


## DeepSeek 政策情绪与两党态度

本阶段固定使用 K=6 实质政策 LDA，不重新训练模型。程序通过 `article_id` 连接 226 篇政策文章的原始正文与完整主题概率；每篇文章只调用一次 DeepSeek，并在同一请求中评价全部六个主题。当前生产结果使用 `deepseek-flash`、`policy_sentiment_v5` 提示词和 `policy_topics_k6_v1` 主题定义，调用时显式关闭思考模式并要求 JSON 输出。

```powershell
conda activate assignment5_guardian_lda
python score_policy_sentiment.py --stage dry-run
python score_policy_sentiment.py --stage trial
python score_policy_sentiment.py --stage full
python score_policy_sentiment.py --stage aggregate
python score_policy_sentiment.py --stage validate
```

密钥从 `.env` 或环境变量 `DEEPSEEK_API_KEY` 读取，不写入缓存、日志或输出。`trial` 默认分层处理 30 篇；`full` 会复用有效缓存，只请求剩余文章。可通过 `--concurrency`、`--max-articles`、`--max-requests` 和 `--max-cost-usd` 限制运行。缓存键包含 article_id、正文哈希、主题定义版本、提示词版本和模型，修改规则时必须提升版本。

输出位于 `data/policy_sentiment/`，主要包括：

- 文章级：`article_policy_sentiment.csv`、`article_policy_sentiment.jsonl`
- 试跑与人工复核：`sample_review.csv`、`human_validation_template.csv`
- 日频：`daily_policy_attention.csv`、`daily_policy_sentiment.csv`、`daily_policy_party_stance.csv`
- 周频：`weekly_policy_attention.csv`、`weekly_policy_sentiment.csv`、`weekly_policy_party_stance.csv`
- 运行审计：`api_usage_summary.json`、`api_request_ledger.jsonl`、`api_failures.jsonl`
- 质量检查：`policy_sentiment_validation_report.json`、`topic_coverage_summary.csv`

图表位于 `figures/policy_sentiment/`。日频保留 92 天，周频按纽约时间周一至周日直接汇集文章重新加权。缺失、`not_expressed`、`unclear` 和 API 失败不填零。两党指数的覆盖样本分别保存，不能直接相减解释为相对党派优势。`mixed` 文章敏感性分析入口已预留，但本轮未执行。

DeepSeek 接口实现依据官方的 [API 文档](https://api-docs.deepseek.com/)、[思考模式说明](https://api-docs.deepseek.com/guides/thinking_mode/)、[JSON 输出说明](https://api-docs.deepseek.com/guides/json_mode/)和[价格表](https://api-docs.deepseek.com/quick_start/pricing/)；未来重跑前应再次核对模型可用性和计费标准。

### 时间序列 Notebook

`policy_time_series.ipynb` 直接读取现有日频和周频输出，不重新训练 LDA，也不调用 DeepSeek。Notebook 同时绘制原始日频、过去 7 天滚动和周频版本。滚动版将当日及前 6 天文章直接按原公式重新加权，不插值、不把缺失填零，并生成 `rolling_7d_policy_attention.csv`、`rolling_7d_policy_sentiment.csv` 和 `rolling_7d_policy_party.csv`。默认绘制 `All`；可将 `SECTION_GROUP` 改为 `Opinion` 或 `Non-Opinion`，或将 `ROLLING_DAYS` 改为 14。橙色点表示窗口内有效文章少于 3 篇。

## Polymarket Democratic Sweep 相关性与样本外检验

本阶段固定使用已有 K=6 政策主题、日频议题情绪、民主党态度、共和党态度和主题关注度，不重新训练 LDA，也不重新调用 DeepSeek。目标市场经 Gamma API 核验为 `2026 Balance of Power: D Senate, D House` / `Democrats Sweep` / market id `562828`，并按 `outcomes` 与 `clobTokenIds` 的数组位置确认 Yes token。

```powershell
conda activate assignment5_guardian_lda
python fetch_polymarket_sweep.py --config sweep_analysis_config.yaml
python analyze_sweep_indices.py --config sweep_analysis_config.yaml
```

下载脚本按最多 7 天分段访问 CLOB `prices-history`，原始 JSON 保存在 `data/polymarket/raw/`，同名缓存不会覆盖。断点续跑时已存在的分段直接读取缓存；如只允许使用缓存，可加 `--offline`。每段仅保留 `startTs <= t < endTs`，因此接口附带的当前价格不会混入历史数据。

清洗后的小时价格与 92 天纽约日频概率位于 `data/polymarket/`。相关性、对齐覆盖、周频稳健性、walk-forward 预测明细、指标、验证报告和中文总结位于 `data/sweep_analysis/`，图表位于 `figures/sweep_analysis/`。日频差分先在完整自然日序列上计算，再筛选有效配对；缺失不补零、不跨日向前填充。样本外模型至少需要 30 个因果训练配对，预测的是次日市场隐含概率变化，不是交易收益。

`sweep_analysis_tables.ipynb` 是已执行的主要结果表格版 Notebook，直接读取上述最终 CSV，集中展示市场覆盖、各主题有效样本、主要同日变化相关、walk-forward 指标、同时优于两个基准的模型和验证结果。它不会重新下载数据或重新运行付费模型。

可在 VS Code/Jupyter 中打开，也可在项目 conda 环境中批量重新执行：

```powershell
conda activate assignment5_guardian_lda
python -m jupyter nbconvert --to notebook --execute --inplace policy_time_series.ipynb
```

## 美国股票与政策情绪的探索性分析

该模块使用已冻结的 K=6 政策情绪结果，不重训 LDA、不重新调用 DeepSeek。股票池在读取收益关联前由 S&P 500、Nasdaq-100 和 iShares Russell 1000 ETF（IWB）公开持仓代理整合，并固定为最多 1,000 只。公开端点只能取得 2026-10 附近的名单快照，因此输出明确保留生存者偏差限制；IWB 持仓不是授权 Russell 1000 历史成分文件。

```powershell
conda activate assignment5_guardian_lda
python fetch_equity_prices.py --stage universe
python fetch_equity_prices.py --stage prices
python analyze_equity_sentiment.py --config equity_sentiment_config.yaml
```

`yfinance` 显式使用 `auto_adjust=False`、`actions=True` 和 `end=2026-10-10`。分批结果缓存在 `data/equity_sentiment/raw_batches/`；再次运行会读取缓存并只重试缺失 ticker。主收益为复权开盘至收盘股票收益减去完全相同区间的 SPY 收益。对目标交易日，新闻信号只使用前一交易日纽约日期 00:00 至目标日 00:00（不含）的文章，并按既有 LDA 概率和标签规则从文章级重新聚合；缺失不补 0。

主要输出位于 `data/equity_sentiment/`：冻结股票池、股票—主题业务映射、原始行情、复权收益、盘前信号、完整压缩对齐表、HAC(5) 主检验、全局 BH-FDR、5 日移动区块 bootstrap、预设联合/市场/行业控制、固定主题篮子、expanding-window 预测和验证报告。图表位于 `figures/equity_sentiment/`。`stock_signal_alignment.csv.gz` 较大，使用 gzip 保存但仍可由 pandas 直接读取。

本模块是约三个月 Guardian 报道样本上的探索性研究。关注度不作情绪解释；同日 close-to-close 结果单列为描述性；相关、条件相关及预测表现均不代表因果或可实现交易收益。既有 Democratic Sweep 非显著结果未被修改。


## DeepSeek政策情绪与两党态度

本阶段固定使用K=6实质政策LDA，不重新训练模型。程序通过`article_id`连接226篇政策文章的原始正文与完整主题概率；每篇文章只调用一次DeepSeek，并在同一请求中评价全部六个主题。

```powershell
conda activate assignment5_guardian_lda
python score_policy_sentiment.py --stage dry-run
python score_policy_sentiment.py --stage trial
python score_policy_sentiment.py --stage full
python score_policy_sentiment.py --stage aggregate
python score_policy_sentiment.py --stage validate
```

密钥从`.env`或环境变量`DEEPSEEK_API_KEY`读取，不写入缓存、日志或输出。`trial`默认分层处理30篇；`full`会复用有效缓存，只请求剩余文章。可通过`--concurrency`、`--max-articles`、`--max-requests`和`--max-cost-usd`限制运行。缓存键包含article_id、正文哈希、主题定义版本、提示词版本和模型，修改规则时必须提升版本。

输出位于`data/policy_sentiment/`，图表位于`figures/policy_sentiment/`。日频保留92天，周频按纽约时间周一至周日直接汇集文章重新加权。缺失、not_expressed、unclear和API失败不填零。两党指数的覆盖样本分别保存，不能直接相减解释为相对党派优势。

## Mid-term equity baskets

`analyze_midterm_baskets.py` builds exploratory equal-weight baskets from the
existing stock-level equity-sentiment tests. It does not redownload prices,
retrain LDA, or call DeepSeek. Configuration is in
`midterm_basket_config.yaml`.

```powershell
& 'C:\Users\Jerry Cai\anaconda3\envs\assignment5_guardian_lda\python.exe' analyze_midterm_baskets.py --config midterm_basket_config.yaml
```

Outputs are in `data/equity_sentiment/midterm_basket/`. The full-sample basket
tests are explicitly post-selection because the same study window was used to
identify significant stocks. The script therefore also performs a chronological
check: constituents are selected only through the first 60% of trading days and
tested from the following trading day onward. Short holdout samples remain
exploratory. Directional baskets are statistical long/short factors and do not
include transaction costs, borrowing costs, spread, or capacity assumptions.
