# RAGAS 评测运行指南

## 这次重构评的是什么

保留 `eval/datasets/*.jsonl` 中已经人工校对的 100 条用例（dev 68、test 32），不把旧测试删掉。新入口 `scripts/evaluate_ragas.py` 在同一批问题上调用项目检索器，再把实际返回的 Chunk ID 和人工标注的相关 Chunk ID 交给 RAGAS 0.4.3 的 `IDBasedContextPrecision`、`IDBasedContextRecall`。因此这里的 Precision/Recall 是**Chunk ID 级别**，不是 LLM 对文本相关性的主观打分。

RAGAS 对“没有召回任何 Chunk”的 Precision 返回 NaN，报告保存为 `null` 并统计 `context_precision_na`；Recall 仍为 0。不可回答题的相关 Chunk 集合本来为空，不能把其 Recall 当作 1 或 0：`retrieval_no_hit` 单独统计误报率，`missing_fact` 单独统计安全拒答。旧评测中的过滤边界、必需事实、禁用事实和 Agentic trace 检查仍运行，汇总为 `legacy_failures`。

可选的生成评测使用 RAGAS `Faithfulness` 和 `AnswerRelevancy`。它评价的是项目 `LocalRAG.answer()` **规则回答基线**，并把该次回答实际使用的检索正文传给评委；不是 LangGraph/DeepSeek Agent 的最终回答，不能据此宣称完整 Agent 已通过生成评测。默认不调用模型，报告里的 `generation.status` 是 `not_run`，不是通过或 0 分。

## 安装与离线运行（PowerShell，在项目根目录）

```powershell
python -m venv .campus_agent_data\ragas-venv --system-site-packages
.\.campus_agent_data\ragas-venv\Scripts\python.exe -m pip install -e ".[ragas-eval,dev]"
$env:RAGAS_DO_NOT_TRACK = "true"
.\.campus_agent_data\ragas-venv\Scripts\python.exe -m scripts.evaluate_ragas --split test --strategies hybrid
```

`ragas-eval` 固定 `ragas==0.4.3`、`langchain-community==0.3.31`，因为 RAGAS 0.4.3 与 `langchain-community` 0.4.x 存在导入兼容问题。虚拟环境位于已忽略的项目数据目录，不会上传到 GitHub。命令只评估检索，不需要 API Key；生成 `eval/reports/ragas-test.json`。`--split dev` 可评估开发集；`--strategies bm25 vector hybrid` 可同场对比。报告包含逐题 ID、得分、旧规则失败原因、语料和数据集指纹，方便定位回归。

## “通过”的定义

**跑完测试不等于质量通过。**不指定阈值时，`gate.status=not_configured`。团队先在 dev 集确定并冻结阈值，再用未参与调参的 test 集做验收。下面仅是命令格式示例，不是凭当前结果设定的正式上线标准：

```powershell
.\.campus_agent_data\ragas-venv\Scripts\python.exe -m scripts.evaluate_ragas --split test --strategies hybrid --min-context-precision 0.50 --min-context-recall 0.90 --max-no-hit-false-positive-rate 0
```

每个策略的平均 Precision、平均 Recall 和无命中误报率都必须过对应阈值；同时 `legacy_failures=0` 且 `missing_fact_safe_abstentions=missing_fact_cases`。任一不满足就输出 `gate.status=failed` 并以非 0 退出，适合接 CI。没有检索到任何 Chunk 的可回答题虽然 Precision 不可定义，但 Recall 为 0，旧规则也会失败。样本数、N/A 数必须随分数一起看：例如 test 集只有 1 道 `no_hit`，0% 误报并不能证明在真实分布中误报率也是 0%。

本次在默认本地 Hashing Embedding 下，hybrid 的观测值：test 集 32 题，25 道可回答题的 Precision 约 0.547、Recall 1.0，1 道 `no_hit` 没误报，6 道 `missing_fact` 均安全拒答，旧规则失败 0；dev 集 68 题，Precision 约 0.563（48/50 题可计算）、Recall 0.92，4 道 `no_hit` 中有 1 道误报，旧规则失败 5。它们是本地基线，不是上线结论。

## 可选：模型评委

先按照项目的 Embedding 准备流程在本机缓存 BGE 中文模型，并配置 `CAMPUS_EMBEDDING_PROVIDER=sentence-transformers`。然后在当前 PowerShell 会话设置 `DEEPSEEK_API_KEY`（不要写入文件或提交 Git），显式运行：

```powershell
.\.campus_agent_data\ragas-venv\Scripts\python.exe -m scripts.evaluate_ragas --split dev --strategies hybrid --judge --max-judge-cases 5
```

这一步会把问题、规则回答和召回正文发给 DeepSeek 作评委，产生 API 费用；`AnswerRelevancy` 使用本地语义 Embedding。可以加 `--min-faithfulness` 与 `--min-answer-relevancy` 设置 0–1 的门槛（必须同时带 `--judge`）。报告记录实际评测条数与模型名。未配置 Key、模型未缓存或评委调用失败时，不得把生成项写为通过；本项目当前未执行在线评委，因此没有真实 Faithfulness/Answer Relevancy 分数。

后续要评价完整 LangGraph Agent，应另行采集它每题的真实最终回答与**同一轮**检索正文，再接这两个生成指标；不得拿 `_verification_inputs` 中的合成答案或 `LocalRAG.answer()` 的规则答案冒充 Agent 输出。

参考：[RAGAS Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)、[Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)、[Faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/)、[Answer Relevancy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/)；[0.4.3 导入兼容问题](https://github.com/vibrantlabsai/ragas/issues/2753)。
