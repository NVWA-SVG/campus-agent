# Campus Agent：P5–P7 实现、运行与迭代说明

这轮迭代解决三个问题：怎样证明 RAG 不是只在几条示例上有效；怎样把演示性质的 Hash 向量换成真实中文语义模型；怎样在没有学校生产凭据时，先把只读业务 API 的工程边界做正确。

## 1. 完整数据流

```text
浏览器问题
  → FastAPI（校验输入、CSRF、session）
  → Rule / DeepSeek Planner
  → LangGraph 条件路由
       ├─ 课程事实 → query_courses
       ├─ 静态流程 → search_campus_knowledge
       │                → metadata pre-filter
       │                → BM25 + BGE Vector
       │                → Weighted RRF
       │                → Grade / 最多一次 Rewrite
       │                → 缺失精确事实检查
       │                → Citation + Verify
       └─ 实时状态 → query_campus_service_status
                        → TTL/Stale/Singleflight Cache
                        → Mock 或 Official Gateway
  → Compose
  → 强制保留 [模拟数据] / [官方业务API] / [缓存旧数据] 标签
  → SQLite commit
  → POST SSE 返回 Trace、回答和 Citation
```

“校园卡丢了怎么办，服务中心现在开门吗？”会被拆为两个工具调用：RAG 提供相对稳定的补办流程，业务 API 提供会变化的开放状态。两类数据不应塞进同一个向量库：流程适合检索，实时状态必须在回答时查询。

## 2. P5：100 条固定评测集

### 数据组成

| 文件 | 数量 | 目的 |
|---|---:|---|
| `retrieval_positive.v1.jsonl` | 60 | 同义改写、多事实、不同校园业务正例 |
| `retrieval_negative.v1.jsonl` | 20 | 完全无关问题与缺失金额/时间/电话等硬负例 |
| `metadata.v1.jsonl` | 10 | domain/category/version/origin 过滤边界 |
| `agentic.v1.jsonl` | 10 | Grade、Rewrite、Verify、重试上限和注入隔离 |

总计 100 条，其中 dev 68 条、冻结 test 32 条；75 条可回答、5 条严格 no-hit、20 条主题相关但证据缺失。数据由 JSON Schema 与代码双重校验，重复 ID、未知 Chunk、未知字段和错误分布都会让校验命令失败。这些用例是合成与策划数据，不是真实校园生产流量；对外发布前仍应由项目负责人逐条复核。

### 为什么把两类负例分开

- 严格 no-hit：例如“火星基地访客证”，检索器应该返回空；对应 `no_hit_accuracy` 与 `false_positive_rate`。
- 缺失事实硬负例：例如“校园卡补办多少钱”，可以召回补办流程，但资料没有金额；它不应算检索误命中，回答器必须明确拒答。对应 `hard_negative_safe_abstention_rate`。

把两类问题混成一个 no-hit 指标，会错误惩罚“正确找到主题资料”的检索器，也无法检查最终回答有没有编造数字。

### 运行

```powershell
python -m scripts.validate_eval_dataset
python -m scripts.evaluate_rag --output-dir eval\reports\hashing-v1
```

每次会生成：

- `summary.json`：配置指纹、总体指标和分组指标；
- `summary.md`：面试和 Code Review 可直接阅读的表格；
- `case_results.jsonl`：每个策略、每条用例的排序和判定；
- `failures.jsonl`：只保留失败案例，供 Error Analysis。

指标包括 Hit@k、真正以相关 Chunk 数为分母的 Recall@k、MRR、No-hit/FPR、必要事实覆盖、metadata 泄漏、Agent Trace 一致性、注入 containment、延迟分位数。阈值只在 dev 上校准，冻结 test 不参与调参。

## 3. P6：真实中文 Embedding

### 模型与可复现性

- 模型：`BAAI/bge-small-zh-v1.5`；
- revision：`7999e1d3359715c523056ef9478215996d62a620`；
- 维度：512；
- 文档使用 `encode_document`，问题使用 `encode_query`；
- 向量统一归一化，使用 NumPy float32 矩阵检索；
- `trust_remote_code=False`；
- Web/CLI 运行期强制 local-only。

Sentence Transformers 官方建议非对称语义搜索区分 Query 与 Document 编码；BGE 模型卡也建议为短查询使用检索指令，并根据自己的数据校准阈值。本项目据此设置 Query Prompt，并在 dev 集选择 0.48，而不是凭感觉填写阈值。

参考：

- [Sentence Transformers Semantic Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [BAAI/bge-small-zh-v1.5 Model Card](https://huggingface.co/BAAI/bge-small-zh-v1.5)

### 一次性准备与离线运行

```powershell
python -m pip install -e ".[dev,semantic]"
python -m scripts.prepare_embedding_model --allow-download --model-cache-dir .campus_agent_data\models

$env:CAMPUS_EMBEDDING_PROVIDER="sentence-transformers"
$env:CAMPUS_EMBEDDING_MODEL_CACHE_DIR=".campus_agent_data\models"
$env:CAMPUS_EMBEDDING_LOCAL_ONLY="true"
$env:CAMPUS_EMBEDDING_MINIMUM_SIMILARITY="0.48"
$env:CAMPUS_EMBEDDING_QUERY_PROMPT="为这个句子生成表示以用于检索相关文章："
python -m campus_agent.web
```

模型配置和 Query Prompt 进入 fingerprint。向量缓存使用不允许 pickle 的 NPZ，附带内容 SHA；只编码新增/变化的 Chunk，写入采用临时文件后原子替换，缓存损坏时自动重建。

### 实测结果

在本仓库 100 条固定集合上，真实 BGE + Hybrid 的结果为：

| 指标 | 结果 |
|---|---:|
| Hit@1 | 94.67% |
| Hit@3 / Recall@3 | 98.67% |
| MRR@3 | 96.67% |
| 严格 No-hit | 80.00% |
| 严格 FPR | 20.00% |
| 缺失事实安全拒答 | 100.00% |
| Hybrid P50 / P95 | 7.15 ms / 14.78 ms |

完整报告位于 `eval/reports/bge-small-zh-v1.5/`。No-hit 只有 5 条，所以 80% 表示有 1 条误召回；样本仍小，不能把这些数字宣传成生产 SLA。两个 Hybrid 失败都在 dev：错误 category 下仍召回图书馆片段，以及“在线续借两个条件”漏召回。32 条冻结 test 全部通过，但其中严格 no-hit 只有 1 条，因此 100% 只表示本次小 test 没有回归。Prompt Injection 的权限边界目前由独立 Graph 自动化测试验证，离线 Runner 不是完整生产 Graph，因此不把组件模拟结果列成端到端 containment 指标。

dev 阈值网格的产物位于 `eval/calibration/bge-small-zh-v1.5-dev.json`。脚本推荐 0.48，但 Hybrid dev No-hit 只有 75%，未达到预设 90% 生产安全门槛，产物明确标记 `production_gate_passed=false`。这意味着下一步应扩充 OOD/no-hit 并增加意图门控或 reranker，而不是把阈值 0.48 描述为已经通过生产验收。

## 4. P7：只读校园业务 API

### 为什么现在不能假装接通“真实学校接口”

华南农业大学公开的数据申请说明表明，业务系统数据访问需要申请、审核和授权，数据平台可按需求创建 RESTful API。因此正确顺序是：先完成契约、Mock、适配器与安全测试，再在拿到学校分配的地址和凭据后联调；不应抓网页或复用浏览器 Cookie 冒充正式接口。

参考：[华南农业大学信息网络中心—数据申请](https://inc.scau.edu.cn/sjsq/list.psp)

### Gateway 边界

```text
CampusServiceStatusTool
  → CachedCampusBusinessGateway
       ├─ MockCampusBusinessGateway
       └─ OfficialHttpCampusBusinessGateway
```

Official 模式的约束：

- 只允许固定路径的 HTTPS GET，不接受模型生成 URL；
- host 必须位于部署者配置的精确 allowlist，不允许 localhost、`.local` 或私网/保留 IP 字面量；生产部署还应在出口防火墙或代理层限制目标网段，防御 DNS rebinding；
- `service_code` 与 `campus` 是固定枚举；
- Bearer Token 只由服务端环境变量读取；
- 禁止重定向、代理环境变量和远程代码；
- 连接/读取/总预算超时，瞬态错误最多重试一次；
- 限制 Content-Type、Content-Length、Content-Encoding、解码后大小和 JSON Schema；
- TTL 缓存、stale-if-error 与 singleflight 防止重复打上游；
- Official 失败绝不自动切 Mock；
- 健康接口只返回模式和可用性，不返回 Token 或上游地址。

HTTPX 的 connect/read/write/pool timeout 语义并不等价于整个业务调用的总预算，因此适配器还单独维护 monotonic deadline。参考：[HTTPX Timeouts](https://www.python-httpx.org/advanced/timeouts/)

### Official 配置

```powershell
$env:CAMPUS_BUSINESS_API_MODE="official"
$env:CAMPUS_BUSINESS_API_BASE_URL="https://api.example.edu.cn"
$env:CAMPUS_BUSINESS_API_ALLOWED_HOSTS="api.example.edu.cn"
$env:CAMPUS_BUSINESS_API_TOKEN="服务端只读Token"
python -m campus_agent.web
```

上游契约：

```http
GET /v1/campus/service-status?service_code=campus_card&campus=main
Authorization: Bearer <server-side-token>
Accept: application/json
Accept-Encoding: identity
```

返回 JSON 必须包含与请求一致的 `service_code`、`campus`，以及受控的 `status/service_name/location/today_hours/updated_at`；队列人数与预计等待分钟数可选。

## 5. 当前到底会访问哪些网页或 API

正常聊天没有 Web Search，也不会打开或抓取网页。

- Rule + Mock + Hashing：完全本地；
- BGE 运行：完全本地，只读预先准备的模型；
- 模型准备命令：经用户显式确认后访问 Hugging Face；
- DeepSeek：只在页面选择 DeepSeek 且后端 Key 有效时访问官方 API；
- Official 校园 API：只在后端显式配置授权地址、允许主机和 Token 后访问固定只读接口。

## 6. 演示顺序

1. 启动默认 Mock Web，问“校园卡丢了怎么办，服务中心现在开门吗？”，展示一次请求内 RAG + API 两个工具。
2. 指出流程答案带 Citation，实时状态强制带 `[模拟数据]`。
3. 问“校园卡补办多少钱？”，展示有主题证据但缺少金额时安全拒答。
4. 切真实 BGE，问同义改写问题，查看 `/api/health` 中匿名化的模型 fingerprint 与缓存统计。
5. 运行 100 条评测，打开 `failures.jsonl` 解释一条误召回及下一步方案。
6. 展示 Official 配置代码和测试，但明确当前未持有学校生产凭据。

## 7. 下一步迭代

1. 将严格 no-hit 从 5 条扩到至少 50 条，再决定是提高阈值、加 reranker，还是增加 OOD/意图分类门控。
2. 收集真实同学问法并人工去标识化，形成独立 challenge set；继续冻结 test，禁止边看失败边调 test。
3. 增加冲突版本、过期通知和访问权限样本，评测 Citation 是否指向正确版本。
4. 获得校方沙箱授权后做契约测试、Token 轮换、数据新鲜度和熔断告警。
5. 只读链路稳定后再讨论写工具；任何预约、取消或提交动作都必须有人类确认、幂等键与审计日志。
