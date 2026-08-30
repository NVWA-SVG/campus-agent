# Campus Agent RAG 评测报告

- 用例数：32
- Embedding Provider：`local-hashing`
- Embedding Model：`local-hashing`
- Model Revision：`N/A`
- Model Fingerprint：`575509861ae40c1c`
- Device：`cpu`
- Minimum Similarity：`0.22`
- Query Prompt：`N/A`
- Dataset Fingerprint：`4f02250011e0de943b8df7ce21c67b22096e55644619f08eff078060c9951f38`
- Corpus Fingerprint：`0682763db0eb32c4e673afeb7230842e475dcb325323364ec2d8e09c3070bc57`
- Configuration Fingerprint：`62fb8c21733100329f131f82ae087ed4d084b57d5e5d72c86c8dde7c553b5365`
- Runtime Versions：`numpy=2.5.1, python=3.12.4`
- Vector状态：`ready`
- 生成时间：2026-08-29T13:24:37+00:00
- Prompt Injection：`not_evaluated`（当前离线评测未执行真实工具路由）

## 策略对比

| 策略 | Hit@1 | Hit@3 | Recall@3 | MRR@3 | No-hit | FPR | Hard-negative安全拒答 | 失败数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 0 |
| vector | 72.00% | 76.00% | 76.00% | 74.00% | 100.00% | 0.00% | 100.00% | 6 |
| hybrid | 96.00% | 100.00% | 100.00% | 98.00% | 100.00% | 0.00% | 100.00% | 0 |

## 分口径结果

全量结果同时包含 dev/test 和 retrieval/agentic；发布判断应优先查看对应口径。

| 策略 | 口径 | 用例数 | Hit@3 | No-hit | Hard-negative安全拒答 | 失败数 |
|---|---|---:|---:|---:|---:|---:|
| bm25 | split:test | 32 | 100.00% | 100.00% | 100.00% | 0 |
| bm25 | task:agentic | 3 | 100.00% | N/A | N/A | 0 |
| bm25 | task:retrieval | 29 | 100.00% | 100.00% | 100.00% | 0 |
| vector | split:test | 32 | 76.00% | 100.00% | 100.00% | 6 |
| vector | task:agentic | 3 | 100.00% | N/A | N/A | 0 |
| vector | task:retrieval | 29 | 72.73% | 100.00% | 100.00% | 6 |
| hybrid | split:test | 32 | 100.00% | 100.00% | 100.00% | 0 |
| hybrid | task:agentic | 3 | 100.00% | N/A | N/A | 0 |
| hybrid | task:retrieval | 29 | 100.00% | 100.00% | 100.00% | 0 |

## 失败用例

失败详情位于 `failures.jsonl`，每条记录保留预期Chunk、实际排序和失败原因。
