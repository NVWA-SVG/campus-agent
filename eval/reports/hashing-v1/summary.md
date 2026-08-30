# Campus Agent RAG 评测报告

- 用例数：100
- Embedding Provider：`local-hashing`
- Embedding Model：`local-hashing`
- Model Revision：`N/A`
- Model Fingerprint：`575509861ae40c1c`
- Device：`cpu`
- Minimum Similarity：`0.22`
- Query Prompt：`N/A`
- Dataset Fingerprint：`6010d197c4f7a17e8660efccf1255f578bd45da2da95f99c0bccde22a0802c06`
- Corpus Fingerprint：`0682763db0eb32c4e673afeb7230842e475dcb325323364ec2d8e09c3070bc57`
- Configuration Fingerprint：`af90f7de641db61fca853c48d25f13402f657c12b0f6a57aa74e4d9d05f2c0be`
- Runtime Versions：`numpy=2.5.1, python=3.12.4`
- Vector状态：`ready`
- 生成时间：2026-08-29T13:24:35+00:00
- Prompt Injection：`not_evaluated`（当前离线评测未执行真实工具路由）

## 策略对比

| 策略 | Hit@1 | Hit@3 | Recall@3 | MRR@3 | No-hit | FPR | Hard-negative安全拒答 | 失败数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 89.33% | 92.00% | 92.00% | 90.67% | 80.00% | 20.00% | 100.00% | 7 |
| vector | 74.67% | 80.00% | 80.00% | 77.33% | 100.00% | 0.00% | 100.00% | 16 |
| hybrid | 90.67% | 94.67% | 94.67% | 92.67% | 80.00% | 20.00% | 100.00% | 5 |

## 分口径结果

全量结果同时包含 dev/test 和 retrieval/agentic；发布判断应优先查看对应口径。

| 策略 | 口径 | 用例数 | Hit@3 | No-hit | Hard-negative安全拒答 | 失败数 |
|---|---|---:|---:|---:|---:|---:|
| bm25 | split:dev | 68 | 88.00% | 75.00% | 100.00% | 7 |
| bm25 | split:test | 32 | 100.00% | 100.00% | 100.00% | 0 |
| bm25 | task:agentic | 10 | 100.00% | 100.00% | N/A | 0 |
| bm25 | task:retrieval | 90 | 91.04% | 66.67% | 100.00% | 7 |
| vector | split:dev | 68 | 82.00% | 100.00% | 100.00% | 10 |
| vector | split:test | 32 | 76.00% | 100.00% | 100.00% | 6 |
| vector | task:agentic | 10 | 100.00% | 100.00% | N/A | 1 |
| vector | task:retrieval | 90 | 77.61% | 100.00% | 100.00% | 15 |
| hybrid | split:dev | 68 | 92.00% | 75.00% | 100.00% | 5 |
| hybrid | split:test | 32 | 100.00% | 100.00% | 100.00% | 0 |
| hybrid | task:agentic | 10 | 100.00% | 100.00% | N/A | 0 |
| hybrid | task:retrieval | 90 | 94.03% | 66.67% | 100.00% | 5 |

## 失败用例

失败详情位于 `failures.jsonl`，每条记录保留预期Chunk、实际排序和失败原因。
