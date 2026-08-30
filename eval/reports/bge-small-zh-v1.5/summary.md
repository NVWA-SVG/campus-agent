# Campus Agent RAG 评测报告

- 用例数：100
- Embedding Provider：`sentence-transformers`
- Embedding Model：`BAAI/bge-small-zh-v1.5`
- Model Revision：`7999e1d3359715c523056ef9478215996d62a620`
- Model Fingerprint：`d7346f499033af67`
- Device：`cpu`
- Minimum Similarity：`0.48`
- Query Prompt：`为这个句子生成表示以用于检索相关文章：`
- Dataset Fingerprint：`6010d197c4f7a17e8660efccf1255f578bd45da2da95f99c0bccde22a0802c06`
- Corpus Fingerprint：`0682763db0eb32c4e673afeb7230842e475dcb325323364ec2d8e09c3070bc57`
- Configuration Fingerprint：`f9fc6c87a95cdde4c9a4369227f2841bc1ed80163b29bc7584b1d6286d3a385e`
- Runtime Versions：`numpy=2.5.1, python=3.12.4, sentence-transformers=5.5.1, torch=2.12.0, transformers=5.10.2`
- Vector状态：`ready`
- 生成时间：2026-08-29T13:24:58+00:00
- Prompt Injection：`not_evaluated`（当前离线评测未执行真实工具路由）

## 策略对比

| 策略 | Hit@1 | Hit@3 | Recall@3 | MRR@3 | No-hit | FPR | Hard-negative安全拒答 | 失败数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 89.33% | 92.00% | 92.00% | 90.67% | 80.00% | 20.00% | 100.00% | 7 |
| vector | 89.33% | 93.33% | 93.33% | 91.33% | 80.00% | 20.00% | 100.00% | 6 |
| hybrid | 94.67% | 98.67% | 98.67% | 96.67% | 80.00% | 20.00% | 100.00% | 2 |

## 分口径结果

全量结果同时包含 dev/test 和 retrieval/agentic；发布判断应优先查看对应口径。

| 策略 | 口径 | 用例数 | Hit@3 | No-hit | Hard-negative安全拒答 | 失败数 |
|---|---|---:|---:|---:|---:|---:|
| bm25 | split:dev | 68 | 88.00% | 75.00% | 100.00% | 7 |
| bm25 | split:test | 32 | 100.00% | 100.00% | 100.00% | 0 |
| bm25 | task:agentic | 10 | 100.00% | 100.00% | N/A | 0 |
| bm25 | task:retrieval | 90 | 91.04% | 66.67% | 100.00% | 7 |
| vector | split:dev | 68 | 94.00% | 75.00% | 100.00% | 4 |
| vector | split:test | 32 | 92.00% | 100.00% | 100.00% | 2 |
| vector | task:agentic | 10 | 100.00% | 100.00% | N/A | 0 |
| vector | task:retrieval | 90 | 92.54% | 66.67% | 100.00% | 6 |
| hybrid | split:dev | 68 | 98.00% | 75.00% | 100.00% | 2 |
| hybrid | split:test | 32 | 100.00% | 100.00% | 100.00% | 0 |
| hybrid | task:agentic | 10 | 100.00% | 100.00% | N/A | 0 |
| hybrid | task:retrieval | 90 | 98.51% | 66.67% | 100.00% | 2 |

## 失败用例

失败详情位于 `failures.jsonl`，每条记录保留预期Chunk、实际排序和失败原因。
