# Campus Agent RAG 评测报告

- 用例数：32
- Embedding Provider：`sentence-transformers`
- Embedding Model：`BAAI/bge-small-zh-v1.5`
- Model Revision：`7999e1d3359715c523056ef9478215996d62a620`
- Model Fingerprint：`d7346f499033af67`
- Device：`cpu`
- Minimum Similarity：`0.48`
- Query Prompt：`为这个句子生成表示以用于检索相关文章：`
- Dataset Fingerprint：`4f02250011e0de943b8df7ce21c67b22096e55644619f08eff078060c9951f38`
- Corpus Fingerprint：`0682763db0eb32c4e673afeb7230842e475dcb325323364ec2d8e09c3070bc57`
- Configuration Fingerprint：`084dad13caa52b4e5105cf6216534f293a8bbf20cfd56074a5f072d0808886ce`
- Runtime Versions：`numpy=2.5.1, python=3.12.4, sentence-transformers=5.5.1, torch=2.12.0, transformers=5.10.2`
- Vector状态：`ready`
- 生成时间：2026-08-29T13:25:06+00:00
- Prompt Injection：`not_evaluated`（当前离线评测未执行真实工具路由）

## 策略对比

| 策略 | Hit@1 | Hit@3 | Recall@3 | MRR@3 | No-hit | FPR | Hard-negative安全拒答 | 失败数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 0 |
| vector | 88.00% | 92.00% | 92.00% | 90.00% | 100.00% | 0.00% | 100.00% | 2 |
| hybrid | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 0 |

## 分口径结果

全量结果同时包含 dev/test 和 retrieval/agentic；发布判断应优先查看对应口径。

| 策略 | 口径 | 用例数 | Hit@3 | No-hit | Hard-negative安全拒答 | 失败数 |
|---|---|---:|---:|---:|---:|---:|
| bm25 | split:test | 32 | 100.00% | 100.00% | 100.00% | 0 |
| bm25 | task:agentic | 3 | 100.00% | N/A | N/A | 0 |
| bm25 | task:retrieval | 29 | 100.00% | 100.00% | 100.00% | 0 |
| vector | split:test | 32 | 92.00% | 100.00% | 100.00% | 2 |
| vector | task:agentic | 3 | 100.00% | N/A | N/A | 0 |
| vector | task:retrieval | 29 | 90.91% | 100.00% | 100.00% | 2 |
| hybrid | split:test | 32 | 100.00% | 100.00% | 100.00% | 0 |
| hybrid | task:agentic | 3 | 100.00% | N/A | N/A | 0 |
| hybrid | task:retrieval | 29 | 100.00% | 100.00% | 100.00% | 0 |

## 失败用例

失败详情位于 `failures.jsonl`，每条记录保留预期Chunk、实际排序和失败原因。
