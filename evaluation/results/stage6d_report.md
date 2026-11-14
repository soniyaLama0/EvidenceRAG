# Stage 6D - Retrieval evaluation

- Generated: `2026-11-14T16:53:10.570131+00:00`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Metric: cosine similarity (Chroma 'cosine' space)
- k values: [3, 5]
- Relevance rule: chunk overlaps >= 50% of itself OR >= 50% of a reviewed gold passage, on the same page, in cleaned-text coordinates

## Collection counts

| collection | before | after |
|---|---:|---:|
| `evidence_chunks` | 39 | 39 |
| `evidence_chunks_v2` | 35 | 35 |

## Macro averages over the six questions

P = precision@k, R = recall@k, MRR = mean reciprocal rank, PR = passage recall@k (0.7 coverage).

| config | k | P | R | MRR | PR |
|---|---:|---:|---:|---:|---:|
| `v1_unfiltered` | 3 | 0.28 | 0.22 | 0.31 | 0.30 |
| `v1_unfiltered` | 5 | 0.23 | 0.36 | 0.39 | 0.45 |
| `v2_unfiltered` | 3 | 0.39 | 0.36 | 0.53 | 0.45 |
| `v2_unfiltered` | 5 | 0.30 | 0.49 | 0.56 | 0.58 |
| `v2_body_only` | 3 | 0.44 | 0.40 | 0.53 | 0.45 |
| `v2_body_only` | 5 | 0.30 | 0.49 | 0.56 | 0.58 |
| `v2_references_only` | 3 | 0.00 | 0.00 | 0.00 | 0.00 |
| `v2_references_only` | 5 | 0.00 | 0.00 | 0.00 | 0.00 |

## Per question (k = 5)

| question | config | P | R | RR | PR | gold chunks | hits |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q1 | `v1_unfiltered` | 0.20 | 0.50 | 0.25 | 0.67 | 2 | 1 |
| q1 | `v2_unfiltered` | 0.20 | 0.50 | 0.33 | 0.67 | 2 | 1 |
| q2 | `v1_unfiltered` | 0.20 | 0.33 | 0.25 | 0.25 | 3 | 1 |
| q2 | `v2_unfiltered` | 0.20 | 0.33 | 0.33 | 0.25 | 3 | 1 |
| q3 | `v1_unfiltered` | 0.00 | 0.00 | 0.00 | 0.00 | 2 | 0 |
| q3 | `v2_unfiltered` | 0.20 | 0.50 | 0.20 | 0.75 | 2 | 1 |
| q4 | `v1_unfiltered` | 0.40 | 0.50 | 0.50 | 0.50 | 4 | 2 |
| q4 | `v2_unfiltered` | 0.60 | 0.75 | 1.00 | 0.50 | 4 | 3 |
| q5 | `v1_unfiltered` | 0.20 | 0.33 | 0.33 | 0.50 | 3 | 1 |
| q5 | `v2_unfiltered` | 0.20 | 0.33 | 0.50 | 0.50 | 3 | 1 |
| q6 | `v1_unfiltered` | 0.40 | 0.50 | 1.00 | 0.80 | 4 | 2 |
| q6 | `v2_unfiltered` | 0.40 | 0.50 | 1.00 | 0.80 | 4 | 2 |

## Error-analysis diagnostics (k = 5)

| config | slots | furniture | references | truncated | boundary |
|---|---:|---:|---:|---:|---:|
| `v1_unfiltered` | 30 | 10 | 1 | 1 | 0 |
| `v2_unfiltered` | 30 | 0 | 2 | 1 | 0 |
| `v2_body_only` | 30 | 0 | 0 | 0 | 0 |
| `v2_references_only` | 30 | 0 | 30 | 24 | 0 |

## Sufficiency

| question | label | note |
|---|---|---|
| q1 | sufficient | Abstract and introduction both define the system: a voice-first, offline Android application for Hinglish business commands. |
| q2 | sufficient | Page 1 states the MSME digitisation gap (63M businesses, >95% undigitised, low digital literacy, connectivity) and page 2 states the integration gap explicitly. |
| q3 | sufficient | Table III on page 4 lists the full stack; page 1 names the same components in prose. |
| q4 | partial | Only feature validation and latency claims, no user study and no quantitative accuracy results. The authors acknowledge the gap. |
| q5 | partial | Findings are limited to 'all twelve features implemented', sub-500 ms STT and sub-100 ms NLP; no measured business outcome. |
| q6 | sufficient | Section VII lists five named limitations: NLP generalisation, scalability, language coverage, GST compliance, formal evaluation. |

## Files

- JSON: `C:\Users\Soniya\OneDrive\文件\Desktop\EvidenceRAG\evaluation\results\stage6d_results.json`
- CSV: `C:\Users\Soniya\OneDrive\文件\Desktop\EvidenceRAG\evaluation\results\stage6d_retrieval.csv`
