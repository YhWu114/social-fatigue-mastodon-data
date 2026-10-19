# FatigueScan: fatigue in decentralized social media

Current companion data and code for language-aware fatigue screening and multilingual topic analysis on Mastodon.

## Current results

- 45,096 retained records (44,975 distinct posts), across nine platform-language groups.
- 44,092 records in topic analysis; 17 shared topics with multiple assignments per record.
- 24,019 records have multiple confirmed topics; 2,132 have no confirmed topic.
- 4,291 records have at least one unresolved topic judgment. This category overlaps the no-topic category.
- 37,341 records pass the additional own-fatigue check used for detailed content summaries. This check does not change the primary screening corpus.
- Repeated topic assignments: ten seeds on 1,343 records, with 715 resolved in every run.
- Human topic matching: 38/49 (77.6%) and 52/56 (92.9%) against the two reviewers; reviewer-to-reviewer overlap is 41/47 (87.2%).

The human topic metric requires at least one common topic. Model comparisons include records with a nonempty human topic set; the human comparison includes records with nonempty sets from both reviewers. The 60-post English/Chinese review contains 30 random and 30 targeted examples. The first review informed method development; the second reviewer assessed the same material independently. These percentages describe topic overlap, not exact-set accuracy. Fatigue screening and topic review are separate assessments.

## Contents

| Path | Contents |
|---|---|
| `data/fatigue_records.csv` | Opaque release ID, language group, month, fatigue types and topic eligibility for 45,096 records. |
| `data/topic_records.csv` | 17-topic assignments, unresolved-topic flags and diagnostic fatigue/affect categories for 44,092 records. |
| `data/specific_content.csv` | Post–topic decision states and specific-content IDs, without text or evidence excerpts. |
| `data/topic_review_labels.csv` | Label-only human/model topic comparisons, with no link to source posts. |
| `lexicon/` | Existing English and shared multilingual retrieval lists. |
| `prompts/screening.json` | Current Qwen3-8B screening prompt and decoding settings. |
| `code/topic_pipeline/` | Current topic definitions, specific-content definitions, prompts, JSON schemas and decision logic. |
| `results/` | Current dimension, topic, human-review and repeatability summaries. |
| `figures/` | Six current manuscript figures, with PDF metadata removed. |
| `release.json` | Version, counts and release scope. |

Topic IDs retain their lineage: T11 was merged into T10, so 17 topics use IDs T01–T10 and T12–T18. Records, not deduplicated texts, are the analysis units. Chinese (S) combines platform labels `zh` and `zh-CN`; Chinese (T) combines `zh-TW` and `zh-HK`. These are platform-defined groups. A record can carry several fatigue types and topics; percentages need not sum to 100%.

## Reproduce the released statistics

Python 3.10 or later is recommended. Verification uses only the standard library:

```bash
python code/verify_release.py
```

To recompute marginal proportions and association effect sizes:

```bash
pip install -r requirements.txt
python code/analyze_labels.py --output /tmp/fatiguescan-recomputed
```

Published bootstrap intervals, sparse-cell simulation tests, account/server checks, and automatic language checks are supplied as aggregate results where available. The effect-size script does not substitute asymptotic p-values for the published simulation tests. Account/server restrictions and language detection require private metadata or original text and cannot be independently rerun from this label-only export.

## Run the screening and topic methods

See [the pipeline guide](code/PIPELINE.md). Inference uses a researcher-controlled local Qwen3-8B server. Raw text, human reference examples and embeddings must be supplied locally; they are not distributed here. The public adapter preserves the current topic decision logic and generation settings, while replacing site-specific file paths and GPU orchestration with explicit local inputs. Results on different inputs or model-serving versions can differ.

The workflow uses keyword candidate retrieval, contextual screening, and an LLM-assisted shared topic set. Topic assignment combines a reading with retrieved human examples, an independent reading without examples, and evidence-grounded conflict resolution. Specific-content checks can trigger a review of the broad topic. The current topic workflow is not the older BERTopic pipeline.

## Privacy and versioning

The current release excludes raw posts, summaries, evidence quotations, post URLs/URIs, account/server identifiers, exact timestamps, original IDs, text hashes, reviewer comments, and local machine paths. Opaque IDs join only the released derived tables. This reduces direct linkage to source posts; it is not a claim of guaranteed anonymity.

Earlier commits contain superseded results and may contain post URIs. They are not the current experiment and this update does not rewrite Git history. For blinded review, use the configured anonymous repository snapshot rather than the personal GitHub URL. No institutional identity or IRB claim is asserted by this repository.

## License

Data, taxonomies and prompts retain CC-BY-4.0; analysis code retains the MIT license, as specified in `LICENSE`.
