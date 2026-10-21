# FatigueScan

Code and derived data for detecting and analyzing social media fatigue in multilingual posts on decentralized social media.

## Contents

The release contains 45,096 retained records (44,975 distinct posts) across nine language groups. Topic analysis covers 44,092 records and 17 shared topics. A record can have multiple fatigue types and topics.

| Directory | Contents |
|---|---|
| `data/` | Fatigue and topic assignments, specific-content categories, and human topic judgments. |
| `code/` | Screening, topic assignment, statistical analysis, and verification scripts. |
| `lexicon/` | Multilingual retrieval keywords and fatigue definitions. |
| `prompts/` | Screening and fatigue-source instructions. |
| `results/` | Screening, human-validation, language-comparison, and topic-stability results. |
| `figures/` | The five manuscript figures. |

Chinese (S) combines platform labels `zh` and `zh-CN`; Chinese (T) combines `zh-TW` and `zh-HK`. Topic IDs are T01–T10 and T12–T18 because two candidate topics were merged. Counts use records as the analysis unit.

## Usage

Use Python 3.10 or later. Verify the released data without installing dependencies:

```bash
python code/verify_release.py
```

Recompute proportions and association effect sizes:

```bash
pip install -r requirements.txt
python code/analyze_labels.py --output /tmp/fatiguescan-results
```

See [the pipeline guide](code/PIPELINE.md) to run screening and topic assignment with locally supplied text and a local Qwen3-8B endpoint. Raw text and human reference examples are not distributed. Account/server checks require private metadata; their aggregate results are provided.

## Privacy

The released tables contain derived labels and opaque record IDs, not raw posts, quotations, post URLs, usernames, account/server identifiers, exact timestamps, or text hashes. Human-review records are not linked to source posts. PDF metadata is removed. For anonymous review, use the anonymous snapshot, not the personal GitHub page or its commit history. Earlier commits are not covered by this cleaned snapshot.

## License

Data, taxonomies, and prompts: CC-BY-4.0. Code: MIT. See `LICENSE`.
