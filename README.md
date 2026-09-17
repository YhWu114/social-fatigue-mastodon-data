# Social Fatigue on Mastodon: Data and Code Release

This repository provides derived data, taxonomies, prompts, and analysis code for reproducible research on fatigue-related expressions in Mastodon posts.

## Contents

| File or directory | Description |
|---|---|
| `data/fatigue_labels_63797.csv` | Per-post derived labels for 63,797 verified fatigue expressions, including URI, language group, categories, and confidence scores. |
| `data/fatigue_labels_63797.jsonl` | The same records in JSONL format. |
| `lexicon/` | English and multilingual keyword taxonomies, together with their source scripts. |
| `prompts/` | Prompts and templates used for the verification workflow. |
| `code/` | Analysis and reproducibility scripts. |
| `results/` | Derived analysis outputs and figures. |

## Ethical note

Raw post text is not redistributed. The repository contains derived labels and metadata only, to reduce re-identification risk and respect Fediverse community norms.

## License

Data, taxonomies, and prompts are released under CC-BY-4.0. The analysis code is released under the MIT License.
