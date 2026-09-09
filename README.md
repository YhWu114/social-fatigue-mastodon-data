# Social Fatigue on Mastodon - Data Release (AN-15)

Companion data and code for the IP&M paper on cross-cultural social fatigue on Mastodon.

## Contents
| File | Description |
|---|---|
| `data/fatigue_labels_63797.csv` | Per-post labels for the 63,797 verified fatigue expressions. Columns: uri, language, group, categories (semicolon-joined 5-dim labels), and one confidence-score column per dimension (0 = absent). |
| `data/fatigue_labels_63797.jsonl` | Same records in JSONL. |
| `lexicon/fatigue_taxonomy.json` | The English fatigue keyword taxonomy (5 dimensions) used for English-post candidate retrieval. |
| `lexicon/fatigue_taxonomy_multilingual.json` | The expanded multilingual keyword taxonomy (same 5 dimensions) used for the seven non-English languages (ja, de, fr, zh, es, nl, pt), including CJK slang and European-language expressions. |
| `lexicon/_src_filter_by_keywords.py` | Source of the English taxonomy (`filter_by_keywords.py`). |
| `lexicon/_src_find_by_keywords_no_en.py` | Source of the multilingual taxonomy (`find_by_keywords_no_en.py`). |
| `prompts/qwen_final_verification_en.txt` | System prompt for the English Qwen3-8B final verification pass. |
| `prompts/qwen_final_verification_nonen.txt` | System prompt for the non-English pass. |
| `prompts/user_template.txt` | User-message template. |
| `code/` | Analysis code that produces every result in the paper. |
| `code/an10_topic_stability.py` | Ten-seed topic-proportion stability analysis (AN-10). Re-fits the production BERTopic flow on a cached 60,141x768 embedding matrix, varying only the UMAP random seed; saves per-record assignments; aligns topics across runs by Jaccard similarity of record memberships with a one-to-one assignment; computes per-language proportion fluctuations and focal contrasts. |
| `code/an10_make_deliverables.py` | Builds the focal-contrast table and the fluctuation figure from `an10_topic_stability_stats.json`. |
| `results/an10_topic_stability_stats.json` | Full AN-10 statistics: ARI/NMI per run against the seed-42 reference, mean and per-topic Jaccard match quality, flagged correspondences, per-(language x topic) fluctuation across the ten runs, and the focal contrasts. |
| `results/an10_core_contrast_table.csv` | The four focal topic contrasts with direction-retention counts and percentage-point ranges. |
| `results/an10_topic_proportion_means.csv` | Full 9 x 29 matrix of mean within-language topic proportions across the ten runs. |
| `results/an10_fluctuation_plot.png` | Fluctuation figure: per-language mean proportions with min-max ranges for the four focal topics, and per-run Jaccard match quality. |
| `results/an10_topic_stability_assignments.csv.gz` | Per-record topic assignments (60,141 records x 10 seeds = 601,410 rows). Columns: record_id, language_group, seed, topic_id. |

## Topic-proportion stability (AN-10)
The production topic model archived only aggregate language-by-topic counts, not per-record assignments, so the seed-stability experiment re-fits the pipeline. Its clusters are close in content to the archived ones but not identical in membership, and cuML UMAP/HDBSCAN is not bit-reproducible. Consequently the **absolute** within-language shares in `results/an10_*` differ from the archived production figures, while the **signed contrasts** reported in the paper reproduce (see the manuscript for the comparison). Reference topic IDs 0-3 in these outputs belong to this experiment and are not identically numbered topics in the original figures.

## Ethical note
Raw post text is intentionally NOT redistributed to respect Fediverse community norms and minimize re-identification risk. Each record carries a stable post URI; the original posts can be re-derived from these identifiers using the public FediLive archive (Min et al., 2025). No personally identifying information is included.

## License
CC-BY-4.0 (data + lexicon + prompts) and MIT (code), per Zenodo deposit.
