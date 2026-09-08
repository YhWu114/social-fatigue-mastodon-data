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

## Ethical note
Raw post text is intentionally NOT redistributed to respect Fediverse community norms and minimize re-identification risk. Each record carries a stable post URI; the original posts can be re-derived from these identifiers using the public FediLive archive (Min et al., 2025). No personally identifying information is included.

## License
CC-BY-4.0 (data + lexicon + prompts) and MIT (code), per Zenodo deposit.
