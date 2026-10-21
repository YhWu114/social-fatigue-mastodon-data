# Pipeline and input contract

## What is provided

The release supplies derived record-level data, aggregate experiment results, final figures, prompts, topic definitions, and the screening and topic-assignment code. Checking the released labels and recomputing proportions and effect sizes do not require a language model.

For new inference, researchers prepare Qwen3-8B weights and a local OpenAI-compatible service with a `/v1/chat/completions` endpoint. The service must support Qwen thinking controls, seeds, and JSON-schema responses; topic inference also sends `top_k`. Set `--model` to the name exposed by that service. Model downloads, serving software, GPU setup, and embedding generation are researcher-managed prerequisites, not included dependencies in `requirements.txt`. The runners accept only loopback endpoints to keep input text local. Record the model revision and serving version when running new experiments.

Researchers also supply cleaned text and, for topic assignment, embeddings, reviewed examples, and candidate topics in the formats below. The released label tables cannot substitute for original text in a new inference run.

## Screening

Use a local OpenAI-compatible Qwen3-8B endpoint. Input JSONL records contain `id`, `language` and cleaned `text`. The convenience runner accepts already-preprocessed posts; collection, spam filtering and record assembly are upstream steps, not silently reproduced by this adapter.

```bash
python code/screen.py --input /path/to/private_candidates.jsonl \
  --output /path/to/private_screening.jsonl --all-candidates \
  --endpoint http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B
```

Omit `--all-candidates` to apply the supplied English broad substring filter or the shared non-English pool (word boundaries for alphabetic expressions; substring matches for Chinese/Japanese expressions). Screening uses temperature 0, 512 output tokens and thinking disabled. Retention requires a positive decision and at least one valid fatigue type. The adapter records failed requests as unresolved, never as a negative prediction; retry handling can be scheduled separately.

The screening seed is 20261016 and the request timeout is 600 seconds. Generation settings and instructions are in `prompts/screening.json`; the input template is in `prompts/user_template.json`. The output path must not already exist. Each output row contains the input ID and a status (`ok`, `keyword_excluded`, or `unresolved`); successful reviews also contain `retained` and `fatigue_types`.

## Topic design

Topic discovery sampled 1,200 posts: 80 per language group and 480 from the remainder, excluding prior review text and limiting account contributions. Candidate generation used 535 clear extractions from an interim 769 processed records. Model-proposed contexts were consolidated into 18 definitions, reviewed against examples, then gaming and media consumption were merged into 17 shared topics. This was an analyst-assisted design process, not a fully automatic clustering algorithm. The final definitions are supplied in `topic_pipeline/broad_topics.json`.

## Topic assignment

Prepare a PRIVATE work directory (do not commit it):

- `posts.jsonl`: original text records with zero-based contiguous `position`, `text`, `text_key` (SHA-256 of text), and `group`.
- `embeddings.npy`: one multilingual embedding per record in the same order. The study used `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, token chunks of 120 and stride 96 with pooled chunk embeddings.
- `embedding_config.json`: `source_sha256`, the SHA-256 of the complete `posts.jsonl` file.
- `human_examples.json`: list of `position`, `topics` (IDs), `comment`, `reference_id`, and optional `exclude`. Use the selected reviewed examples, including examples with no topic. Four ambiguous development examples were excluded in the study; exclusion is an explicit input field in this adapter. Original reference text and comments are withheld.
- `prior_candidates.json`: list of `position` and `item: {"topics": [{"id": "T01"}, ...]}` for every input record. These are candidates from a prior assignment, not trusted final labels. For a new study, supply its prior candidates; empty candidates are allowed but change this input condition.

```bash
python code/topic_pipeline/run.py --work /path/to/private_work \
  --endpoint http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B
```

The retrieval stage selects up to two nonempty and one empty human reference by embedding similarity, excluding the target text. The engine retains agreement between two readings and resolves disputed/prior candidates using source-sentence evidence. Fine-content mismatches trigger another review. JSON schemas constrain output names and source IDs; semantic validation and bounded retries reject invalid responses. Output failures remain explicit. Inference caches and outputs contain private source text and must stay outside the public repository.

### Assignment parameters

| Setting | Value |
|---|---|
| Default seed | 20261017; override with `--seed` |
| Thinking stages | Topic proposal, independent reading, conflict resolution, and specific-content assignment |
| Thinking sampling | Temperature 0.6, top-p 0.95, top-k 20 |
| Non-thinking sampling | Temperature 0.7, top-p 0.8, top-k 20 |
| Output budget | 4,096 tokens initially, increasing by 1,024 per format-repair attempt, up to 8,192 within each call |
| Validation attempts | Up to five per call; some failures invoke an additional recovery stage |
| Request timeout | 600 seconds; up to two request attempts within an inference call |
| Evidence segments | At most 650 characters each; together they preserve the full input text |

Thinking is disabled after repeated format failures or a truncated response. The implemented settings are in `runtime.py` and `engine.py`. Use `--limit N` to test a small number of posts before a full run. Outputs are written to `topic_run_<seed>.jsonl`; an existing output file is not overwritten. Successful requests are cached in `inference.sqlite3`. Keep the work directory private, and use a fresh directory for a new run as required by the runner.

The adapter uses the same production prompts, schema and topic-decision engine; local example-loading and serving adapters replace private infrastructure. It does not redistribute the private input data or promise an exact fresh-inference reproduction of the original run. Existing label-based statistics can be verified offline without inference.

## Repeated assignments

The ten recorded seeds are 20261017, 1, 2, 3, 7, 11, 21, 57, 99 and 123. Definitions remain fixed. The released repeat-run proportions and Jaccard files cover the common set of 715 records resolved in every run, with per-language retention supplied separately. Main-text contrasts are platform use (English minus Chinese S), entertainment and relationships (Chinese S minus English). These are repeated assignments, not refitted BERTopic clusters.

## Source review

The separate 60-post fatigue-source exercise supplies qualitative examples in the paper. It is distinct from the 60-post topic-assignment review. Neither source excerpts nor private reviewer comments are redistributed; no population source frequencies are inferred from that small sample.

## Available results and offline checks

| Files | Use |
|---|---|
| `data/fatigue_records.csv`, `data/topic_records.csv` | Recompute type/topic proportions and association effect sizes with `code/analyze_labels.py`. |
| `data/specific_content.csv` | Verify the specific-content counts in `results/topics/specific_content_counts.csv`. |
| `data/topic_review_labels.csv` | Verify model–human and human–human topic overlap counts. |
| `results/screening_*.json`, `results/screening_human_agreement.csv` | Inspect the completed screening comparisons and workload assessment. |
| `results/dimensions/`, `results/topics/` | Inspect completed statistical tests, confidence intervals, sensitivity checks, and repeated assignments. |
| `figures/` | Use the five completed manuscript figures. |

Run `python code/verify_release.py` from the repository root to check record counts, proportions, specific-content counts, topic-review overlap, and repeat-run contrasts against the released results. Run `python code/test_pipeline.py` after installing `requirements.txt` for four synthetic offline checks of screening rules, keyword matching, evidence segments, and topic-output validation. These tests do not call a model.

The analysis script recomputes proportions and effect sizes, not the published bootstrap intervals or simulation p-values. Those results are supplied as completed outputs. Account/server restrictions need private identifiers, and new model runs need original text. Earlier screening comparisons and the analyst-assisted topic-discovery process are documented results, not separate automated runners in this release.
