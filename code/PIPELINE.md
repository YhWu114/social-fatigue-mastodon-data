# Pipeline and input contract

## Screening

Use a local OpenAI-compatible Qwen3-8B endpoint. Input JSONL records contain `id`, `language` and cleaned `text`. The convenience runner accepts already-preprocessed posts; collection, spam filtering and record assembly are upstream steps, not silently reproduced by this adapter.

```bash
python code/screen.py --input /path/to/private_candidates.jsonl \
  --output /path/to/private_screening.jsonl --all-candidates \
  --endpoint http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B
```

Omit `--all-candidates` to apply the supplied English broad substring filter or the shared non-English pool (word boundaries for alphabetic expressions; substring matches for Chinese/Japanese expressions). Screening uses temperature 0, 512 output tokens and thinking disabled. Retention requires a positive decision and at least one valid fatigue type. The adapter records failed requests as unresolved, never as a negative prediction; retry handling can be scheduled separately.

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

The adapter uses the same production prompts, schema and topic-decision engine; local example-loading and serving adapters replace private infrastructure. It does not redistribute the private input data or promise an exact fresh-inference reproduction of the original run. Existing label-based statistics can be verified offline without inference.

## Repeated assignments

The ten recorded seeds are 20261017, 1, 2, 3, 7, 11, 21, 57, 99 and 123. Definitions remain fixed. The released repeat-run proportions and Jaccard files cover the common set of 715 records resolved in every run, with per-language retention supplied separately. Main-text contrasts are platform use (English minus Chinese S), entertainment and relationships (Chinese S minus English). These are repeated assignments, not refitted BERTopic clusters.

## Source review

The separate 60-post fatigue-source exercise supplies qualitative examples in the paper. It is distinct from the 60-post topic-assignment review. Neither source excerpts nor private reviewer comments are redistributed; no population source frequencies are inferred from that small sample.
