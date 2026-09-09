"""
AN-10 extension: topic-proportion stability across random seeds.

Design (per annotation):
  - Reuse the 60,141 texts + 9 language groups used by the existing AN-10.
  - Replicate the PRODUCTION topic_modeling.py flow (cuML GPU UMAP/HDBSCAN,
    nr_topics=30, reduce_outliers("embeddings") + update_topics), varying ONLY
    the UMAP random_state across 10 seeds. We do NOT mix CPU/GPU implementations.
  - The LLM representation (qwen3-8B via vLLM) only renames topics; it does NOT
    affect per-doc cluster assignments. We therefore use the default c-TF-IDF
    representation so the run does not depend on a running vLLM server, while the
    *clustering* (and thus all proportions) stays identical to production.
  - Embeddings are computed ONCE and reused for every seed.
  - Save per-doc assignment: record_id | language_group | seed | topic_id.
  - Align topics across seeds via Jaccard on POST OVERLAP (linear_sum_assignment),
    NOT using language proportions; flag unreliable correspondences.
  - Recompute p_{s,l,t} and report fluctuation mean/std/min/max in percentage points.
  - Report the 3 pre-specified core contrasts (family zh>en, platform en>zh,
    daily-life ja vs nl): seeds retaining direction + range of differences.

Run under the `scft` conda env (bertopic 0.17.4 + cuml 26.02).
"""
import os, glob, json, argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import torch
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# ---- GPU-accelerated reduction / clustering (production flow) ----
from cuml.manifold import UMAP as cuUMAP
from cuml.cluster import HDBSCAN as cuHDBSCAN

# ============================================================
# Paths / config
# ============================================================
SOURCE_DIRS = [
    '/home/user/social-fatigue/results/mastodon_qwen_8b',
    '/home/user/social-fatigue/results/mastodon_qwen_8b_non_en',
]
EMBED_MODEL = '/home/user/social-fatigue/model/paraphrase-multilingual-mpnet-base-v2'
RESULT_DIR = '/home/user/social-fatigue/results'
EMB_CACHE = os.path.join(RESULT_DIR, 'an10_embeddings.npy')
ASSIGN_DIR = os.path.join(RESULT_DIR, 'an10_seed_assignments')
os.makedirs(ASSIGN_DIR, exist_ok=True)

SEEDS = [1, 2, 3, 7, 11, 21, 42, 57, 99, 123]
REF_SEED = 42
# Reference-topic (in OUR seed-42 run) -> theme.
# BERTopic renumbers clusters per run, so these IDs are derived from THIS
# run's own language profiles (not the production run's IDs): topic 1 =
# Chinese-dominant family/emotional cluster, topic 2 = English-dominant social
# media ("platform"), topic 0 = Japanese/Dutch daily-routine cluster, topic 3 =
# German/Dutch political cluster. Validated against Results.tex lines 289-291.
REF_TOPIC_THEME = {1: 'family', 2: 'platform', 0: 'daily-life', 3: 'fascist'}
# 9 language groups in the order used by the paper
LANG_GROUPS = ['en', 'zh-S', 'ja', 'de', 'zh-T', 'fr', 'es', 'pt', 'nl']
JACCARD_FLAG = 0.5  # below this best-match Jaccard -> unreliable correspondence


def to_group(lang):
    """Map raw platform language code to the paper's 9 operational groups."""
    l = str(lang).strip().lower()
    if l in ('zh-cn', 'zh'):
        return 'zh-S'
    if l in ('zh-tw', 'zh-hk'):
        return 'zh-T'
    return l


def load_data(directories):
    docs, uris, langs = [], [], []
    files = []
    for d in directories:
        files.extend(glob.glob(os.path.join(d, 'livefeeds_*.jsonl')))
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = item.get('content', '')
                if isinstance(content, str):
                    text = content.strip()
                    if len(text) > 25:
                        docs.append(text)
                        uris.append(item.get('uri') or item.get('id') or '')
                        langs.append(item.get('language', 'unknown'))
    return docs, uris, langs


def build_embeddings(docs, device):
    """Return (model, E). E is computed once and cached to disk.

    The SentenceTransformer is returned so it can be passed as BERTopic's
    embedding_model: this satisfies reduce_outliers(strategy="embeddings"),
    which reuses the *stored* embeddings_ (set from the precomputed E below)
    rather than re-embedding.
    """
    model = SentenceTransformer(EMBED_MODEL, device=device)
    if os.path.exists(EMB_CACHE):
        print(f'[embed] loading cached embeddings {EMB_CACHE}')
        E = np.load(EMB_CACHE)
        if E.shape[0] == len(docs):
            return model, E.astype(np.float32)
        print('[embed] cache size mismatch, recomputing')
    E = model.encode(docs, show_progress_bar=True, batch_size=256,
                     convert_to_numpy=True, normalize_embeddings=False)
    E = E.astype(np.float32)
    np.save(EMB_CACHE, E)
    print(f'[embed] computed {E.shape}, saved')
    return model, E


def run_seed(seed, docs, E, embed_model, vectorizer, ctfidf):
    """Run the production BERTopic flow for one UMAP seed; return final per-doc topics."""
    umap_model = cuUMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                        metric='cosine', random_state=seed)
    hdbscan_model = cuHDBSCAN(min_samples=10, min_cluster_size=20,
                              prediction_data=True, gen_min_span_tree=True)
    topic_model = BERTopic(
        # embedding_model is set so reduce_outliers(strategy="embeddings") works;
        # fit_transform receives precomputed E, so no re-embedding occurs and
        # self.embeddings_ is reused by the outlier-recovery step.
        embedding_model=embed_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        nr_topics=30,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=E)
    new_topics = topic_model.reduce_outliers(docs, topics, strategy='embeddings')
    topic_model.update_topics(docs, topics=new_topics,
                              vectorizer_model=vectorizer, ctfidf_model=ctfidf)
    return np.asarray(new_topics, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default=None,
                    help='comma list of seeds to (re)run; default all')
    ap.add_argument('--no-cluster', action='store_true',
                    help='skip clustering, only aggregate from saved assignments')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}')

    print('[load] reading source jsonl ...')
    docs, uris, raw_langs = load_data(SOURCE_DIRS)
    n = len(docs)
    print(f'[load] {n} docs')
    groups = [to_group(l) for l in raw_langs]
    # keep only the 9 known groups (safety)
    groups = [g if g in LANG_GROUPS else 'other' for g in groups]
    group_arr = np.array(groups)
    print('[load] language group counts:')
    for g in LANG_GROUPS:
        print(f'   {g}: {int((group_arr == g).sum())}')

    if not args.no_cluster:
        embed_model, E = build_embeddings(docs, device)
        # fixed vectorizer / ctfidf (identical across seeds -> only UMAP seed varies)
        custom_en_noise = ["https","http","www","com","org","net","ve","ll","re","don","just","like","rt","amp","really","got","know","think","people","time","day","today","going","im","dont"]
        multilingual_noise = ["und","der","die","das","ist","du","ich","nicht","es","in","zu","von","mit","für","le","la","les","et","de","en","un","une","est","je","il","pour","dans","sur","el","los","las","que","por","para","con","una","um","na","no","se","su","van","ik","te","dat","die","een","hij","het","voor","zijn","的","了","是","我","你","他","在","就","都","而","及","与","这","那","これ","それ","あれ","この","その","あの","ここ","そこ","私","僕","俺","する","いる","ある"]
        stop = list(ENGLISH_STOP_WORDS) + custom_en_noise + multilingual_noise
        vectorizer = CountVectorizer(stop_words=stop, min_df=5, ngram_range=(1, 2))
        ctfidf = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

        run_seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else SEEDS
        for seed in run_seeds:
            out = os.path.join(ASSIGN_DIR, f'seed_{seed}.npy')
            if os.path.exists(out):
                print(f'[seed {seed}] assignment exists, skip')
                continue
            print(f'\n===== seed {seed} =====')
            topics = run_seed(seed, docs, E, embed_model, vectorizer, ctfidf)
            np.save(out, topics)
            print(f'[seed {seed}] topics saved, n_unique={len(set(topics.tolist()))}')

    # ============================================================
    # Aggregation
    # ============================================================
    print('\n[aggregate] loading per-seed assignments ...')
    assign = {}      # seed -> int32 array
    for seed in SEEDS:
        p = os.path.join(ASSIGN_DIR, f'seed_{seed}.npy')
        if not os.path.exists(p):
            print(f'[aggregate] WARNING missing seed {seed}, skipping')
            continue
        assign[seed] = np.load(p)

    if not assign:
        print('[aggregate] no assignments found, abort')
        return

    ref = assign[REF_SEED]
    # ARI/NMI of each seed vs reference (clustering stability)
    ari_nmi = {}
    for seed, a in assign.items():
        if seed == REF_SEED:
            ari_nmi[seed] = (1.0, 1.0)
        else:
            ari_nmi[seed] = (float(adjusted_rand_score(ref, a)),
                             float(normalized_mutual_info_score(ref, a)))

    # ---- Topic alignment (Jaccard on post overlap) vs reference ----
    print('[align] computing Jaccard + linear_sum_assignment vs reference ...')
    ref_topics = sorted(set(ref.tolist()))
    n_ref = len(ref_topics)
    # boolean membership for reference
    ref_bool = np.zeros((n, n_ref), dtype=bool)
    for j, t in enumerate(ref_topics):
        ref_bool[:, j] = (ref == t)
    ref_sizes = ref_bool.sum(0)

    mapping = {}         # seed -> {ref_topic: (seed_topic, jaccard)}
    match_quality = {}   # seed -> {ref_topic: jaccard}
    for seed, a in assign.items():
        if seed == REF_SEED:
            mapping[seed] = {t: (t, 1.0) for t in ref_topics}
            match_quality[seed] = {t: 1.0 for t in ref_topics}
            continue
        seed_topics = sorted(set(a.tolist()))
        n_s = len(seed_topics)
        seed_bool = np.zeros((n, n_s), dtype=bool)
        for j, t in enumerate(seed_topics):
            seed_bool[:, j] = (a == t)
        seed_sizes = seed_bool.sum(0).astype(np.int64)
        # NOTE: matmul of two BOOL arrays yields a bool result that clips counts
        # to 0/1. Cast to int64 first so overlap counts are correct.
        inter = ref_bool.T.astype(np.int64) @ seed_bool.astype(np.int64)  # (n_ref, n_s)
        union = ref_sizes[:, None] + seed_sizes[None, :] - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            jac = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        # maximize Jaccard -> minimize -Jaccard
        ri, si = linear_sum_assignment(-jac)
        m = {}
        mq = {}
        for r, s in zip(ri, si):
            rt = ref_topics[r]
            st = seed_topics[s]
            m[rt] = (st, float(jac[r, s]))
            mq[rt] = float(jac[r, s])
        mapping[seed] = m
        match_quality[seed] = mq

    # flag unreliable reference topics (low best Jaccard in ANY seed)
    unreliable = {}
    for rt in ref_topics:
        mins = min(match_quality[s].get(rt, 0.0) for s in assign)
        if mins < JACCARD_FLAG:
            unreliable[rt] = mins

    # ---- per-language-group topic proportions p_{s,l,t} ----
    print('[props] computing p_{s,l,t} ...')
    # Initialize dict: p[seed][l][ref_topic] = proportion (%)
    p_sl_t = {seed: {} for seed in assign}
    for seed, a in assign.items():
        m = mapping[seed]
        a_df = pd.DataFrame({'g': group_arr, 't': a})
        # within each language group, proportion of posts in each *mapped* ref topic
        for l in LANG_GROUPS:
            sub = a_df[a_df['g'] == l]
            total = len(sub)
            if total == 0:
                continue
            counts = sub['t'].value_counts().to_dict()
            for rt in ref_topics:
                st = m[rt][0]
                c = counts.get(st, 0)
                p_sl_t[seed].setdefault(l, {})[rt] = 100.0 * c / total

    # fluctuation stats per (l, ref_topic)
    fluct = {}
    for l in LANG_GROUPS:
        for rt in ref_topics:
            vals = [p_sl_t[s][l][rt] for s in assign if l in p_sl_t[s] and rt in p_sl_t[s][l]]
            if vals:
                arr = np.array(vals)
                fluct[(l, rt)] = {
                    'mean': float(arr.mean()), 'std': float(arr.std(ddof=0)),
                    'min': float(arr.min()), 'max': float(arr.max()),
                    'range': float(arr.max() - arr.min()),
                }

    # full 9 x (n_ref) proportion matrix (mean across seeds), appendix material
    mean_mat = pd.DataFrame(index=LANG_GROUPS, columns=ref_topics, dtype=float)
    for l in LANG_GROUPS:
        for rt in ref_topics:
            vals = [p_sl_t[s][l][rt] for s in assign if l in p_sl_t[s] and rt in p_sl_t[s][l]]
            mean_mat.loc[l, rt] = float(np.mean(vals)) if vals else np.nan

    # ---- core contrasts (reference topics) ----
    print('[contrasts] evaluating pre-specified core contrasts ...')
    contrasts = {}
    for rt, theme in REF_TOPIC_THEME.items():
        rec = {'theme': theme, 'ref_topic': rt}
        if theme == 'family':       # zh-S > en
            pairs = [('zh-S', 'en')]
        elif theme == 'platform':   # en > zh-S
            pairs = [('en', 'zh-S')]
        elif theme == 'daily-life': # ja vs nl
            pairs = [('ja', 'nl')]
        else:                       # fascist: de vs nl (bonus)
            pairs = [('de', 'nl')]
        rec['pairs'] = []
        for a_l, b_l in pairs:
            diffs = []
            dir_retain = 0
            for s in assign:
                if a_l in p_sl_t[s] and b_l in p_sl_t[s] and rt in p_sl_t[s][a_l] and rt in p_sl_t[s][b_l]:
                    va = p_sl_t[s][a_l][rt]
                    vb = p_sl_t[s][b_l][rt]
                    diffs.append(va - vb)
                    if (va - vb) > 0:
                        dir_retain += 1
            diffs = np.array(diffs)
            expected_dir = 'a>b'  # a_l should exceed b_l
            rec['pairs'].append({
                'a': a_l, 'b': b_l, 'expected': expected_dir,
                'seeds_retaining': int(dir_retain),
                'seeds_total': int(len(diffs)),
                'mean_diff_pp': float(diffs.mean()),
                'min_diff_pp': float(diffs.min()),
                'max_diff_pp': float(diffs.max()),
            })
        contrasts[theme] = rec

    # ============================================================
    # Save outputs
    # ============================================================
    # per-doc assignment CSV (record_id | language_group | seed | topic_id)
    rows = []
    for seed in SEEDS:
        if seed not in assign:
            continue
        a = assign[seed]
        for i in range(n):
            rows.append((uris[i], group_arr[i], seed, int(a[i])))
    assign_df = pd.DataFrame(rows, columns=['record_id', 'language_group', 'seed', 'topic_id'])
    assign_csv = os.path.join(RESULT_DIR, 'an10_topic_stability_assignments.csv')
    assign_df.to_csv(assign_csv, index=False)
    print(f'[save] {assign_csv}  ({len(assign_df)} rows)')

    stats = {
        'n_docs': n,
        'seeds': SEEDS,
        'ref_seed': REF_SEED,
        'ari_nmi_vs_ref': {str(k): v for k, v in ari_nmi.items()},
        'ref_topics': ref_topics,
        'n_ref_topics': n_ref,
        'match_quality_mean': {str(s): float(np.mean(list(match_quality[s].values()))) for s in assign},
        'unreliable_ref_topics': unreliable,
        'fluctuation_pp': {f'{l}|{t}': fluct[(l, t)] for (l, t) in fluct},
        'contrasts': contrasts,
    }
    stats_json = os.path.join(RESULT_DIR, 'an10_topic_stability_stats.json')
    with open(stats_json, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f'[save] {stats_json}')

    # full mean proportion matrix
    mean_csv = os.path.join(RESULT_DIR, 'an10_topic_proportion_means.csv')
    mean_mat.to_csv(mean_csv)
    print(f'[save] {mean_csv}')

    # ---- console summary ----
    print('\n================ SUMMARY ================')
    print(f'n_docs={n}, seeds={len(assign)}, ref_seed={REF_SEED}')
    print('ARI/NMI vs ref:', {str(k): (round(v[0],3), round(v[1],3)) for k,v in ari_nmi.items()})
    print('Mean match quality (Jaccard) per seed:',
          {str(s): round(stats["match_quality_mean"][str(s)],3) for s in assign})
    if unreliable:
        print('UNRELIABLE reference topics (best Jaccard < %.2f in some seed): %s'
              % (JACCARD_FLAG, unreliable))
    else:
        print('All reference topics had reliable (>%.2f) cross-seed correspondence.' % JACCARD_FLAG)
    print('\nCore contrasts (direction retention across seeds):')
    for theme, rec in contrasts.items():
        for pr in rec['pairs']:
            print(f"  {theme:10s} ({rec['ref_topic']:2d}): {pr['a']} vs {pr['b']} -> "
                  f"retain {pr['seeds_retaining']}/{pr['seeds_total']}, "
                  f"diff mean={pr['mean_diff_pp']:+.2f}pp "
                  f"[{pr['min_diff_pp']:+.2f}, {pr['max_diff_pp']:+.2f}]")
    print('\nDone.')


if __name__ == '__main__':
    main()
