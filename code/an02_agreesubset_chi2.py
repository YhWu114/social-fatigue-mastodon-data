#!/usr/bin/env python3
"""
AN-02 deliverable: recompute the per-dimension chi-square / Cramer's V on the
Qwen∩DeepSeek AGREE-SUBSET, and report it alongside the full-sample estimates.
Reads results/an02_deepseek_labels.jsonl (produced by an02_deepseek_label.py).
"""
import json, collections, math
import scipy.stats as stats

DIMS = ["physical_fatigue","mental_cognitive_fatigue","emotional_fatigue",
        "digital_social_fatigue","existential_resignation"]
DIM_LABEL = {"physical_fatigue":"Physical","mental_cognitive_fatigue":"Mental/Cog",
             "emotional_fatigue":"Emotional","digital_social_fatigue":"Digital/Soc",
             "existential_resignation":"Existential"}

# language -> paper group (match tab:lang_fatigue / figure caption)
def to_group(lang):
    if lang in ("zh-CN","zh"): return "zh-S"
    if lang in ("zh-TW","zh-HK"): return "zh-T"
    if lang in ("es","fr"): return "es/fr"
    return lang   # en, ja, de, nl, pt

GROUPS_ALL = ["en","zh-S","ja","de","zh-T","es/fr","nl","pt"]
GROUPS_MAJOR = ["en","zh-S","ja","de","zh-T","es/fr"]   # 6 major groups (fig caption)

def load():
    recs=[]
    for line in open("results/an02_deepseek_labels.jsonl"):
        line=line.strip()
        if not line: continue
        d=json.loads(line)
        if d.get("deepseek_categories") is None:   # failed call
            continue
        recs.append(d)
    return recs

def cramers_v(chi2, n, rows, cols):
    return math.sqrt(chi2/(n*(min(rows,cols)-1))) if n>0 else None

def per_dim_v(records, use_agree_subset, groups):
    """Build per-dimension group x 2 contingency and return (chi2, V, n, per_group_yes, per_group_total)."""
    out={}
    for dim in DIMS:
        # group -> [yes, no]
        mat=collections.defaultdict(lambda:[0,0])
        for r in records:
            q=set(r["qwen_categories"]); ds=set(r["deepseek_categories"])
            qy = dim in q; dy = dim in ds
            if use_agree_subset and (qy != dy):
                continue   # exclude disagreement
            g=to_group(r.get("lang"))
            if g not in groups: continue
            if qy: mat[g][0]+=1
            else:  mat[g][1]+=1
        # contingency table rows=groups, cols=[yes,no]
        rows=[groups]
        yes=[mat[g][0] for g in groups]
        no =[mat[g][1] for g in groups]
        table=[yes,no]
        n=sum(yes)+sum(no)
        if n==0 or len(groups)<2:
            out[dim]=(None,None,n,yes,[sum(x) for x in table])
            continue
        chi2,p,dof,_=stats.chi2_contingency(table, correction=False)
        v=cramers_v(chi2,n,len(groups),2)
        out[dim]=(round(chi2,1),round(v,3),n,yes,[sum(x) for x in table])
    return out

def main():
    recs=load()
    print(f"records with both labels: {len(recs)}\n")
    full_all   = per_dim_v(recs, False, GROUPS_ALL)
    full_major = per_dim_v(recs, False, GROUPS_MAJOR)
    agree_all   = per_dim_v(recs, True, GROUPS_ALL)
    agree_major = per_dim_v(recs, True, GROUPS_MAJOR)

    # agreement / subset size per dim
    agree_n=collections.defaultdict(int); total_n=collections.defaultdict(int)
    for r in recs:
        for dim in DIMS:
            q=dim in set(r["qwen_categories"]); ds=dim in set(r["deepseek_categories"])
            total_n[dim]+=1
            if q==ds: agree_n[dim]+=1

    print("=== PER-DIMENSION CRAMER's V: full-sample vs Qwen∩DeepSeek agree-subset ===")
    print(f"{'dim':<13}{'full V(6)':>11}{'agree V(6)':>12}{'agree n(6)':>12}{'full V(8)':>11}{'agree V(8)':>12}{'agree n(8)':>12}")
    for dim in DIMS:
        fv6=full_major[dim][1]; av6=agree_major[dim][1]; an6=agree_major[dim][2]
        fv8=full_all[dim][1];   av8=agree_all[dim][1];   an8=agree_all[dim][2]
        print(f"{DIM_LABEL[dim]:<13}{('' if fv6 is None else fv6):>11}{('' if av6 is None else av6):>12}{an6:>12}{('' if fv8 is None else fv8):>11}{('' if av8 is None else av8):>12}{an8:>12}")
    print()
    print("=== agree-subset size and excluded (disagree) counts per dimension ===")
    print(f"{'dim':<13}{'total':>9}{'agree':>9}{'excluded':>10}{'agree%':>9}")
    for dim in DIMS:
        t=total_n[dim]; a=agree_n[dim]
        print(f"{DIM_LABEL[dim]:<13}{t:>9}{a:>9}{t-a:>10}{100*a/t:>8.1f}%")
    print()
    print("=== per-group YES counts in agree-subset (6 major groups) ===")
    ag=agree_major
    hdr="dim       "+"".join(f"{g:>9}" for g in GROUPS_MAJOR)
    print(hdr)
    for dim in DIMS:
        yes=ag[dim][3]
        print(f"{DIM_LABEL[dim]:<10}"+"".join(f"{y:>9}" for y in yes))
    print()
    print("=== sanity: full-sample V (all 8 groups, Qwen labels) should match paper ===")
    print("   paper: Digital/Soc 0.263, Emotional 0.203, Physical 0.185, Mental/Cog 0.095, Existential 0.041")
    print("   here :", {DIM_LABEL[d]:full_all[d][1] for d in DIMS})

if __name__=="__main__":
    main()
