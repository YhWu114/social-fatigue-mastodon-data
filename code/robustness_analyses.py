#!/usr/bin/env python3
"""
Supplementary robustness / validation analyses for the IP&M social-fatigue paper.
Reads the labeled 63,797-post dataset in results/mastodon_qwen_8b*/ and the
inter-annotator + keyword files, and computes the Yihang tasks:
  AN-01  lexicon term counts per language x dimension
  AN-02  annotator sensitivity (interrater_sample_combined)
  AN-03  instance robustness
  AN-06  author-level analysis
  AN-07  adjusted standardized residuals + expected<5 cells
  AN-08  Wilson CIs on proportions + bootstrap CI on Cramer's V
  AN-11  Mastodon language-label validation (langdetect)
  AN-12  length-threshold distribution
  AN-14  temporal breakdown
Outputs results_yihang.json and a printed report.
"""
import json, glob, collections, re, html, random, math, os
import numpy as np
from scipy import stats

random.seed(42)
DIMS = ["physical_fatigue","mental_cognitive_fatigue","emotional_fatigue",
        "digital_social_fatigue","existential_resignation"]
DIM_LABEL = {"physical_fatigue":"Physical","mental_cognitive_fatigue":"Mental/Cog",
             "emotional_fatigue":"Emotional","digital_social_fatigue":"Digital/Soc",
             "existential_resignation":"Existential"}

# Map raw Mastodon language -> paper group
def to_group(lang):
    if lang is None: return None
    lang = str(lang)
    if lang in ("zh-CN","zh"): return "zh-S"
    if lang in ("zh-TW","zh-HK"): return "zh-T"
    return lang  # en, ja, de, fr, es, pt, nl

def strip_html(t):
    if not t: return ""
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"http\S+", " ", t)
    return t.strip()

# ---------------------------------------------------------------- load corpus
def load_labeled():
    files = (glob.glob("results/mastodon_qwen_8b/livefeeds_*.jsonl") +
             glob.glob("results/mastodon_qwen_8b_non_en/livefeeds_*.jsonl"))
    rows = []
    for fn in files:
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: d = json.loads(line)
                except: continue
                cats = d.get("fatigue_categories") or []
                if not cats: continue
                g = to_group(d.get("language"))
                if g is None: continue
                rows.append({
                    "group": g,
                    "lang": d.get("language"),
                    "instance": d.get("instance_name"),
                    "author": (d.get("account") or {}).get("id"),
                    "created_at": d.get("created_at"),
                    "content": d.get("content"),
                    "cats": [c for c in cats if c in DIMS],
                })
    return rows

# ---------------------------------------------------------------- AN-03 instance
def an03_instance(rows):
    by_group = collections.defaultdict(list)
    for r in rows: by_group[r["group"]].append(r)
    out = {}
    dropped = []
    for g, rs in by_group.items():
        inst = collections.Counter(r["instance"] for r in rs if r["instance"])
        n = len(rs)
        top3 = sum(c for _,c in inst.most_common(3))
        shares = [c/n for c in inst.values()]
        hhi = sum(s*s for s in shares)
        top_inst, top_n = inst.most_common(1)[0]
        out[g] = {"n":n, "n_instances":len(inst), "top3_share":top3/n,
                  "hhi":hhi, "top_instance":top_inst, "top_instance_share":top_n/n}
        dropped.append((g, top_inst, top_n/n))
    # re-run dimension comparison with top instance per group removed
    kept = [r for r in rows if not any(r["group"]==g and r["instance"]==ti for g,ti,_ in dropped)]
    return out, cramers_per_dim(kept, label="AN-03 (largest instance/group removed)")

# ---------------------------------------------------------------- AN-06 author
def an06_author(rows):
    by_group = collections.defaultdict(list)
    for r in rows: by_group[r["group"]].append(r)
    out = {}
    for g, rs in by_group.items():
        authors = collections.Counter(r["author"] for r in rs if r["author"])
        n = len(rs); na = len(authors)
        cnts = list(authors.values())
        maxp = max(cnts); sing = sum(1 for c in cnts if c==1)/na
        top_share = cnts[0]/n if cnts else 0
        out[g] = {"n_posts":n, "n_authors":na, "posts_per_author":round(n/na,2),
                  "max_posts_by_one_author":maxp, "pct_authors_single_post":round(100*sing,1),
                  "top_author_share":round(top_share,3)}
    # one post per author globally
    seen=set(); dedup=[]
    for r in rows:
        a=r["author"]
        if a is None or a not in seen:
            dedup.append(r); seen.add(a)
    return out, cramers_per_dim(dedup, label="AN-06 (one post per author)")

# ---------------------------------------------------------------- per-dim Cramer's V
def cramers_per_dim(rows, label=""):
    groups = ["en","zh-S","ja","de","zh-T","fr","es","pt","nl"]
    gset = [g for g in groups if any(r["group"]==g for r in rows)]
    out = {}
    for dim in DIMS:
        # build 2 x len(gset) contingency: rows = [fatigue_yes, fatigue_no]
        cols = {g: [0,0] for g in gset}
        for r in rows:
            yes = 1 if dim in r["cats"] else 0
            cols[r["group"]][yes]+=1
        mat = np.array([cols[g] for g in gset])  # groups x 2
        # chi-square on groups x 2
        chi2,p,dof,_ = stats.chi2_contingency(mat, correction=False)
        n = mat.sum()
        v = math.sqrt(chi2/(n*(min(mat.shape)-1)))
        out[dim]={"chi2":round(chi2,1),"V":round(v,3),"p":p}
    return out

# ---------------------------------------------------------------- AN-07 residuals
def an07_residuals(rows):
    groups = ["en","zh-S","ja","de","zh-T","fr","es","pt","nl"]
    gset = [g for g in groups if any(r["group"]==g for r in rows)]
    out = {}
    for dim in DIMS:
        cols = {g:[0,0] for g in gset}
        for r in rows:
            yes = 1 if dim in r["cats"] else 0
            cols[r["group"]][yes]+=1
        mat = np.array([cols[g] for g in gset])
        chi2,p,dof,exp = stats.chi2_contingency(mat, correction=False)
        n = mat.sum()
        # adjusted standardized residuals: (O-E)/sqrt(E*(1-row_margin/n)*(1-col_margin/n))
        rm = mat.sum(1, keepdims=True); cm = mat.sum(0, keepdims=True)
        resid = (mat-exp)/np.sqrt(exp*(1-rm/n)*(1-cm/n))
        exp_lt5 = int((exp<5).sum())
        # find driving cells |resid|>1.96
        drivers=[]
        for i,g in enumerate(gset):
            for j in (0,1):
                if abs(resid[i,j])>1.96:
                    drivers.append((g, "yes" if j==1 else "no", round(float(resid[i,j]),2)))
        out[dim]={"chi2":round(chi2,1),"V":round(math.sqrt(chi2/(n*(min(mat.shape)-1))),3),
                  "p":p,"expected_lt5_cells":exp_lt5,"n_cells":int(exp.size),
                  "driving_cells":drivers}
    return out

# ---------------------------------------------------------------- AN-08 CIs
def wilson(k, n, z=1.96):
    if n==0: return (0,0)
    p=k/n; denom=1+z*z/n
    centre=(p+z*z/(2*n))/denom
    half=(z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/denom
    return (max(0,centre-half), min(1,centre+half))

def an08_ci(rows):
    groups=["en","zh-S","ja","de","zh-T","fr","es","pt","nl"]
    props={}
    for g in groups:
        rs=[r for r in rows if r["group"]==g]
        n=len(rs)
        props[g]={dim: (sum(1 for r in rs if dim in r["cats"]), n) for dim in DIMS}
    # bootstrap Cramer's V per dim
    boot={}
    N=1000
    for dim in DIMS:
        vs=[]
        for _ in range(N):
            samp=random.choices(rows,k=len(rows))
            gset=[g for g in groups if any(r["group"]==g for r in samp)]
            cols={g:[0,0] for g in gset}
            for r in samp:
                cols[r["group"]][1 if dim in r["cats"] else 0]+=1
            mat=np.array([cols[g] for g in gset])
            chi2,_,_,_=stats.chi2_contingency(mat,correction=False)
            vs.append(math.sqrt(chi2/(mat.sum()*(min(mat.shape)-1))))
        boot[dim]=(round(np.percentile(vs,2.5),3), round(np.percentile(vs,97.5),3),
                   round(np.mean(vs),3))
    return props, boot

# ---------------------------------------------------------------- AN-11 lang label
def an11_lang(rows, sample_per=400):
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed=42
    by_group=collections.defaultdict(list)
    for r in rows:
        t=strip_html(r["content"])
        if len(t)>=10: by_group[r["group"]].append(t)
    out={}; overall=collections.Counter(); overall_collapsed=collections.Counter()
    for g, texts in by_group.items():
        samp=random.sample(texts, min(sample_per,len(texts)))
        agree=0; agree_coll=0; tot=0
        det_map={}
        for t in samp:
            try: det=detect(t).split("-")[0]
            except: continue
            tot+=1
            # normalize gold: zh variants -> zh
            gold = "zh" if g.startswith("zh") else g
            if det==gold: agree_coll+=1; agree+=1
            elif det==g: agree+=1
            det_map.setdefault(det,0); det_map[det]+=1
        out[g]={"n":tot,"exact_match_pct":round(100*agree/tot,1) if tot else None,
                "collapsed_zh_match_pct":round(100*agree_coll/tot,1) if tot else None,
                "detected_dist":det_map}
        overall["tot"]+=tot; overall["match"]+=agree
        overall_collapsed["tot"]+=tot; overall_collapsed["match"]+=agree_coll
    out["_overall"]={"exact_match_pct":round(100*overall["match"]/overall["tot"],1),
                     "collapsed_zh_match_pct":round(100*overall_collapsed["match"]/overall_collapsed["tot"],1)}
    return out

# ---------------------------------------------------------------- AN-12 length
def an12_length(rows):
    out={}
    for g in ["en","zh-S","ja","de","zh-T","fr","es","pt","nl"]:
        lens=[len(strip_html(r["content"])) for r in rows if r["group"]==g]
        if not lens: continue
        lens=np.array(lens)
        out[g]={"median":int(np.median(lens)),"mean":round(float(lens.mean()),1),
                "pct_below_30":round(100*float((lens<30).mean()),1),
                "pct_below_15":round(100*float((lens<15).mean()),1)}
    return out

# ---------------------------------------------------------------- AN-14 temporal
def an14_temporal(rows):
    by_month=collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        ca=r["created_at"]
        if not ca: continue
        m=ca[:7]  # YYYY-MM
        by_month[m][r["group"]]+=1
    months=sorted(by_month)
    out={"months":months}
    for g in ["en","zh-S","ja","de","zh-T","fr","es","pt","nl"]:
        out[g]=[by_month[m].get(g,0) for m in months]
    return out

# ---------------------------------------------------------------- AN-01 keyword term counts
def an01_keywords():
    files=glob.glob("results/keywords/classified_posts_json_keyword*.jsonl")
    cnt=collections.defaultdict(lambda: collections.Counter())
    for fn in files:
        with open(fn) as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: d=json.loads(line)
                except: continue
                g=to_group(d.get("language"))
                if g is None: continue
                mc=d.get("matched_categories") or []
                if isinstance(mc,str):
                    try: mc=json.loads(mc)
                    except: mc=[]
                for c in mc:
                    if c in DIMS: cnt[g][c]+=1
    return {g:dict(c) for g,c in cnt.items()}

# ---------------------------------------------------------------- AN-02 annotator sensitivity
def an02_annotator():
    fn="interrater_sample_combined.jsonl"
    rows=[]
    with open(fn) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            d=json.loads(line)
            q=set(d.get("qwen_categories") or []); ds=set(d.get("deepseek_categories") or [])
            rows.append((q,ds))
    def table(subset, label):
        stats={}
        for dim in DIMS:
            qy=sum(1 for q,ds in subset if dim in q)
            dsy=sum(1 for q,ds in subset if dim in ds)
            # agreement on binary (both yes / both no / disagree)
            agree=sum(1 for q,ds in subset if (dim in q)==(dim in ds))
            n=len(subset)
            acc=agree/n
            # Cohen kappa for this dim (2x2 yes/no)
            a=sum(1 for q,ds in subset if dim in q and dim in ds)
            b=sum(1 for q,ds in subset if dim in q and dim not in ds)
            c=sum(1 for q,ds in subset if dim not in q and dim in ds)
            d0=sum(1 for q,ds in subset if dim not in q and dim not in ds)
            po=(a+d0)/n
            pe=((a+b)/n)*((a+c)/n)+((c+d0)/n)*((b+d0)/n)
            kappa=(po-pe)/(1-pe) if (1-pe)!=0 else None
            stats[dim]={"qwen_yes":qy,"ds_yes":dsy,"acc":round(acc,3),
                        "kappa":round(kappa,3) if kappa is not None else None}
        return {"n":len(subset),"per_dim":stats}
    full=table(rows,"full")
    agree_subset=[(q,ds) for q,ds in rows if q==ds]
    agree=table(agree_subset,"agree_subset")
    return {"full":full,"agree_subset":agree,
            "n_full":len(rows),"n_agree":len(agree_subset)}

# ---------------------------------------------------------------- main
def main():
    print("Loading labeled corpus...")
    rows = load_labeled()
    print(f"  loaded {len(rows)} labeled posts")
    results={}
    results["n_total"]=len(rows)

    print("AN-03 instance robustness...")
    inst, inst_cram = an03_instance(rows)
    results["AN03_instance"]=inst
    results["AN03_instance_cramers_after"]=inst_cram

    print("AN-06 author-level...")
    auth, auth_cram = an06_author(rows)
    results["AN06_author"]=auth
    results["AN06_author_cramers_after"]=auth_cram

    print("AN-07 residuals...")
    results["AN07_residuals"]=an07_residuals(rows)

    print("AN-08 CIs...")
    props, boot = an08_ci(rows)
    results["AN08_props"]={g:{dim:(k,n) for dim,(k,n) in d.items()} for g,d in props.items()}
    results["AN08_boot_cramersV"]=boot

    print("AN-11 language label...")
    results["AN11_langlabel"]=an11_lang(rows)

    print("AN-12 length...")
    results["AN12_length"]=an12_length(rows)

    print("AN-14 temporal...")
    results["AN14_temporal"]=an14_temporal(rows)

    print("AN-01 keyword term counts...")
    results["AN01_keyword_counts"]=an01_keywords()

    print("AN-02 annotator sensitivity...")
    results["AN02_annotator"]=an02_annotator()

    # original Cramer's V per dim (for comparison with after-exclusion)
    results["AN00_cramers_original"]=cramers_per_dim(rows, label="original")

    with open("results_yihang.json","w") as f:
        json.dump(results,f,indent=2,ensure_ascii=False)
    print("\nSaved results_yihang.json")
    return results

if __name__=="__main__":
    main()
