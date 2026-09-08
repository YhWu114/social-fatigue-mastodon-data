"""
AN-01 FULL-SCALE validation: classify the ENTIRE unfiltered raw corpus with Qwen3-8B
zero-shot (no lexicon gate), estimate fatigue-dimension proportions, and compare to the
lexicon-gated 63,797 results. Full usable corpus = 214,002 posts across 8 languages.

Run with NO_PROXY so requests hit the vLLM endpoint directly (bypass the broker proxy).
Resumable: writes incremental checkpoints to results/an01_full_checkpoint.jsonl
and a final summary to results/an01_full.json.
"""
import json, re, os, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

EP = "http://localhost:8004/v1/chat/completions"
MODEL = "qwen3-8B-instruct"
RAW_DIR = Path("/home/user/social-fatigue/results/mastodon")
LLM_EN  = Path("/home/user/social-fatigue/results/mastodon_qwen_8b")
LLM_NON = Path("/home/user/social-fatigue/results/mastodon_qwen_8b_non_en")
OUT      = Path("/home/user/social-fatigue/results/an01_full.json")
CKPT     = Path("/home/user/social-fatigue/results/an01_full_checkpoint.jsonl")

TARGET_LANGS = ["en","zh","ja","de","es","fr","nl","pt"]
ALLOWED = ["physical_fatigue","mental_cognitive_fatigue","emotional_fatigue",
           "digital_social_fatigue","existential_resignation"]
CAT_LABEL = {"physical_fatigue":"Physical","mental_cognitive_fatigue":"Mental/Cognitive",
             "emotional_fatigue":"Emotional","digital_social_fatigue":"Digital/Social",
             "existential_resignation":"Existential"}

SYS_EN = ('You are an expert sociologist classifying social media posts.\n'
 'Determine if the text expresses genuine "Social, Mental, or Physical Fatigue".\n'
 'Strictly exclude:\n'
 '1. Sarcasm, rants, or complaints about daily inconveniences (e.g., traffic, chores, people in the way).\n'
 '2. Objective news, song lyrics, or automated updates.\n\n'
 'Categories available: physical_fatigue, mental_cognitive_fatigue, emotional_fatigue, digital_social_fatigue, existential_resignation.\n\n'
 'You must output ONLY valid JSON. Format:\n'
 '{\n  "is_fatigue": true or false,\n'
 '  "categories": {"category_name": confidence_score_between_0_and_1},\n'
 '  "reason": "Brief reason for your decision"\n}')

SYS_NONEN = ('You are an expert sociologist.\n'
 'Determine if the text expresses genuine "Social, Mental, or Physical Fatigue".\n'
 'The text will be in Japanese, German, French, Chinese, Spanish, Dutch, or Portuguese.\n'
 'Read the original text carefully, but you MUST output the JSON and your \'reason\' in ENGLISH.\n'
 'Strictly exclude:\n'
 '1. Sarcasm, rants, or complaints about daily inconveniences (e.g., traffic).\n'
 '2. Objective news, song lyrics, or automated updates.\n\n'
 'Categories available: physical_fatigue, mental_cognitive_fatigue, emotional_fatigue, digital_social_fatigue, existential_resignation.\n\n'
 'You must output ONLY valid JSON. Format:\n'
 '{\n  "is_fatigue": true or false,\n'
 '  "categories": {"category_name": confidence_score_between_0_and_1},\n'
 '  "reason": "Brief reason for your decision in ENGLISH"\n}')

def sys_prompt(lang): return SYS_EN if lang == "en" else SYS_NONEN

def clean(text):
    t = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", t).strip()

def passes_basic(post):
    lang = (post.get("language") or "").lower()
    if lang not in TARGET_LANGS: return None
    if post.get("account",{}).get("bot"): return None
    txt = clean(post.get("content",""))
    if len(txt) < 15: return None
    low = txt.lower()
    if any(k in low for k in ["nowplaying","listeningto","utm_source","dlvr.it","#bot"," rss","news roundup","podcast"]):
        return None
    return lang, txt

def parse_response(text):
    if text is None: return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```json", "", text); text = re.sub(r"```", "", text)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: return None
    try: d = json.loads(m.group(0))
    except: return None
    is_fat = bool(d.get("is_fatigue", False))
    cats = d.get("categories", {}) or {}
    valid = {}
    for k,v in cats.items():
        if k in ALLOWED:
            try: valid[k] = min(max(float(v),0.0),1.0)
            except: valid[k] = 0.9
    return {"is_fatigue": is_fat, "categories": valid}

def classify(lang, txt):
    for attempt in range(4):
        try:
            r = requests.post(EP, json={
                "model": MODEL,
                "messages":[{"role":"system","content":sys_prompt(lang)},
                            {"role":"user","content":f'Text: "{txt}"\nAnalyze this and output ONLY valid JSON.'}],
                "temperature":0.1, "max_tokens": 256, "timeout":30,
                "chat_template_kwargs": {"enable_thinking": False}}, timeout=30)
            if r.status_code == 200:
                return parse_response(r.json()["choices"][0]["message"]["content"])
        except Exception:
            time.sleep(0.5)
    return None

def sample_raw():
    pools = {l:[] for l in TARGET_LANGS}
    for jf in sorted(RAW_DIR.glob("livefeeds_*.json")):
        data = json.load(open(jf, encoding="utf-8", errors="replace"))
        for post in data:
            res = passes_basic(post)
            if not res: continue
            lang, txt = res
            pools[lang].append(txt)
    return pools

def gated_proportions():
    counts = {l:{c:0 for c in ALLOWED} for l in TARGET_LANGS}
    totals = {l:0 for l in TARGET_LANGS}
    for d in [LLM_EN, LLM_NON]:
        for jf in sorted(d.glob("*.jsonl")):
            for line in open(jf, encoding="utf-8", errors="replace"):
                try: row=json.loads(line)
                except: continue
                lang=(row.get("language") or "").lower()
                if lang not in TARGET_LANGS: continue
                cats=row.get("fatigue_categories",[]) or []
                if not cats: continue
                totals[lang]+=1
                for c in set(cats):
                    if c in ALLOWED: counts[lang][c]+=1
    return {l:{c: (counts[l][c]/totals[l]*100 if totals[l] else 0) for c in ALLOWED}
            for l in TARGET_LANGS}, totals

def main():
    t0=time.time()
    print(f"[t=0] Loading full raw corpus (unfiltered)...", flush=True)
    pools = sample_raw()
    for l in TARGET_LANGS:
        print(f"  {l}: {len(pools[l])} usable posts", flush=True)
    total = sum(len(pools[l]) for l in TARGET_LANGS)
    print(f"[t={time.time()-t0:.0f}s] Total to classify: {total}", flush=True)

    results = {l:{"n_total":len(pools[l]),"n_fatigue":0,"dims":{c:0 for c in ALLOWED}} for l in TARGET_LANGS}
    ckpt_f = open(CKPT,"w",encoding="utf-8")
    done=0
    with ThreadPoolExecutor(max_workers=96) as ex:
        fut={}
        for l in TARGET_LANGS:
            for txt in pools[l]:
                fut[ex.submit(classify, l, txt)] = l
        for f in as_completed(fut):
            l=fut[f]; done+=1
            r=f.result()
            rec={"lang":l,"ok": r is not None, "is_fatigue": bool(r["is_fatigue"]) if r else None}
            if r:
                if r["is_fatigue"]:
                    results[l]["n_fatigue"]+=1
                    for c in r["categories"]:
                        results[l]["dims"][c]+=1
            ckpt_f.write(json.dumps(rec,ensure_ascii=False)+"\n")
            if done % 2000 == 0:
                ckpt_f.flush()
                print(f"[t={time.time()-t0:.0f}s] classified {done}/{total}  ({done/total*100:.1f}%)", flush=True)
    ckpt_f.close()

    unfilt={}
    for l in TARGET_LANGS:
        nf=results[l]["n_fatigue"]
        unfilt[l]={c:(results[l]["dims"][c]/nf*100 if nf else 0) for c in ALLOWED}
    gated, gtot = gated_proportions()
    out={"unfiltered":unfilt,"gated":gated,
         "unfiltered_n_fatigue":{l:results[l]["n_fatigue"] for l in TARGET_LANGS},
         "unfiltered_n_total":{l:results[l]["n_total"] for l in TARGET_LANGS},
         "gated_n_total":gtot}
    json.dump(out, open(OUT,"w"), indent=2, ensure_ascii=False)
    print(f"\n[t={time.time()-t0:.0f}s] DONE. Full unfiltered vs gated proportions:", flush=True)
    print(f"{'lang':>5} {'n_unf':>7} {'n_fat':>7} | " + " ".join(f"{CAT_LABEL[c][:4]:>5}" for c in ALLOWED), flush=True)
    for l in TARGET_LANGS:
        u=unfilt[l]; g=gated[l]
        print(f"{l:>5} {results[l]['n_total']:>7} {results[l]['n_fatigue']:>7} | " +
              " ".join(f"{u[c]:4.1f}/{g[c]:4.1f}" for c in ALLOWED) + f"   (unfilt/gated %)", flush=True)
    print(f"\nSaved -> {OUT}", flush=True)

if __name__=="__main__":
    main()
