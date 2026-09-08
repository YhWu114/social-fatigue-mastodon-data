#!/usr/bin/env python3
"""
AN-02: re-annotate the Qwen-labeled corpus with DeepSeek-V4-Flash to build a
Qwen∩DeepSeek agree-subset, then recompute the per-dimension chi-square.

This script does BATCH PACKING: N posts per API call, returns a JSON array.
It checkpoints incrementally and resumes (skips already-done ids), so a
partial run is safe.

Phase 1 (default): a stratified PILOT (--n 300) to measure Qwen↔DeepSeek
agreement before committing to the full 63,797-post run.

Usage:
  python3 an02_deepseek_label.py --n 300          # pilot
  python3 an02_deepseek_label.py --all             # full run
  python3 an02_deepseek_label.py --all --batch 25 --workers 4
"""
import os, sys, json, glob, re, html, argparse, time, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# -------------------------------------------------------------- config
BASE_URL = "https://api.deepseek.com/v1"
MODEL    = "deepseek-v4-flash"
BATCH    = 25          # posts per API call
WORKERS  = 4           # concurrent calls
SEED     = 42

DIMS = ["physical_fatigue","mental_cognitive_fatigue","emotional_fatigue",
        "digital_social_fatigue","existential_resignation"]
DIM_DESC = {
 "physical_fatigue": "bodily tiredness, physical exhaustion, lack of energy, sleep deprivation, needing rest/recovery; complaints about the body being worn out",
 "mental_cognitive_fatigue": "cognitive overload: brain fog, difficulty concentrating, mental exhaustion from thinking/decision-making, feeling one's mind is overloaded or burnt out (NOT merely feeling sad or emotionally drained)",
 "emotional_fatigue": "emotional exhaustion, feeling emotionally drained, hurt, irritable, or overwhelmed by feelings; interpersonal/emotional burnout",
 "digital_social_fatigue": "tiredness from social media / online platforms / notifications / constant digital interaction; social overload; wanting to disconnect, mute, or delete accounts",
 "existential_resignation": "an explicit stance of opting out / giving up on effort in society or life: resignation, fatalistic withdrawal, refusal to compete or participate (e.g. Chinese 'lying flat' 躺平, 'letting it rot' 摆烂, or equivalent 'what's the point, I quit'). STRICT: label ONLY when the post expresses this resigned/disengaged stance. DO NOT label ordinary sadness, tiredness, complaining, or transient hopelessness that is not a stance of giving up.",
}
OUT = "results/an02_deepseek_labels.jsonl"
CKPT_DONE = set()

SYSTEM = (
 "You are a careful multilingual social-media fatigue annotator. "
 "Given short social-media posts (any language: English, Chinese, Japanese, German, French, Spanish, Dutch, Portuguese), "
 "decide which of the five fatigue dimensions each post expresses. "
 "A post may express more than one dimension, or none."
)

def user_prompt(items):
    dim_block = "\n".join(f"- {k}: {v}" for k,v in DIM_DESC.items())
    posts = "\n".join(f"[{i}] {t}" for i,t in items)
    return (
 f"Five fatigue dimensions:\n{dim_block}\n\n"
 "Classify each post below into the subset of dimensions it expresses.\n"
 "Respond with ONLY a valid JSON array (no prose, no markdown), one object per post in the same order, "
 "formatted exactly as: [{\"idx\":0,\"categories\":[\"emotional_fatigue\"]}, ...]. "
 "Use ONLY these five keys: " + ", ".join(DIMS) + ".\n\n"
 f"Posts:\n{posts}"
    )

# -------------------------------------------------------------- data
def strip_html(t):
    if not t: return ""
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"http\S+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def load_labeled():
    files = (glob.glob("results/mastodon_qwen_8b/livefeeds_*.jsonl") +
             glob.glob("results/mastodon_qwen_8b_non_en/livefeeds_*.jsonl"))
    out = []
    for f in files:
        for line in open(f):
            line=line.strip()
            if not line: continue
            d=json.loads(line)
            c=d.get("fatigue_categories")
            if not c: continue
            text=strip_html(d.get("content") or d.get("text") or "")
            if len(text) < 3: continue
            out.append({
                "id": d.get("id") or d.get("_id") or d.get("uri"),
                "lang": d.get("language"),
                "text": text,
                "qwen_categories": c if isinstance(c,list) else [k for k,v in c.items() if v],
            })
    return out

def load_done():
    if not os.path.exists(OUT): return
    for line in open(OUT):
        line=line.strip()
        if not line: continue
        try: d=json.loads(line)
        except: continue
        if d.get("deepseek_categories") is not None:
            CKPT_DONE.add(d.get("id"))

# -------------------------------------------------------------- API
def make_client():
    return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)

def call_batch(client, items, attempt=0):
    """items: list of (idx, text). Returns dict idx->categories or raises."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":user_prompt(items)}],
        temperature=0.0, max_tokens=2048, timeout=90,
        extra_body={"reasoning_effort":"none"},  # disable thinking -> fast + fits token budget
    )
    content = resp.choices[0].message.content or ""
    # extract the JSON array defensively (handle ```json fences / prose)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.DOTALL)
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    arr = json.loads(m.group(0) if m else raw)
    result = {}
    for obj in arr:
        i = obj.get("idx")
        cats = obj.get("categories") or []
        # keep only valid dims
        cats = [c for c in cats if c in DIMS]
        result[i] = cats
    return result

# -------------------------------------------------------------- agreement
def cohen_kappa(a, b):
    # a,b: lists of 0/1 same length
    n=len(a)
    if n==0: return None
    po=sum(1 for x,y in zip(a,b) if x==y)/n
    pa=sum(a)/n; pb=sum(b)/n
    pe=pa*pb+(1-pa)*(1-pb)
    return (po-pe)/(1-pe) if (1-pe)!=0 else None

def report(records):
    print("\n==================== AN-02 PILOT AGREEMENT ====================")
    print(f"posts compared : {len(records)}")
    full=0
    per=collections.defaultdict(lambda:{"qy":0,"dy":0,"agree":0,"n":0,"qa":[],"da":[]})
    for r in records:
        q=set(r["qwen_categories"]); d=set(r["deepseek_categories"])
        if q==d: full+=1
        for dim in DIMS:
            p=per[dim]; p["n"]+=1
            qy=1 if dim in q else 0; dy=1 if dim in d else 0
            p["qy"]+=qy; p["dy"]+=dy; p["agree"]+= (qy==dy)
            p["qa"].append(qy); p["da"].append(dy)
    print(f"full 5-dim exact match: {full}/{len(records)} = {100*full/len(records):.1f}%")
    print(f"{'dimension':<26}{'Qwen+':>8}{'DeepSeek+':>12}{'acc':>8}{'kappa':>8}")
    for dim in DIMS:
        p=per[dim]
        acc=p["agree"]/p["n"]
        k=cohen_kappa(p["qa"],p["da"])
        print(f"{dim:<26}{p['qy']:>8}{p['dy']:>12}{100*acc:>7.1f}%{('' if k is None else f'{k:.3f}'):>8}")
    print("==============================================================\n")

# -------------------------------------------------------------- main
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="pilot sample size")
    ap.add_argument("--all", action="store_true", help="run full corpus")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args=ap.parse_args()

    all_posts = load_labeled()
    load_done()
    todo = [p for p in all_posts if p["id"] not in CKPT_DONE]
    print(f"labeled total={len(all_posts)}  already-done={len(all_posts)-len(todo)}  todo={len(todo)}")

    if args.all:
        sample = todo
        print(f"FULL RUN: {len(sample)} posts")
    else:
        # stratified pilot: sample across languages for multilingual coverage
        by_lang=collections.defaultdict(list)
        for p in todo: by_lang[p["lang"]].append(p)
        rng=__import__("random").Random(SEED)
        sample=[]
        per_lang=max(1, args.n//max(1,len(by_lang)))
        for lg,ps in by_lang.items():
            sample+=rng.sample(ps, min(per_lang, len(ps)))
        sample=sample[:args.n]
        print(f"PILOT: {len(sample)} posts across {len(by_lang)} languages")

    client=make_client()
    done_records=[]
    def persist(rec):
        with open(OUT,"a") as f:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")

    # process in batches with limited concurrency
    batches=[sample[i:i+args.batch] for i in range(0,len(sample),args.batch)]
    print(f"-> {len(batches)} API calls ({args.batch}/call, {args.workers} workers)")
    ok=0; fail=0
    def worker(b):
        items=[(j,p["text"]) for j,p in enumerate(b)]
        for attempt in range(4):
            try:
                res=call_batch(client, items)
                recs=[]
                for j,p in enumerate(b):
                    recs.append({**{k:p[k] for k in ("id","lang","text","qwen_categories")},
                                 "deepseek_categories": res.get(j, [])})
                return recs
            except Exception as e:
                if attempt==3:
                    return [{"id":p["id"],"lang":p["lang"],"text":p["text"],
                             "qwen_categories":p["qwen_categories"],
                             "deepseek_categories":None,"error":str(e)[:200]} for p in b]
                time.sleep(2*(attempt+1))
        return []
    t0=time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(worker,b) for b in batches]
        for fut in as_completed(futs):
            recs=fut.result()
            for r in recs:
                persist(r); done_records.append(r)
                if r.get("deepseek_categories") is not None: ok+=1
                else: fail+=1
    print(f"done in {time.time()-t0:.1f}s  ok={ok} fail={fail}  -> {OUT}")
    # report only on successfully labeled
    ok_recs=[r for r in done_records if r.get("deepseek_categories") is not None]
    if ok_recs: report(ok_recs)

if __name__=="__main__":
    main()
