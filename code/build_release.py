#!/usr/bin/env python3
"""
Build the AN-15 data-release package (Zenodo-ready) for the IP&M social-fatigue paper.

Packages exactly what the Data availability statement promises:
  (i)   stable post identifiers (URIs) of the 63,797 verified fatigue expressions
  (ii)  per-post derived labels (five-dimensional fatigue categories + confidence scores)
  (iii) the multilingual fatigue lexicon (TAXONOMY)
  (iv)  the exact LLM prompts (Qwen3-8B final verification, EN + non-EN)
  (v)   the analysis code used to produce the results

Raw post text is intentionally NOT redistributed (Fediverse community norms).
Posts can be re-derived from the URIs via the FediLive archive.
"""
import os, re, ast, glob, json, csv, shutil

ROOT = "/home/user/social-fatigue"
OUT  = os.path.join(ROOT, "AN15_data_release")
os.makedirs(os.path.join(OUT, "data"),    exist_ok=True)
os.makedirs(os.path.join(OUT, "lexicon"), exist_ok=True)
os.makedirs(os.path.join(OUT, "prompts"), exist_ok=True)
os.makedirs(os.path.join(OUT, "code"),    exist_ok=True)

DIMS = ["physical_fatigue","mental_cognitive_fatigue","emotional_fatigue",
        "digital_social_fatigue","existential_resignation"]

def to_group(lang):
    if lang in ("zh-CN","zh"): return "zh-S"
    if lang in ("zh-TW","zh-HK"): return "zh-T"
    if lang in ("es","fr"): return "es/fr"
    return lang

# ---------------------------------------------------------------- (i)+(ii) labels
label_files = (glob.glob(os.path.join(ROOT,"results/mastodon_qwen_8b/livefeeds_*.jsonl")) +
               glob.glob(os.path.join(ROOT,"results/mastodon_qwen_8b_non_en/livefeeds_*.jsonl")))
records = []
for f in label_files:
    for line in open(f):
        line=line.strip()
        if not line: continue
        d=json.loads(line)
        cats=d.get("fatigue_categories")
        if not cats: continue
        uri=d.get("uri") or d.get("id") or d.get("_id")
        weights=d.get("fatigue_weights") or {}
        if not isinstance(weights, dict): weights={}
        records.append({
            "uri": uri,
            "language": d.get("language"),
            "group": to_group(d.get("language")),
            "categories": cats if isinstance(cats,list) else [k for k,v in cats.items() if v],
            "weights": {k:(weights.get(k) if isinstance(weights.get(k),(int,float)) else 0.0) for k in DIMS},
        })

print("[labels] extracted %d verified fatigue posts" % len(records))

csv_path = os.path.join(OUT,"data","fatigue_labels_63797.csv")
with open(csv_path,"w",newline="",encoding="utf-8") as fh:
    w=csv.writer(fh)
    w.writerow(["uri","language","group","categories",
                "physical_fatigue","mental_cognitive_fatigue","emotional_fatigue",
                "digital_social_fatigue","existential_resignation"])
    for r in records:
        w.writerow([r["uri"], r["language"], r["group"], ";".join(r["categories"]),
                    r["weights"]["physical_fatigue"], r["weights"]["mental_cognitive_fatigue"],
                    r["weights"]["emotional_fatigue"], r["weights"]["digital_social_fatigue"],
                    r["weights"]["existential_resignation"]])
with open(os.path.join(OUT,"data","fatigue_labels_63797.jsonl"),"w",encoding="utf-8") as fh:
    for r in records:
        fh.write(json.dumps(r,ensure_ascii=False)+"\n")
print("[labels] wrote %s" % csv_path)

# ---------------------------------------------------------------- (iii) lexicon
def extract_balanced(src, start):
    depth=0; i=start
    while i < len(src):
        c=src[i]
        if c=='{': depth+=1
        elif c=='}':
            depth-=1
            if depth==0: return src[start:i+1], i+1
        i+=1
    return None, i

src_lex = open(os.path.join(ROOT,"filter_by_keywords.py")).read()
m=re.search(r"TAXONOMY\s*=\s*\{", src_lex)
block, _ = extract_balanced(src_lex, m.end()-1)
taxonomy = ast.literal_eval(block)
with open(os.path.join(OUT,"lexicon","fatigue_taxonomy.json"),"w",encoding="utf-8") as fh:
    json.dump(taxonomy, fh, ensure_ascii=False, indent=2)
shutil.copy(os.path.join(ROOT,"filter_by_keywords.py"), os.path.join(OUT,"lexicon","fatigue_taxonomy_source.py"))
print("[lexicon] dimensions: %s" % list(taxonomy.keys()))

# ---------------------------------------------------------------- (iv) prompts
def extract_triple_quote(src, key):
    mm=re.search(key+r'\s*=\s*"""(.*?)"""', src, re.DOTALL)
    return mm.group(1) if mm else None

for tag, fn in [("en","find_by_keywords_bert.py"), ("nonen","find_by_keywords_no_en.py")]:
    s=open(os.path.join(ROOT,fn)).read()
    sp=extract_triple_quote(s,"SYSTEM_PROMPT")
    if sp:
        with open(os.path.join(OUT,"prompts",f"qwen_final_verification_{tag}.txt"),"w",encoding="utf-8") as fh:
            fh.write(sp.strip()+"\n")
ut='Text: "{text}"\nAnalyze this and output ONLY valid JSON.'
with open(os.path.join(OUT,"prompts","user_template.txt"),"w",encoding="utf-8") as fh:
    fh.write(ut+"\n")
print("[prompts] wrote SYSTEM_PROMPT (en/nonen) + user_template")

# ---------------------------------------------------------------- (v) code
CODE_FILES = [
    "filter_by_keywords.py",
    "find_by_keywords_bert.py",
    "find_by_keywords_no_en.py",
    "robustness_analyses.py",
    "an01_full_v2.py",
    "an02_deepseek_label.py",
    "an02_agreesubset_chi2.py",
    "gen_fig2_annotated.py",
    "build_release.py",
]
for fn in CODE_FILES:
    p=os.path.join(ROOT,fn)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(OUT,"code",fn))
print("[code] copied %d scripts" % len(CODE_FILES))

# ---------------------------------------------------------------- README + LICENSE
readme = (
"# Social Fatigue on Mastodon - Data Release (AN-15)\n\n"
"Companion data and code for the IP&M paper on cross-cultural social fatigue on Mastodon.\n\n"
"## Contents\n"
"| File | Description |\n"
"|---|---|\n"
"| `data/fatigue_labels_63797.csv` | Per-post labels for the 63,797 verified fatigue expressions. Columns: uri, language, group, categories (semicolon-joined 5-dim labels), and one confidence-score column per dimension (0 = absent). |\n"
"| `data/fatigue_labels_63797.jsonl` | Same records in JSONL. |\n"
"| `lexicon/fatigue_taxonomy.json` | The multilingual fatigue keyword taxonomy (5 dimensions) used for candidate retrieval. |\n"
"| `lexicon/fatigue_taxonomy_source.py` | Source file the taxonomy was extracted from. |\n"
"| `prompts/qwen_final_verification_en.txt` | System prompt for the English Qwen3-8B final verification pass. |\n"
"| `prompts/qwen_final_verification_nonen.txt` | System prompt for the non-English pass. |\n"
"| `prompts/user_template.txt` | User-message template. |\n"
"| `code/` | Analysis code that produces every result in the paper. |\n\n"
"## Ethical note\n"
"Raw post text is intentionally NOT redistributed to respect Fediverse community norms and minimize re-identification risk. Each record carries a stable post URI; the original posts can be re-derived from these identifiers using the public FediLive archive (Min et al., 2025). No personally identifying information is included.\n\n"
"## License\n"
"CC-BY-4.0 (data + lexicon + prompts) and MIT (code), per Zenodo deposit.\n"
)
with open(os.path.join(OUT,"README.md"),"w",encoding="utf-8") as fh:
    fh.write(readme)

license_txt = (
"CC-BY-4.0\n\n"
"This data release is licensed under the Creative Commons Attribution 4.0 "
"International License (CC-BY-4.0). You are free to share and adapt the material, "
"provided appropriate credit is given. The analysis code in `code/` is released under the MIT License.\n"
)
with open(os.path.join(OUT,"LICENSE"),"w",encoding="utf-8") as fh:
    fh.write(license_txt)

print("\nDONE -> %s" % OUT)
print("label rows: %d" % len(records))
