"""
Multilingual Cascading Pipeline for Social Fatigue (Non-English Top 7)
Supported: ja, de, fr, zh, es, nl, pt (English Excluded)
Model: Qwen3-8B-Instruct via local vLLM (OpenAI API)
Features: Ultra-Expanded CJK Slang Taxonomy, Concurrent API, Robust Parsing
"""

import os
import json
import ijson
import re
import warnings
import concurrent.futures
from tqdm import tqdm

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

from openai import OpenAI

# ============================================================
# 1. 路径、API & 并发参数
# ============================================================
INPUT_DIR = "/home/user/social-fatigue/mastodonposts/backup"
# 【核心修改 1】：输出到独立的文件夹，物理隔离英文数据
OUTPUT_DIR = "/home/user/social-fatigue/results/mastodon_qwen_8b_non_en"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128
MAX_CONCURRENT_REQ = 64
MIN_TEXT_LEN = 15  # 进一步放宽，因为中文“烦死了摆烂了”虽然字数少，但信息量极大

# 【核心修改 2】：只放行非英文的 Top 7 语言
ALLOWED_LANGS = {"ja", "de", "fr", "zh", "es", "nl", "pt"}

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)
MODEL_NAME = "qwen3-8B-instruct" 

# ============================================================
# 2. 垃圾信息过滤
# ============================================================
SPAM_KEYWORDS = ["arxiv.org", "utm_source", "dlvr.it", "#bot", "_bot", "bot_toot", "rss", "news roundup", "podcast"]

def is_spam(text: str) -> bool:
    return any(spam in text.lower() for spam in SPAM_KEYWORDS)

# ============================================================
# 3. 史诗级扩充多语言海选词库 (重点强化中日文互联网俚语)
# ============================================================
TAXONOMY = {
    "physical_fatigue": [
        # 中文 (含俚语/口语)
        "累", "疲惫", "犯困", "筋疲力尽", "乏力", "累瘫", "累成狗", "搬砖累", 
        "身体被掏空", "没力气", "睁不开眼", "累死", "浑身酸痛", "好困",
        # 日文
        "疲れた", "眠い", "だるい", "ヘトヘト", "くたくた", "限界", "バテバテ", "体力ゼロ", "疲労困憊", "起きられない",
        # 德、法、西、荷、葡
        "müde", "erschöpft", "schläfrig", "kraftlos", "kaputt",
        "fatigué", "épuisé", "somnolent", "vidé", "crevé",
        "cansado", "exhausto", "agotado", "reventado",
        "moe", "uitgeput", "slaperig", "gesloopt",
        "cansado", "exausto", "esgotado", "morto"
    ],
    "mental_cognitive_fatigue": [
        # 中文
        "脑雾", "精神内耗", "无法集中", "用脑过度", "脑子转不动了", "神经衰弱", 
        "费脑子", "精神恍惚", "注意力涣散", "CPU烧了", "宕机", "脑壳疼", "心力交瘁",
        # 日文
        "頭が回らない", "集中できない", "脳の疲労", "ぼーっとする", "キャパオーバー", "思考停止", "頭がパンク",
        # 德、法、西、荷、葡
        "geistige erschöpfung", "gehirnnebel", "überreizt", "überfordert",
        "fatigue mentale", "brouillard cérébral", "surstimulé", "cerveau grillé",
        "fatiga mental", "niebla mental", "sobreestimulado", "cerebro frito",
        "mentale vermoeidheid", "hersenmist", "overprikkeld",
        "fadiga mental", "névoa mental", "superestimulado"
    ],
    "emotional_fatigue": [
        # 中文
        "心累", "烦躁", "焦虑", "崩溃", "撑不住了", "职业倦怠", "破防", "抑郁", 
        "情绪稳定不下来", "烦死了", "压抑", "喘不过气", "炸毛", "emo", "致郁", "发疯","烦死了", "心态崩了", "情绪崩了", "感觉被掏空了", "感觉被榨干了", "感觉被压垮了",
        # 日文
        "病む", "メンタル", "ストレス", "しんどい", "もう無理", "燃え尽き症候群", "イライラ", "憂鬱", " 心が折れた", "感情が不安定", "精神的に疲れた",
        # 德、法、西、荷、葡
        "ausgelaugt", "burnout", "gestresst", "am ende", "kann nicht mehr",
        "épuisement émotionnel", "anxieux", "stressé", "ras le bol", "à bout",
        "agotamiento emocional", "quemado", "ansioso", "estresado", "harto",
        "emotioneel uitgeput", "angstig", "gestrest", "kan niet meer",
        "esgotamento emocional", "ansioso", "farto", "não aguento mais"
    ],
    "digital_social_fatigue": [
        # 中文
        "电子榨菜", "戒网", "数字排毒", "冲浪疲劳", "信息过载", "退网", "社交恐惧", 
        "社恐", "不想回消息", "微信恐惧症", "远离手机", "屏幕疲劳", "群消息烦",
        # 日文
        "SNS疲れ", "デジタルデトックス", "スマホ見すぎ", "情報過多", "ミュート", "リプ返せない", "既読スルーしたい", "ネット疲れ",
        # 德、法、西、荷、葡
        "digital detox", "bildschirmzeit", "informationsüberflutung",
        "détox digitale", "hors ligne", "temps d'écran", "surcharge",
        "desintoxicación digital", "desconectado", "tiempo de pantalla",
        "digitale detox", "schermtijd", "informatie-overload",
        "detox digital", "tempo de tela"
    ],
    "existential_resignation": [
        # 中文
        "躺平", "摆烂", "毁灭吧", "没意思", "随它去吧", "毫无意义", "烂透了", 
        "累觉不爱", "混吃等死", "佛系", "算了", "关我屁事", "人生无望", "咸鱼", "爱咋咋地",
        # 日文
        "どうでもいい", "無意味", "生きるのが辛い", "虚無", "何もしたくない", "終わってる", "人生詰んだ", "無気力", "サレンダー",
        # 德、法、西、荷、葡
        "aufgeben", "sinnlos", "hoffnungslos", "apathie", "mir egal",
        "à quoi bon", "inutile", "désespéré", "apathie", "peu importe",
        "rendirse", "para qué", "sin sentido", "desesperanzado", "apatía",
        "opgeven", "zinloos", "hopeloos", "apathie", "laat maar",
        "desistir", "sem sentido", "sem esperança", "tanto faz"
    ]
}

# 智能正则构建：中日文无缝匹配，字母表语言加 \b
def is_cjk(text):
    return any(
        '\u4e00' <= char <= '\u9fff' or '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff'
        for char in text
    )

cjk_keywords, alpha_keywords = [], []

for phrases in TAXONOMY.values():
    for phrase in phrases:
        p_lower = phrase.lower()
        if is_cjk(p_lower):
            cjk_keywords.append(re.escape(p_lower))
        else:
            alpha_keywords.append(re.escape(p_lower))

patterns = []
if alpha_keywords:
    patterns.append(r'\b(?:' + '|'.join(alpha_keywords) + r')\b')
if cjk_keywords:
    patterns.append(r'(?:' + '|'.join(cjk_keywords) + r')')

keyword_filter_regex = re.compile('|'.join(patterns), re.IGNORECASE)

def passes_keyword_filter(text: str) -> bool:
    return bool(keyword_filter_regex.search(text))

# ============================================================
# 4. vLLM (Qwen3-8B) 终审逻辑
# ============================================================
# 修改 Prompt，告知模型输入不再包含英文，但要求英文输出
SYSTEM_PROMPT = """You are an expert sociologist.
Determine if the text expresses genuine "Social, Mental, or Physical Fatigue".
The text will be in Japanese, German, French, Chinese, Spanish, Dutch, or Portuguese.
Read the original text carefully, but you MUST output the JSON and your 'reason' in ENGLISH.

Strictly exclude:
1. Sarcasm, rants, or complaints about daily inconveniences (e.g., traffic).
2. Objective news, song lyrics, or automated updates.

Categories available: physical_fatigue, mental_cognitive_fatigue, emotional_fatigue, digital_social_fatigue, existential_resignation.

You must output ONLY valid JSON. Format:
{
  "is_fatigue": true or false,
  "categories": {"category_name": confidence_score_between_0_and_1},
  "reason": "Brief reason for your decision in ENGLISH"
}"""

def get_llm_judgment(text: str):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Text: \"{text}\"\nAnalyze this and output ONLY valid JSON."}
            ],
            temperature=0.1,
            max_tokens=800,
            timeout=15
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None

def process_batch(texts, posts, out_file, stats):
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQ) as executor:
        results = list(executor.map(get_llm_judgment, texts))

    for post, generated_text in zip(posts, results):
        if not generated_text:
            stats["API_ERRORS"] = stats.get("API_ERRORS", 0) + 1
            continue
            
        try:
            # 暴力拆除 <think>
            text_no_think = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
            if "<think>" in text_no_think:
                text_no_think = text_no_think.split("</think>")[-1].strip()

            clean_json_str = text_no_think.replace("```json", "").replace("```", "").strip()
            start_idx = clean_json_str.find('{')
            end_idx = clean_json_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                clean_json_str = clean_json_str[start_idx:end_idx+1]
            else:
                raise ValueError("No JSON braces found")
                
            result = json.loads(clean_json_str)
            
            # 只有模型判定为 True，才继续处理
            if result.get("is_fatigue") is True and result.get("categories"):
                raw_categories = result["categories"]
                llm_reason = str(result.get("reason", ""))
                
                valid_categories = {}
                allowed_keys = ["physical_fatigue", "mental_cognitive_fatigue", "emotional_fatigue", "digital_social_fatigue", "existential_resignation"]
                
                # 终极 Schema 漂移装甲
                if isinstance(raw_categories, dict):
                    if "reason" in raw_categories:
                        if not llm_reason: llm_reason = str(raw_categories["reason"])
                        del raw_categories["reason"]
                        
                    for k, v in raw_categories.items():
                        if k in allowed_keys:
                            try: valid_categories[k] = min(max(float(v), 0.0), 1.0)
                            except: valid_categories[k] = 0.90
                                
                elif isinstance(raw_categories, list):
                    for item in raw_categories:
                        if isinstance(item, str) and item in allowed_keys:
                            valid_categories[item] = 0.90
                        elif isinstance(item, dict):
                            for k, v in item.items():
                                if k in allowed_keys:
                                    try: valid_categories[k] = float(v)
                                    except: valid_categories[k] = 0.90
                
                if valid_categories:
                    stats["TOTAL_POSTS_MATCHED"] += 1
                    post["fatigue_categories"] = list(valid_categories.keys())
                    post["fatigue_weights"] = valid_categories
                    post["llm_reason"] = llm_reason
                    
                    out_file.write(json.dumps(post, ensure_ascii=False) + "\n")
                    out_file.flush()
                
        except (json.JSONDecodeError, ValueError):
            stats["JSON_PARSE_ERRORS"] = stats.get("JSON_PARSE_ERRORS", 0) + 1

# ============================================================
# 5. 数据流引擎
# ============================================================
def stream_json_objects(file_path):
    with open(file_path, "rb") as f:
        try:
            yield from ijson.items(f, "item")
        except ijson.common.IncompleteJSONError:
            f.seek(0)
            decoder = json.JSONDecoder()
            buffer = ""
            for chunk in iter(lambda: f.read(1024 * 1024).decode("utf-8", errors="ignore"), ""):
                buffer += chunk
                while buffer:
                    buffer = buffer.lstrip()
                    try:
                        obj, idx = decoder.raw_decode(buffer)
                        yield obj
                        buffer = buffer[idx:]
                    except json.JSONDecodeError:
                        break

def process_file(input_path, output_path):
    stats = {"TOTAL_SCANNED": 0, "LANG_MATCHED": 0, "REGEX_PASSED": 0, "TOTAL_POSTS_MATCHED": 0}
    buffer_texts, buffer_posts = [], []

    with open(output_path, "a", encoding="utf-8") as out_file:
        for post in tqdm(stream_json_objects(input_path), desc=os.path.basename(input_path)):
            stats["TOTAL_SCANNED"] += 1
            
            # 【核心层】：拦截英文及无关小语种，只放行 Top 7 非英语
            lang = post.get("language")
            if not lang or not isinstance(lang, str): continue
            lang_code = lang.strip().lower().split('-')[0]
            if lang_code not in ALLOWED_LANGS: 
                continue
                
            stats["LANG_MATCHED"] += 1

            # 机器人与自动化过滤
            if post.get("reblog"): continue
            account_info = post.get("account", {})
            if account_info and (account_info.get("bot") is True or "bot" in account_info.get("acct", "").lower()): continue
            app_info = post.get("application") or {}
            if any(x in app_info.get("name", "").lower() for x in ["script", "ifttt", "zapier", "bot", "crossposter", "buffer", "rss"]): continue
            tags = [t.get("name", "").lower() for t in post.get("tags", [])]
            if "nowplaying" in tags or "listeningto" in tags: continue

            raw_html = post.get("content", "")
            if not raw_html: continue

            # 清洗文本与链接
            clean_text = BeautifulSoup(raw_html, "html.parser").get_text().replace('\n', ' ').strip()
            text_without_urls = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', clean_text)
            
            if len(text_without_urls.strip()) < MIN_TEXT_LEN: continue
            if is_spam(clean_text): continue

            # 正则海选
            if not passes_keyword_filter(clean_text):
                continue
                
            stats["REGEX_PASSED"] += 1
            post["content"] = clean_text 

            buffer_texts.append(clean_text)
            buffer_posts.append(post)

            # 打包送入 GPU
            if len(buffer_texts) >= BATCH_SIZE:
                process_batch(buffer_texts, buffer_posts, out_file, stats)
                buffer_texts.clear()
                buffer_posts.clear()

        # 处理残余数据
        if buffer_texts:
            process_batch(buffer_texts, buffer_posts, out_file, stats)

    return stats

# ============================================================
# 6. 主入口
# ============================================================
def main():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".json") and "livefeeds" in f])
    
    for fname in files:
        print(f"\n📂 Processing {fname}")
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname.replace(".json", ".jsonl")) 
        
        # 每次运行前清理同名旧文件
        if os.path.exists(out_path): os.remove(out_path) 

        stats = process_file(in_path, out_path)

        print(f"📊 Summary for {fname}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

