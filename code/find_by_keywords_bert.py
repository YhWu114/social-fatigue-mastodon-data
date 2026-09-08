"""
Cascading Funnel Pipeline for Social Fatigue (English Baseline)
Model: Qwen3-8B-Instruct via local vLLM (OpenAI API)
Features: Ultra-Robust JSON Parsing, Concurrent API calls, Regex Pre-filtering, Breakpoint Resume
"""

import os
import json
import ijson
import re
import warnings
import concurrent.futures
from tqdm import tqdm

# 屏蔽 BeautifulSoup 烦人的 URL 解析警告
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

from openai import OpenAI

# ============================================================
# 新增: 进度管理类 (断点续传核心)
# ==========================================
class ProgressTracker:
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.data = self.load()

    def load(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"last_scanned_index": 0}

    def save(self, scanned_index):
        with open(self.checkpoint_file, 'w') as f:
            json.dump({"last_scanned_index": scanned_index}, f)

# ============================================================
# 1. 路径、API & 并发参数
# ============================================================
INPUT_DIR = "/home/user/social-fatigue/mastodonposts/backup"
OUTPUT_DIR = "/home/user/social-fatigue/results/mastodon_qwen_8b"

BATCH_SIZE = 128          # 每次积攒的数据量
MAX_CONCURRENT_REQ = 64   # 并发请求数 (可根据 4090 负载调至 128)
MIN_TEXT_LEN = 30         # 去除链接后的纯文本最小字数

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 指向本地 vLLM 的 8000 端口
client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)
MODEL_NAME = "qwen3-8B-instruct" 

# ============================================================
# 2. 垃圾信息过滤 (Metadata & Bot Shields)
# ============================================================
SPAM_KEYWORDS = ["arxiv.org", "utm_source", "dlvr.it", "#bot", "_bot", "bot_toot", "rss", "news roundup", "podcast"]

def is_spam(text: str) -> bool:
    return any(spam in text.lower() for spam in SPAM_KEYWORDS)

# ============================================================
# 3. 扩充版关键词海选 (High-Recall CPU Regex)
# ============================================================
TAXONOMY = {
    "physical_fatigue": ["tired", "exhausted", "sleepy", "nap", "drained", "weary", "sluggish", "lethargic", "wiped out", "beat", "dead on my feet", "running on empty", "fatigued"],
    "mental_cognitive_fatigue": ["mental fatigue", "brain fog", "focus", "overstimulated", "brain fried", "cannot concentrate", "can't concentrate", "scattered", "mind is mush", "overwhelmed"],
    "emotional_fatigue": ["emotionally drained", "burnout", "burnt out", "anxious", "stressed", "fed up", "at my wits end", "can't take it anymore", "emotionally spent"],
    "digital_social_fatigue": ["social media", "doomscroll", "doomscrolling", "notification", "digital detox", "offline", "screen time", "unplug", "chronically online", "log off", "information overload"],
    "existential_resignation": ["giving up", "what's the point", "done", "pointless", "hopeless", "apathy", "whatever", "can't be bothered", "why bother"]
}

BASE_KEYWORDS = set()
for phrases in TAXONOMY.values():
    BASE_KEYWORDS.update([p.lower() for p in phrases])

regex_pattern = r'\b(?:' + '|'.join(map(re.escape, BASE_KEYWORDS)) + r')\b'
keyword_filter_regex = re.compile(regex_pattern, re.IGNORECASE)

def passes_keyword_filter(text: str) -> bool:
    return bool(keyword_filter_regex.search(text))

# ============================================================
# 4. vLLM (Qwen3-8B-Instruct) 终审与容错解析
# ============================================================
SYSTEM_PROMPT = """You are an expert sociologist classifying social media posts.
Determine if the text expresses genuine "Social, Mental, or Physical Fatigue".
Strictly exclude:
1. Sarcasm, rants, or complaints about daily inconveniences (e.g., traffic, chores, people in the way).
2. Objective news, song lyrics, or automated updates.

Categories available: physical_fatigue, mental_cognitive_fatigue, emotional_fatigue, digital_social_fatigue, existential_resignation.

You must output ONLY valid JSON. Format:
{
  "is_fatigue": true or false,
  "categories": {"category_name": confidence_score_between_0_and_1},
  "reason": "Brief reason for your decision"
}"""

def get_llm_judgment(text: str):
    """单次 API 请求，包含超时保护"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Text: \"{text}\"\nAnalyze this and output ONLY valid JSON."}
            ],
            temperature=0.1, # 极低温度保证稳定性
            max_tokens=800,  # 给足 Token 应对 <think>
            timeout=15       # 防止个别请求死锁卡住进度
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None

def process_batch(texts, posts, out_file, stats):
    # 多线程并发狂轰本地 API
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQ) as executor:
        results = list(executor.map(get_llm_judgment, texts))

    for post, generated_text in zip(posts, results):
        if not generated_text:
            stats["API_ERRORS"] = stats.get("API_ERRORS", 0) + 1
            continue
            
        try:
            # --- 🛡️ 第一层装甲：暴力拆除 <think> 和 Markdown ---
            text_no_think = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
            if "<think>" in text_no_think:
                text_no_think = text_no_think.split("</think>")[-1].strip()

            clean_json_str = text_no_think.replace("```json", "").replace("```", "").strip()
            
            # 精准定位 JSON 大括号
            start_idx = clean_json_str.find('{')
            end_idx = clean_json_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                clean_json_str = clean_json_str[start_idx:end_idx+1]
            else:
                raise ValueError("No JSON braces found")
                
            result = json.loads(clean_json_str)
            
            # --- 🛡️ 第二层装甲：Schema 漂移智能纠错 ---
            if result.get("is_fatigue") is True and result.get("categories"):
                raw_categories = result["categories"]
                llm_reason = str(result.get("reason", ""))
                
                valid_categories = {}
                allowed_keys = ["physical_fatigue", "mental_cognitive_fatigue", "emotional_fatigue", "digital_social_fatigue", "existential_resignation"]
                
                # 情况 1：标准字典
                if isinstance(raw_categories, dict):
                    # 修复错位的 reason
                    if "reason" in raw_categories:
                        if not llm_reason: llm_reason = str(raw_categories["reason"])
                        del raw_categories["reason"]
                        
                    for k, v in raw_categories.items():
                        if k in allowed_keys:
                            try:
                                weight = float(v)
                                valid_categories[k] = min(max(weight, 0.0), 1.0) # 钳制在 0-1 之间
                            except:
                                valid_categories[k] = 0.90 # 强行兜底
                                
                # 情况 2：叛逆的列表格式
                elif isinstance(raw_categories, list):
                    for item in raw_categories:
                        if isinstance(item, str) and item in allowed_keys:
                            valid_categories[item] = 0.90
                        elif isinstance(item, dict):
                            for k, v in item.items():
                                if k in allowed_keys:
                                    try:
                                        valid_categories[k] = float(v)
                                    except:
                                        valid_categories[k] = 0.90
                
                # --- 落盘 ---
                if valid_categories:
                    stats["TOTAL_POSTS_MATCHED"] += 1
                    post["fatigue_categories"] = list(valid_categories.keys())
                    post["fatigue_weights"] = valid_categories
                    post["llm_reason"] = llm_reason
                    
                    out_file.write(json.dumps(post, ensure_ascii=False) + "\n")
                    out_file.flush() # 强制写入磁盘，防崩溃
                
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
    # 断点续传初始化
    checkpoint_path = output_path + ".checkpoint"
    tracker = ProgressTracker(checkpoint_path)
    resume_index = tracker.data.get("last_scanned_index", 0)

    stats = {"TOTAL_RECORDS_SCANNED": 0, "PASSED_KEYWORD_FILTER": 0, "TOTAL_POSTS_MATCHED": 0}
    buffer_texts, buffer_posts = [], []

    if resume_index > 0:
        print(f"\n⏩ 发现历史进度！正在快速跳过前 {resume_index} 条记录，请稍候...")
        stats["TOTAL_RECORDS_SCANNED"] = resume_index # 让统计数据直接从断点开始

    try:
        with open(output_path, "a", encoding="utf-8") as out_file:
            # 记录当前迭代的索引，用于跳过
            current_index = 0
            
            for post in tqdm(stream_json_objects(input_path), desc=os.path.basename(input_path)):
                current_index += 1
                
                # 【核心】：快速跳过已处理的记录
                if current_index <= resume_index:
                    continue

                stats["TOTAL_RECORDS_SCANNED"] += 1
                
                # Metadata 拦截
                if post.get("language") != "en": continue
                if post.get("reblog"): continue
                
                account_info = post.get("account", {})
                if account_info and (account_info.get("bot") is True or "bot" in account_info.get("acct", "").lower()):
                    continue

                app_info = post.get("application") or {}
                if any(x in app_info.get("name", "").lower() for x in ["script", "ifttt", "zapier", "bot", "crossposter", "buffer", "rss"]):
                    continue

                tags = [t.get("name", "").lower() for t in post.get("tags", [])]
                if "nowplaying" in tags or "listeningto" in tags:
                    continue

                raw_html = post.get("content", "")
                if not raw_html: continue

                clean_text = BeautifulSoup(raw_html, "html.parser").get_text().replace('\n', ' ').strip()
                text_without_urls = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', clean_text)
                
                if len(text_without_urls.strip()) < MIN_TEXT_LEN: continue
                if is_spam(clean_text): continue

                # CPU 海选
                if not passes_keyword_filter(clean_text):
                    # 为了防止大片连续不匹配导致进度丢失，每 10000 条没匹配的也存一次断点
                    if stats["TOTAL_RECORDS_SCANNED"] % 10000 == 0:
                        tracker.save(stats["TOTAL_RECORDS_SCANNED"])
                    continue
                    
                stats["PASSED_KEYWORD_FILTER"] += 1
                post["content"] = clean_text 

                buffer_texts.append(clean_text)
                buffer_posts.append(post)

                # LLM 并发终审
                if len(buffer_texts) >= BATCH_SIZE:
                    process_batch(buffer_texts, buffer_posts, out_file, stats)
                    buffer_texts.clear()
                    buffer_posts.clear()
                    # 一批 (128条) 成功处理并写入后，安全保存进度
                    tracker.save(stats["TOTAL_RECORDS_SCANNED"])

            # 处理尾部
            if buffer_texts:
                process_batch(buffer_texts, buffer_posts, out_file, stats)
                tracker.save(stats["TOTAL_RECORDS_SCANNED"])
                
    except KeyboardInterrupt:
        print(f"\n\n🛑 检测到手动中断 (Ctrl+C)。")
        # 安全退回：扣除 buffer 中没被处理的数据，保证下一次能重新处理它们
        safe_index = stats["TOTAL_RECORDS_SCANNED"] - len(buffer_texts)
        tracker.save(safe_index)
        print(f"✅ 进度已安全保存在第 {safe_index} 条，下次运行将直接从此处继续！")
        return stats

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
        
        # ⚠️ 这里是以前导致你数据丢失的罪魁祸首，已经被我注销掉了
        # if os.path.exists(out_path): os.remove(out_path) 

        stats = process_file(in_path, out_path)

        print(f"📊 Summary for {fname} (This run):")
        for k, v in stats.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()