import pandas as pd
import json
import glob
import os
import re
from collections import Counter
import time
import numpy as np


# Attempt to import sentence_transformers, handle case if not installed
try:
    from sentence_transformers import SentenceTransformer, util
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

# ==========================================
# 1. Configuration & Taxonomy
# ==========================================

# --- 核心配置开关 ---
# 如果设为 50000，程序在扫描了 50000 条原始数据后就会停止（无论匹配到了多少条）。
# 如果想跑全量数据，请将其设为 0 或 None。
MAX_RECORDS_TO_PROCESS = None  

TAXONOMY = {
    "physical_fatigue": [
        "tired", "so tired", "very tired", "exhausted", "exhaustion",
        "sleepy", "need sleep", "sleep deprived", "sleep-deprived",
        "no energy", "low energy", "worn out", "run down", "spent",
        "need a nap", "nap", "body hurts"
    ],
    "mental_cognitive_fatigue": [
        "mentally tired", "mental fatigue", "mental exhaustion",
        "brain fog", "foggy", "fried", "brain fried",
        "can't focus", "cant focus", "can't concentrate", "cant concentrate",
        "can't think", "cant think", "head is spinning", "overstimulated"
    ],
    "emotional_fatigue": [
        "emotionally drained", "emotionally exhausted", "emotional exhaustion",
        "drained", "draining", "overwhelmed", "overwhelming",
        "burned out", "burnt out", "burnout", "burning out",
        "anxious", "anxiety", "stressed", "stress", "on edge", "frustrated"
    ],
    "digital_social_fatigue": [
        "tired of social media", "social media fatigue", "social media burnout",
        "digital fatigue", "digital burnout",
        "screen fatigue", "screen time", "too much screen time",
        "doomscroll", "doomscrolling",
        "notifications", "too many notifications", "notification overload",
        "always online", "chronically online", "terminally online",
        "logging off", "log off", "going offline", "disconnect", "unplug",
        "digital detox", "taking a break", "need a break"
    ],
    "existential_resignation": [
        "i'm done", "im done", "i’m done",
        "so done", "done with this", "had enough",
        "can't deal", "cant deal",
        "can't cope", "cant cope",
        "what's the point", "whats the point",
        "i give up", "giving up",
        "this is too much", "at my limit", "pushed to my limit"
    ]
}

# High-recall filter
TIRED_FILTER = [
    "tired", "exhausted", "drained", "burnout", "burned out", "burnt out",
    "overwhelmed", "fatigue", "sleep deprived", "brain fog", "can't focus", "cant focus",
    "logging off", "going offline", "digital detox", "doomscroll"
]

# ==========================================
# 2. Classifier Implementations
# ==========================================

class KeywordClassifier:
    def __init__(self, taxonomy, broad_filter):
        self.taxonomy = taxonomy
        self.broad_filter = broad_filter

    def is_broadly_relevant(self, text):
        if not isinstance(text, str):
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.broad_filter)

    def categorize(self, text):
        if not isinstance(text, str):
            return []
        
        text_lower = text.lower()
        matched_categories = []
        
        for category, keywords in self.taxonomy.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matched_categories.append(category)
                    break 
        return matched_categories


class BertSemanticClassifier:
    def __init__(self, taxonomy, model_name='all-MiniLM-L6-v2', threshold=0.45):
        if not BERT_AVAILABLE:
            raise ImportError("Please install sentence-transformers: `pip install sentence-transformers`")
        
        print(f"Loading BERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.category_embeddings = {}
        
        print("Pre-computing category embeddings...")
        for category, keywords in taxonomy.items():
            category_desc = ", ".join(keywords) 
            self.category_embeddings[category] = self.model.encode(category_desc, convert_to_tensor=True)
            
    def categorize(self, text):
        if not isinstance(text, str) or len(text.strip()) < 5:
            return []

        text_embedding = self.model.encode(text, convert_to_tensor=True)
        matched_categories = []
        
        for category, cat_embedding in self.category_embeddings.items():
            score = util.cos_sim(text_embedding, cat_embedding).item()
            if score >= self.threshold:
                matched_categories.append(category)
                
        return matched_categories

# ==========================================
# 3. Utilities
# ==========================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def clean_html(raw_html):
    """
    Removes HTML tags (e.g., <p>, <br>) from content commonly found in Mastodon/JSON exports.
    """
    if not isinstance(raw_html, str):
        return ""
    clean_r = re.compile('<.*?>')
    text = re.sub(clean_r, ' ', raw_html) 
    return text.strip()

# ==========================================
# 4. Processing Engines
# ==========================================

def process_parquet_files(input_dir, output_file, classifier_method, limit=None):
    if classifier_method == "bert":
        print("\n[Parquet Mode] Initializing BERT Semantic Classifier...")
        classifier = BertSemanticClassifier(TAXONOMY)
    else:
        print("\n[Parquet Mode] Initializing Keyword Classifier...")
        classifier = KeywordClassifier(TAXONOMY, TIRED_FILTER)
    
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    _run_processing(parquet_files, output_file, classifier, classifier_method, file_type='parquet', limit=limit)


def process_json_files(input_dir, id, output_file, classifier_method, limit=None):
    if classifier_method == "bert":
        print("\n[JSON Mode] Initializing BERT Semantic Classifier...")
        classifier = BertSemanticClassifier(TAXONOMY)
    else:
        print("\n[JSON Mode] Initializing Keyword Classifier...")
        classifier = KeywordClassifier(TAXONOMY, TIRED_FILTER)
    
    json_files = glob.glob(os.path.join(input_dir, id))
    _run_processing(json_files, output_file, classifier, classifier_method, file_type='json', limit=limit)


def _run_processing(file_list, output_file, classifier, method, file_type, limit=None):
    """ Shared internal processing loop with Limit Logic """
    total_categorized = 0
    total_scanned = 0  # Counter for total rows read
    category_counts = Counter()
    start_time = time.time()
    
    stop_processing = False # Flag to break outer loop

    if not file_list:
        print(f"No .{file_type} files found.")
        return

    print(f"Found {len(file_list)} {file_type} files. Starting...")
    if limit:
        print(f"⚠️  Limit active: Will stop after scanning {limit} records.")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for file_path in file_list:
            if stop_processing:
                break
                
            print(f"Processing file: {os.path.basename(file_path)}...")
            
            try:
                # --- LOAD DATA ---
                records = []
                if file_type == 'parquet':
                    df = pd.read_parquet(file_path)
                    records = df.to_dict(orient='records')
                elif file_type == 'json':
                    with open(file_path, 'r', encoding='utf-8') as f_in:
                        try:
                            data = json.load(f_in)
                            if isinstance(data, list):
                                records = data
                            elif isinstance(data, dict):
                                records = [data]
                        except json.JSONDecodeError:
                            f_in.seek(0)
                            # Simple line reading
                            records = [json.loads(line) for line in f_in]

                # --- PROCESS RECORDS ---
                for row in records:
                    # 1. Check Limit
                    if limit and total_scanned >= limit:
                        print(f"\n[Limit Reached] Stopped after scanning {total_scanned} records.")
                        stop_processing = True
                        break

                    total_scanned += 1
                    
                    # 2. Extract Text
                    if file_type == 'parquet':
                        raw_text = row.get('text', '')
                    else:
                        # JSON/Mastodon content often has HTML
                        raw_content = row.get('content', '')
                        raw_text = clean_html(raw_content)

                    # 3. Filter & Categorize
                    if method == "keyword":
                        if not classifier.is_broadly_relevant(raw_text):
                            continue
                    
                    categories = classifier.categorize(raw_text)
                    
                    if categories:
                        total_categorized += 1
                        category_counts.update(categories)
                        
                        row['cleaned_text_for_analysis'] = raw_text 
                        row['matched_categories'] = categories
                        row['classification_method'] = method
                        
                        if 'created_at' in row and not isinstance(row['created_at'], str):
                             row['created_at'] = str(row['created_at'])

                        f_out.write(json.dumps(row, cls=NumpyEncoder, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # --- STATS ---
    elapsed = time.time() - start_time
    print(f"\n{'='*30}")
    print(f"Processing Complete. Time: {elapsed:.2f}s")
    print(f"Total Records Scanned: {total_scanned}")
    print(f"Matched Posts Found:   {total_categorized}")
    print(f"{'='*30}\n")
    
    # 1. 创建总览数据的 DataFrame (放在 CSV 最顶部)
    summary_data = {
        'TOTAL_RECORDS_SCANNED': total_scanned,
        'TOTAL_POSTS_MATCHED': total_categorized
    }
    summary_df = pd.DataFrame.from_dict(summary_data, orient='index', columns=['count'])
    summary_df.index.name = 'category'
    
    # 2. 创建分类统计的 DataFrame (如果有的匹配的话)
    final_stats_df = summary_df # 默认为 summary
    
    if len(category_counts) > 0:
        cats_df = pd.DataFrame.from_dict(category_counts, orient='index', columns=['count'])
        cats_df.index.name = 'category'
        cats_df = cats_df.sort_values(by='count', ascending=False)
        
        # 3. 合并：总览在上面，分类细节在下面
        final_stats_df = pd.concat([summary_df, cats_df])
        
        # 打印一下细节到控制台方便查看
        print("=== Category Breakdown ===")
        print(cats_df)

    # 4. 保存 CSV
    base_name = os.path.splitext(output_file)[0]
    stats_filename = f"{base_name}_stats.csv"
    
    try:
        final_stats_df.to_csv(stats_filename)
        print(f"\n✅ Statistics (including totals) saved to: {stats_filename}")
    except Exception as e:
        print(f"\n❌ Error saving statistics CSV: {e}")

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    # --- CONFIGURATION ---
    
    DATA_SOURCE_TYPE = "json"  # "parquet" or "json"
    PROCESSING_MODE = "keyword" # "keyword" or "bert"
    INPUT_DIRECTORY = "/home/user/social-fatigue/mastodonposts/backup" 
    ID = "livefeeds_5.json"
    
    # 结果文件名会加上 _limit 标识，方便区分
    if MAX_RECORDS_TO_PROCESS:
        limit_suffix = f"_limit{MAX_RECORDS_TO_PROCESS}"
    else:
        limit_suffix = ""

    OUTPUT_FILENAME = f"results/classified_posts_{DATA_SOURCE_TYPE}_{PROCESSING_MODE}{limit_suffix}_{ID}.jsonl"

    # ---------------------

    if not os.path.exists("results"):
        os.makedirs("results")

    if not os.path.exists(INPUT_DIRECTORY):
        print(f"Error: Directory '{INPUT_DIRECTORY}' does not exist.")
    else:
        if DATA_SOURCE_TYPE == "parquet":
            process_parquet_files(INPUT_DIRECTORY, OUTPUT_FILENAME, PROCESSING_MODE, limit=MAX_RECORDS_TO_PROCESS)
        elif DATA_SOURCE_TYPE == "json":
            process_json_files(INPUT_DIRECTORY, ID, OUTPUT_FILENAME, PROCESSING_MODE, limit=MAX_RECORDS_TO_PROCESS)
        else:
            print("Invalid DATA_SOURCE_TYPE. Please set to 'parquet' or 'json'.")