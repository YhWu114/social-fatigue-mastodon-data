"""Keyword candidate selection and current Qwen3 screening on local input text."""
import argparse
import json
from pathlib import Path
import re
import urllib.request
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
TYPES = {'physical_fatigue', 'mental_cognitive_fatigue', 'emotional_fatigue',
         'digital_social_fatigue', 'existential_resignation'}


def normalize(raw):
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip()
    start = raw.find('{')
    if start < 0:
        raise ValueError('no_json_object')
    obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    if type(obj.get('is_fatigue')) is not bool:
        raise ValueError('non_boolean_fatigue_decision')
    categories = obj.get('categories', {})
    if not isinstance(categories, (dict, list)):
        raise ValueError('invalid_categories')
    types = sorted({x for x in categories if isinstance(x,str)} & TYPES)
    return dict(retained=obj['is_fatigue'] and bool(types), fatigue_types=types)


def candidate(text, language):
    if language == 'en':
        words = json.loads((ROOT/'lexicon/english_broad_filter.json').read_text())
        return any(w in text.lower() for w in words)
    if language not in {'zh', 'zh-CN', 'zh-TW', 'zh-HK', 'ja', 'de', 'fr', 'es', 'nl', 'pt'}:
        return False
    taxonomy = json.loads((ROOT/'lexicon/fatigue_taxonomy_multilingual.json').read_text())
    cjk, alpha = [], []
    for words in taxonomy.values():
        for word in words:
            (cjk if re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', word) else alpha).append(re.escape(word.lower()))
    patterns = ([r'\b(?:'+'|'.join(alpha)+r')\b'] if alpha else []) + (['(?:'+'|'.join(cjk)+')'] if cjk else [])
    return bool(re.search('|'.join(patterns), text, re.I))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True,help='JSONL: id, language, text; cleaned text supplied by researcher')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:8000/v1')
    p.add_argument('--model',default='Qwen/Qwen3-8B')
    p.add_argument('--all-candidates',action='store_true',help='Input has already passed keyword retrieval')
    args=p.parse_args()
    if urlparse(args.endpoint).hostname not in {'localhost','127.0.0.1','::1'}:
        p.error('Use a local endpoint to avoid sending post text externally')
    spec=json.loads((ROOT/'prompts/screening.json').read_text())
    template=json.loads((ROOT/'prompts/user_template.json').read_text())['template']
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with args.input.open() as source, args.output.open('x') as dest:
        for line in source:
            post=json.loads(line)
            if not args.all_candidates and not candidate(post['text'],post['language']):
                result=dict(id=post['id'],status='keyword_excluded')
            else:
                request=dict(model=args.model,temperature=spec['temperature'],max_tokens=spec['max_tokens'],
                    seed=spec['seed'],chat_template_kwargs={'enable_thinking':False},
                    messages=[dict(role='system',content=spec['prompt']),dict(role='user',content=template.format(text=post['text']))])
                try:
                    req=urllib.request.Request(args.endpoint+'/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
                    with opener.open(req,timeout=600) as response: answer=json.load(response)
                    if answer['choices'][0]['finish_reason']=='length':raise ValueError('truncated_output')
                    result=dict(id=post['id'],status='ok',**normalize(answer['choices'][0]['message']['content']))
                except Exception as error:
                    result=dict(id=post['id'],status='unresolved',error_type=type(error).__name__)
            dest.write(json.dumps(result)+'\n');dest.flush()


if __name__=='__main__':
    main()
