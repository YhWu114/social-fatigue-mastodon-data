"""Portable local inference adapter; preserves production generation and validation settings."""
from pathlib import Path
import hashlib
import copy
import json
import os
import re
import signal
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request

HERE=Path(__file__).resolve().parent
WORK=Path(os.environ.get('FATIGUESCAN_WORK', './private_work'))
MODEL=os.environ.get('FATIGUESCAN_MODEL', 'Qwen/Qwen3-8B')
ALIAS=MODEL
ENDPOINTS=[os.environ.get('FATIGUESCAN_ENDPOINT', 'http://127.0.0.1:8000/v1')]
SOURCE=WORK

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def key(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def save(name,obj):
    p=WORK/name;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def posts():
    with (SOURCE/'posts.jsonl').open() as stream:return [json.loads(s) for s in stream if s.strip()]
def segments(text,maximum=650):
    """Cover the whole original once; never translate, truncate or lose spaces."""
    cuts={0,len(text)}
    for m in re.finditer(r'\n+|[。！？]+|[.!?]+(?=\s|$)',text):cuts.add(m.end())
    ordered=sorted(cuts);parts=[]
    for a,b in zip(ordered,ordered[1:]):
        while a<b:
            end=min(a+maximum,b)
            if end<b:
                space=text.rfind(' ',a+maximum//2,end)
                if space>a:end=space+1
            parts.append(dict(id=f'S{len(parts)+1:03d}',start=a,end=end,text=text[a:end]));a=end
    assert ''.join(p['text'] for p in parts)==text
    return parts
def evidence(parts,ids):
    if not isinstance(ids,list) or not ids or any(not isinstance(x,str) for x in ids):raise ValueError('missing_sentence_ids')
    lookup={p['id']:p for p in parts}
    if not set(ids)<=set(lookup):raise ValueError('unknown_sentence_id')
    return [dict(sentence_id=i,text=lookup[i]['text'],start=lookup[i]['start'],end=lookup[i]['end']) for i in dict.fromkeys(ids)]
PLACEHOLDERS=('一句简短理由','一句具体理由','具体讨论内容','主项连续原文','支持判断的原文','提供的完整名称','真实内容概括','不是A，而是B','真实的具体理由')
def meaningful(value,minimum=3):
    if not isinstance(value,str) or len(value.strip())<minimum:return False
    s=value.strip()
    return not any(s==p or s.startswith(p+'，') or s.startswith(p+'。') for p in PLACEHOLDERS)
def db():
    WORK.mkdir(parents=True, exist_ok=True)
    c=sqlite3.connect(WORK/'inference.sqlite3',timeout=120)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('CREATE TABLE IF NOT EXISTS calls (request_key TEXT PRIMARY KEY,stage TEXT,payload TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS failed_calls (request_key TEXT,stage TEXT,error TEXT,payload TEXT)')
    return c
def infer(endpoint,stage,prompt,payload,seed=20261017,budget=1200):
    from urllib.parse import urlparse
    if urlparse(endpoint).hostname not in {'localhost', '127.0.0.1', '::1'}:raise ValueError('Only local endpoints allowed; no post text is sent to external services')
    from output_schema import schema_for
    from jsonschema import validate
    schema=schema_for(stage,payload)
    thinking=stage in {'proposal_names','independent_names','resolve_names','specific_content_names'}
    if payload.get('format_repair_attempt',0)>=2 or 'truncated_output' in payload.get('format_error',''):
        thinking=False
    spec=dict(model=ALIAS,messages=[dict(role='system',content=prompt),dict(role='user',content=json.dumps(payload,ensure_ascii=False))],
              temperature=.6 if thinking else .7,top_p=.95 if thinking else .8,top_k=20,seed=seed,max_tokens=max(4096,budget) if thinking else max(2048,budget),
              chat_template_kwargs={'enable_thinking':thinking},response_format={'type':'json_schema','json_schema':{'name':stage,'strict':True,'schema':schema}})
    request_key=key(spec);c=db();found=c.execute('SELECT payload FROM calls WHERE request_key=?',(request_key,)).fetchone()
    if not found:
        # Reuse successful outputs from the identical task with the stricter
        # former string-length grammar. Never reuse a failed request or label.
        legacy=copy.deepcopy(spec)
        def stricter(node):
            if isinstance(node,dict):
                if node.get('type')=='string' and 'minLength' in node:node['maxLength']=240
                for value in node.values():stricter(value)
            elif isinstance(node,list):
                for value in node:stricter(value)
        stricter(legacy['response_format']['json_schema']['schema'])
        found=c.execute('SELECT payload FROM calls WHERE request_key=?',(key(legacy),)).fetchone()
    c.close()
    if found:return json.loads(found[0])
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));last=None
    for attempt in range(2):
        raw=None
        try:
            request=urllib.request.Request(endpoint+'/chat/completions',data=json.dumps(spec).encode(),headers={'Content-Type':'application/json'})
            start=time.monotonic()
            with opener.open(request,timeout=600) as response:raw=json.load(response)
            choice=raw['choices'][0]
            if choice['finish_reason']=='length':raise ValueError('truncated_output')
            content=choice['message'].get('content','');begin=content.find('{')
            if begin<0:raise ValueError('no_json')
            obj=json.JSONDecoder().raw_decode(content[begin:])[0]
            if not isinstance(obj,dict):raise ValueError('not_object')
            validate(instance=obj,schema=schema)
            result=dict(request_key=request_key,output=obj,raw=content,usage=raw.get('usage'),seconds=round(time.monotonic()-start,3),stage=stage)
            c=db();c.execute('INSERT OR REPLACE INTO calls VALUES (?,?,?)',(request_key,stage,json.dumps(result,ensure_ascii=False)));c.commit();c.close()
            return result
        except Exception as e:
            last=e
            if isinstance(e,urllib.error.HTTPError):last=RuntimeError(str(e)+e.read().decode(errors='replace')[:500])
            c=db();c.execute('INSERT INTO failed_calls VALUES (?,?,?,?)',(request_key,stage,str(last),json.dumps(raw,ensure_ascii=False)));c.commit();c.close()
            if isinstance(e,ValueError):break
    raise RuntimeError(str(last))
