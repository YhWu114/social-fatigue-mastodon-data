"""Evidence-grounded topic decisions; no legacy labels override a new reading."""
from collections import Counter
import copy
import json
import numpy as np
import prompts
import runtime as r

from examples import Examples

def required_text(obj,name,minimum=3):
    if not r.meaningful(obj.get(name),minimum):raise ValueError('missing_or_placeholder_'+name)
    return obj[name].strip()
def validate_proposal(obj,parts,definitions):
    summary=required_text(obj,'summary',5);raw=obj.get('topics');topics=[];merged={}
    if not isinstance(raw,list):raise ValueError('topics_not_list')
    names={d.get('name',tid):tid for tid,d in definitions.items()}
    for t in raw:
        if not isinstance(t,dict):raise ValueError('unknown_topic')
        tid=names.get(t.get('topic_name')) if 'topic_name' in t else t.get('id')
        if tid not in definitions:raise ValueError('unknown_topic_name: '+str(t.get('topic_name',t.get('id')))+'; valid names: '+str(list(names)))
        ev=r.evidence(parts,t.get('sids'));focus=required_text(t,'focus',2)
        if tid in merged:
            previous=merged[tid];known={q['sentence_id'] for q in previous['evidence']}
            previous['evidence'] += [q for q in ev if q['sentence_id'] not in known]
            if focus not in previous['focus']:previous['focus']+='；'+focus
        else:
            item=dict(id=tid,focus=focus,evidence=ev);topics.append(item);merged[tid]=item
    context=obj.get('context_reason','');context_ev=[]
    if not topics:
        context=required_text(obj,'context_reason',5)
        context_ev=r.evidence(parts,obj['context_sids']) if obj.get('context_sids') else []
    return dict(summary_zh=summary,topics=topics,context_reason=context,context_evidence=context_ev)
def validate_decision(obj,parts):
    state=obj.get('state')
    if state not in {'yes','no','uncertain'}:raise ValueError('invalid_decision_state')
    reason=required_text(obj,'reason',5)
    ev=r.evidence(parts,obj['sids']) if obj.get('sids') else []
    if state=='yes' and not ev:raise ValueError('missing_positive_topic_evidence')
    focus=required_text(obj,'focus',2) if state=='yes' else obj.get('focus','')
    return dict(state=state,reason=reason,evidence=ev,focus=focus)
def validate_fine(obj,parts,allowed):
    state=obj.get('state')
    if state not in {'matched','broad_mismatch','outside_subtopics','insufficient_detail','uncertain'}:raise ValueError('invalid_fine_state')
    activity=required_text(obj,'activity',2);reason=required_text(obj,'reason',5);ev=r.evidence(parts,obj.get('sids'))
    raw=obj.get('choices');choices=[];seen=set()
    if not isinstance(raw,list):raise ValueError('choices_not_list')
    for x in raw:
        if not isinstance(x,dict):raise ValueError('unknown_specific_content')
        sid=allowed.get(x.get('name')) if isinstance(allowed,dict) and 'name' in x else x.get('id')
        if sid not in (set(allowed.values()) if isinstance(allowed,dict) else allowed):raise ValueError('unknown_specific_content')
        evidence=r.evidence(parts,x.get('sids'))
        if sid not in seen:choices.append(dict(id=sid,evidence=evidence));seen.add(sid)
    if (state=='matched')!=bool(choices):raise ValueError('state_choices_conflict')
    suggestion=required_text(obj,'suggestion',4) if state=='outside_subtopics' else obj.get('suggestion','')
    return dict(state=state,activity=activity,reason=reason,evidence=ev,choices=choices,suggestion=suggestion)
def validate_experience(obj,parts):
    fatigue=obj.get('fatigue');affect=obj.get('affect')
    if fatigue not in {'self_clear','self_implicit','other_or_quoted','no_fatigue_expression','uncertain'}:raise ValueError('invalid_fatigue')
    if affect not in {'positive','negative','mixed','neutral','unclear'}:raise ValueError('invalid_affect')
    ev=r.evidence(parts,obj['fatigue_sids']) if obj.get('fatigue_sids') else []
    if fatigue in {'self_clear','self_implicit','other_or_quoted'} and not ev:raise ValueError('missing_fatigue_evidence')
    positive=r.evidence(parts,obj['positive_sids']) if obj.get('positive_sids') else []
    negative=r.evidence(parts,obj['negative_sids']) if obj.get('negative_sids') else []
    if affect in {'positive','mixed'} and not positive:raise ValueError('missing_positive_evidence')
    if affect in {'negative','mixed'} and not negative:raise ValueError('missing_negative_evidence')
    if affect=='neutral' and (positive or negative):raise ValueError('neutral_with_valence_evidence')
    return dict(fatigue=fatigue,affect=affect,fatigue_evidence=ev,positive_evidence=positive,negative_evidence=negative,reason=required_text(obj,'reason',5))

def call(endpoint,stage,prompt,payload,validator,seed,budget):
    attempts=[]
    for attempt in range(5):
        current=prompt;request=copy.deepcopy(payload)
        if attempt:
            current+='\n上一输出未通过机械校验。保持原文判断任务不变，修复这些错误：'+str(attempts[-1]['error'])+'。输出实际答案，不复制格式说明。'
            request['format_repair_attempt']=attempt
            request['format_error']=attempts[-1]['error']
            request['technical_retry_round']=getattr(r,'REPAIR_ROUND',0)
        try:
            result=r.infer(endpoint,stage,current,request,seed,max(4096,budget)+1024*attempt)
            attempts.append(dict(request_key=result['request_key']))
            item=validator(result['output'])
            return dict(status='ok',item=item,attempts=attempts,request_key=result['request_key'])
        except Exception as e:
            if attempts and 'error' not in attempts[-1]:attempts[-1]['error']=str(e)[:500]
            else:attempts.append(dict(error=str(e)[:500]))
            if stage in {'proposal_names','independent_names'} and str(e)=='missing_or_placeholder_context_reason':
                fixed=call(endpoint,'proposal_explained',prompt+'\n必须填写context_reason，结合原帖解释有没有具体情境；无具体主题时说明原帖实际在表达什么。',payload,validator,seed,2048)
                fixed['attempts']=attempts+fixed['attempts'];fixed['format_recovery']='required_context_explanation'
                return fixed
            if stage=='experience' and str(e)=='missing_fatigue_evidence':
                fixed=call(endpoint,'experience_grounded',prompt,payload,validator,seed,2048)
                fixed['attempts']=attempts+fixed['attempts'];fixed['format_recovery']='conditional_fatigue_evidence'
                return fixed
            if stage=='resolve_names' and str(e) in {'selected_uncertain_conflict','missing_or_placeholder_context_reason'}:
                # A flat decision per candidate avoids contradictory parallel
                # lists and the nested-object recovery's malformed outputs.
                recovered=[];topics=[];uncertain=[];reasons=[]
                parts=r.segments(payload['original_post'])
                for topic in payload['topics']:
                    one_payload=dict(payload,topics=[topic])
                    fixed=call(endpoint,'resolve_single',prompts.RESOLVE,one_payload,
                        lambda value:validate_decision(value,parts),seed,2048)
                    recovered+=fixed['attempts']
                    if fixed['status']!='ok':
                        return dict(status='technical_failed',item=None,attempts=attempts+recovered,format_recovery='flat_topic_decisions')
                    decision=fixed['item'];name=topic['name']
                    if decision['state']=='yes':
                        topics.append(dict(topic_name=name,sids=[x['sentence_id'] for x in decision['evidence']],focus=decision['focus']))
                    elif decision['state']=='uncertain':uncertain.append(name)
                    reasons.append(name+'：'+decision['reason'])
                item=validator(dict(summary='；'.join(reasons),topics=topics,context_reason='；'.join(reasons),
                    context_sids=[],reviewed_topics=[t['name'] for t in payload['topics']],uncertain_topics=uncertain))
                return dict(status='ok',item=item,attempts=attempts+recovered,format_recovery='flat_topic_decisions')
    return dict(status='technical_failed',item=None,attempts=attempts)

class Engine:
    def __init__(self,source,previous):
        self.source=source;self.previous=previous;self.examples=Examples(source)
        self.definitions={x['id']:x for x in r.read(r.HERE/'broad_topics.json')}
        self.fine={x['topic_id']:x for x in r.read(r.HERE/'specific_content_codebook.json')}
    def classify(self,pos,endpoint,seed=20261017,with_content=True):
        post=self.source[pos];parts=r.segments(post['text']);numbered=[dict(id=x['id'],text=x['text']) for x in parts]
        refs=self.examples.get(post)
        for ref in refs:ref['human_topics']=[self.definitions[t]['name'] for t in ref['human_topics']]
        catalogue=[dict(name=t['name'],definition=t['definition']) for t in self.definitions.values()]
        base_payload=dict(topics=catalogue,original_post=post['text'],target_sentences=numbered)
        proposal=call(endpoint,'proposal_names',prompts.PROPOSE,
            dict(**base_payload,human_reference_examples=refs),
            lambda obj:validate_proposal(obj,parts,self.definitions),seed,1300)
        result=dict(position=pos,text_key=post['text_key'],group=post['group'],seed=seed,
            proposal=proposal,reference_ids=[x['reference_id'] for x in refs],
            topic_decisions=[],specific_content=[],topics=[],unresolved_topics=[],screening_unchanged=True)
        if proposal['status']!='ok':result['topic_status']='technical_failed';return result
        fresh={t['id']:t for t in proposal['item']['topics']}
        legacy={t['id'] for t in self.previous[pos]['item']['topics']}
        second=call(endpoint,'independent_names',prompts.SECOND,base_payload,lambda obj:validate_proposal(obj,parts,self.definitions),seed,1300)
        result['second_reading']=second
        if second['status']!='ok':result['topic_status']='partial_technical_failure';return result
        other={t['id']:t for t in second['item']['topics']}
        agreed=set(fresh)&set(other);disputed=(set(fresh)|set(other)|legacy)-agreed
        result['summary_zh']=proposal['item']['summary_zh']
        def resolve(tids,extra=None):
            selected_defs={tid:self.definitions[tid] for tid in tids};wanted={v['name'] for v in selected_defs.values()}
            def validate(obj):
                item=validate_proposal(obj,parts,selected_defs)
                reviewed=obj.get('reviewed_topics');uncertain=obj.get('uncertain_topics',[])
                if not isinstance(reviewed,list) or set(reviewed)!=wanted:raise ValueError('incomplete_reviewed_topic_names')
                if not isinstance(uncertain,list) or not set(uncertain)<=wanted:raise ValueError('unknown_uncertain_topic')
                chosen={self.definitions[x['id']]['name'] for x in item['topics']}
                if chosen&set(uncertain):raise ValueError('selected_uncertain_conflict')
                item['uncertain_ids']=[tid for tid in tids if self.definitions[tid]['name'] in uncertain]
                return item
            return call(endpoint,'resolve_names',prompts.GLOBAL_RESOLVE,
                dict(topics=[dict(name=t['name'],definition=t['definition']) for t in selected_defs.values()],
                    original_post=post['text'],target_sentences=numbered,**(extra or {})),validate,seed,1400)
        chosen={tid:dict(**other[tid],decision='two_independent_readings_agree') for tid in agreed}
        resolution=resolve(sorted(disputed)) if disputed else None;result['global_resolution']=resolution
        if resolution and resolution['status']=='ok':
            for t in resolution['item']['topics']:chosen[t['id']]=dict(**t,decision='conflict_adjudicated')
            result['unresolved_topics'] += [dict(id=tid,kind='semantic_uncertainty') for tid in resolution['item']['uncertain_ids']]
        elif resolution:result['unresolved_topics'] += [dict(id=tid,kind='technical_failure') for tid in disputed]
        result['topics']=[chosen[tid] for tid in list(fresh)+sorted(set(chosen)-set(fresh)) if tid in chosen]
        result['topic_decisions']=[dict(topic_id=tid,first_selected=tid in fresh,second_selected=tid in other,
            legacy_candidate=tid in legacy,final_selected=tid in chosen,agreement=tid in agreed) for tid in sorted(set(fresh)|set(other)|legacy)]
        result['topic_status']='ok' if not any(t['kind']=='technical_failure' for t in result['unresolved_topics']) else 'partial_technical_failure'
        if with_content:
            for topic in result['topics'].copy():
                tid=topic['id'];book=self.fine[tid]
                fine=call(endpoint,'specific_content_names',prompts.FINE,
                    dict(topic=dict(name=self.definitions[tid]['name'],definition=self.definitions[tid]['definition']),
                         specific_contents=[dict(name=x['name_zh'],definition=x['definition_zh']) for x in book['subtopics']],
                         original_post=post['text'],target_sentences=numbered),
                    lambda obj:validate_fine(obj,parts,{x['name_zh']:x['id'] for x in book['subtopics']}),seed,800)
                record=dict(topic_id=tid,result=fine)
                if fine['status']=='ok' and fine['item']['state']=='broad_mismatch':
                    resolution=resolve([tid],dict(content_check=fine['item']))
                    record['broad_resolution']=resolution
                    if resolution['status']=='ok' and not resolution['item']['topics'] and tid not in resolution['item']['uncertain_ids']:
                        result['topics']=[t for t in result['topics'] if t['id']!=tid];record['broad_rejected_after_content_review']=True
                    else:record['unresolved_content_conflict']=True
                result['specific_content'].append(record)
            result['experience']=call(endpoint,'experience',prompts.EXPERIENCE,dict(target_sentences=numbered),lambda obj:validate_experience(obj,parts),seed,850)
        return result
