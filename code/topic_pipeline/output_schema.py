"""Constrain generation, including names and source IDs, before decoding."""

def obj(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)

def text(minimum=0):
    # XGrammar's maxLength expands the character-count grammar and made mask
    # filling ~60x slower in our local check. The prompt requests concise text;
    # types, mandatory fields, label enums and source IDs remain constrained.
    # This installed XGrammar compiles explicit minLength=0 as exactly one
    # character. Omit the redundant zero bound to allow normal empty/free text.
    return dict(type='string',minLength=minimum) if minimum else dict(type='string')

def enum(values):
    return dict(type='string',enum=list(values))

def array(item,minimum=0,maximum=None):
    out=dict(type='array',items=item,minItems=minimum)
    if maximum is not None:out['maxItems']=maximum
    return out

def schema_for(stage,payload):
    sentence_ids=[s['id'] for s in payload['target_sentences']]
    def ids(minimum=0):return array(enum(sentence_ids),minimum,min(4,len(sentence_ids)))
    if stage=='resolve_single':
        return obj(dict(state=enum(['yes','no','uncertain']),sids=ids(),focus=text(2),reason=text(5)))
    if stage=='resolve_exclusive':
        decision={'anyOf':[obj(dict(state=enum([state]),sids=ids(1 if state=='yes' else 0),focus=text(5)))
            for state in ('yes','no','uncertain')]}
        return obj(dict(summary=text(5),decisions=obj({t['name']:decision for t in payload['topics']})))
    if stage in {'proposal_names','independent_names','resolve_names','proposal_explained'}:
        names=[t['name'] for t in payload['topics']]
        topic=obj(dict(topic_name=enum(names),sids=ids(1),focus=text(2)))
        properties=dict(summary=text(5),topics=array(topic,0,len(names)),context_reason=text(),context_sids=ids())
        if stage=='proposal_explained':properties['context_reason']=text(5)
        if stage=='resolve_names':
            # This is a receipt for the fixed input list, not a model judgment.
            properties['reviewed_topics']={'const':names}
            properties['uncertain_topics']=array(enum(names),0,len(names))
        return obj(properties)
    if stage=='specific_content_names':
        names=[s['name'] for s in payload['specific_contents']]
        choice=obj(dict(name=enum(names),sids=ids(1)))
        variants=[]
        for state in ('matched','broad_mismatch','outside_subtopics','insufficient_detail','uncertain'):
            variants.append(obj(dict(state=enum([state]),activity=text(2),
                choices=array(choice,1,len(names)) if state=='matched' else array(choice,0,0),
                reason=text(5),sids=ids(1),suggestion=text(4 if state=='outside_subtopics' else 0))))
        return {'anyOf':variants}
    if stage in {'experience','experience_grounded'}:
        variants=[]
        for affect in ('positive','negative','mixed','neutral','unclear'):
            positive=ids(1) if affect in ('positive','mixed') else ids()
            negative=ids(1) if affect in ('negative','mixed') else ids()
            if affect=='neutral':positive=negative=array(enum(sentence_ids),0,0)
            variants.append(obj(dict(
                fatigue=enum(['self_clear','self_implicit','other_or_quoted','no_fatigue_expression','uncertain']),
                fatigue_sids=ids(),affect=enum([affect]),positive_sids=positive,negative_sids=negative,reason=text(5))))
        if stage=='experience_grounded':
            import copy
            grounded=[]
            for variant in variants:
                for statuses in (['self_clear','self_implicit','other_or_quoted'],['no_fatigue_expression','uncertain']):
                    value=copy.deepcopy(variant);value['properties']['fatigue']=enum(statuses)
                    value['properties']['fatigue_sids']=ids(1 if statuses[0]=='self_clear' else 0)
                    grounded.append(value)
            variants=grounded
        return {'anyOf':variants}
    raise ValueError('No output schema for stage '+stage)
