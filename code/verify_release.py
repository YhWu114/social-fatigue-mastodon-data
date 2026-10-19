"""Offline verification of released labels against the manuscript's current results."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT/name).open() as f:
        return list(csv.DictReader(f))


def labels(value):
    return set(value.split(';')) if value else set()


def main():
    primary=rows('data/fatigue_records.csv');topics=rows('data/topic_records.csv')
    assert len(primary)==45096 and len(topics)==44092
    assert len({x['record_id'] for x in primary})==45096
    assert {x['record_id'] for x in topics}=={x['record_id'] for x in primary if x['topic_eligible']=='1'}
    totals=Counter(x['language_group'] for x in primary)
    counts=Counter((x['language_group'],t) for x in primary for t in labels(x['fatigue_types']))
    for x in rows('results/dimensions/dimension_counts.csv'):
        assert int(x['n_posts'])==totals[x['group']]
        for t in ('physical_fatigue','mental_cognitive_fatigue','emotional_fatigue','digital_social_fatigue','existential_resignation'):
            assert int(x[t])==counts[x['group'],t]
    totals=Counter(x['language_group'] for x in topics)
    counts=Counter((x['language_group'],t) for x in topics for t in labels(x['topics']))
    for x in rows('results/topics/topic_proportions_long.csv'):
        assert int(x['posts'])==counts[x['group'],x['topic_id']]
        assert int(x['denominator'])==totals[x['group']]
        assert abs(float(x['percent'])-100*int(x['posts'])/int(x['denominator']))<1e-9
    assert sum(len(labels(x['topics']))>1 for x in topics)==24019
    assert sum(not x['topics'] for x in topics)==2132
    assert sum(bool(x['unresolved_topics']) for x in topics)==4291
    assert sum(x['diagnostic_fatigue'] in ('self_clear','self_implicit') for x in topics)==37341
    specific=rows('data/specific_content.csv')
    counts=Counter((x['topic_id'],s) for x in specific if x['diagnostic_own_fatigue']=='1' for s in labels(x['subtopics']))
    for x in rows('results/topics/specific_content_counts.csv'):
        assert int(x['posts'])==counts[x['topic_id'],x['subtopic_id']],x
    reviews=rows('data/topic_review_labels.csv')
    for who,expected in [('reviewer1',(38,49)),('reviewer2',(52,56))]:
        subset=[x for x in reviews if labels(x[who+'_topics'])]
        hits=sum(bool(labels(x[who+'_topics'])&labels(x['model_topics'])) for x in subset)
        assert (hits,len(subset))==expected
    subset=[x for x in reviews if x['reviewer1_topics'] and x['reviewer2_topics']]
    assert (sum(bool(labels(x['reviewer1_topics'])&labels(x['reviewer2_topics'])) for x in subset),len(subset))==(41,47)
    prop=rows('results/topics/topic_seed_panel_proportions.csv')
    lookup={(x['seed'],x['topic_id'],x['group']):float(x['percent']) for x in prop}
    seeds=sorted({x['seed'] for x in prop})
    for x in rows('results/topics/main_text_contrasts.csv'):
        diff=[lookup[s,x['topic_id'],x['group_a']]-lookup[s,x['topic_id'],x['group_b']] for s in seeds]
        assert sum(v>0 for v in diff)==int(x['positive_runs'])
        assert abs(sum(diff)/len(diff)-float(x['mean_pp']))<1e-9
    print(json.dumps(dict(status='passed',primary_records=len(primary),topic_records=len(topics),
        topics=17,review_posts=len(reviews),checks=['dimension counts','topic proportions','specific content counts',
        'uncertainty flags','human topic matching','repeat-run contrasts']),indent=2))


if __name__=='__main__':
    main()
