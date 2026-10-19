"""Synthetic, offline checks; no requests and no real post text."""
import importlib.util
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'topic_pipeline'))
import runtime
import engine
import output_schema
import screen
from jsonschema import validate


class PipelineTests(unittest.TestCase):
    def test_screening_rule(self):
        self.assertFalse(screen.normalize('{"is_fatigue":true,"categories":{}}')['retained'])
        self.assertTrue(screen.normalize('{"is_fatigue":true,"categories":{"physical_fatigue":0.9}}')['retained'])
        with self.assertRaises(ValueError):screen.normalize('{"is_fatigue":"yes"}')

    def test_keywords(self):
        self.assertTrue(screen.candidate('I feel tired today.','en'))
        self.assertTrue(screen.candidate('今天好困。','zh'))
        self.assertFalse(screen.candidate('Good morning.','en'))

    def test_segments(self):
        text='A short sentence.\n另一个句子。'+'x'*1400
        parts=runtime.segments(text)
        self.assertEqual(''.join(x['text'] for x in parts),text)
        self.assertTrue(runtime.evidence(parts,['S001']))
        with self.assertRaises(ValueError):runtime.evidence(parts,['S999'])

    def test_proposal_schema_and_evidence(self):
        parts=runtime.segments('I had a nap.')
        definitions={'T03':dict(name='Sleep',definition='Sleep and rest')}
        payload=dict(target_sentences=parts,topics=list(definitions.values()))
        obj=dict(summary='A person had a nap.',topics=[dict(topic_name='Sleep',sids=['S001'],focus='Had a nap')],context_reason='',context_sids=[])
        validate(obj,output_schema.schema_for('proposal_names',payload))
        self.assertEqual(engine.validate_proposal(obj,parts,definitions)['topics'][0]['id'],'T03')
        obj['topics'][0]['sids']=['S999']
        with self.assertRaises(Exception):validate(obj,output_schema.schema_for('proposal_names',payload))


if __name__=='__main__':unittest.main()
