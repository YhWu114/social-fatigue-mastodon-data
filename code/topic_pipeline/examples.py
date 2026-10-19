"""Retrieve locally supplied human examples using precomputed multilingual embeddings."""
import numpy as np
import runtime as r


class Examples:
    def __init__(self, source):
        config = r.read(r.WORK / 'embedding_config.json')
        if config['source_sha256'] != r.sha(r.WORK / 'posts.jsonl'):
            raise ValueError('Embeddings do not match the input file')
        self.E = np.load(r.WORK / 'embeddings.npy', mmap_mode='r')
        if len(self.E) != len(source):
            raise ValueError('One embedding is required for each post')
        self.refs = []
        for item in r.read(r.WORK / 'human_examples.json'):
            if item.get('exclude', False):
                continue
            pos = item['position']; post = source[pos]
            if len(post['text']) > 6000:
                continue
            self.refs.append(dict(position=pos, text_key=post['text_key'], text=post['text'],
                                  topics=item['topics'], comment=item.get('comment', ''),
                                  sample_id=item['reference_id']))
        if not self.refs:
            raise ValueError('Human-reviewed examples are required')
        self.M = np.array(self.E[[x['position'] for x in self.refs]], dtype=float)
        self.M /= np.maximum(1e-12, np.linalg.norm(self.M, axis=1, keepdims=True))

    def get(self, post):
        vector = self.E[post['position']]
        similarities = self.M @ vector / max(1e-12, np.linalg.norm(vector))
        order = np.argsort(-similarities); selected = []
        for nonempty, n in [(True, 2), (False, 1)]:
            selected += [i for i in order if bool(self.refs[i]['topics']) == nonempty
                         and self.refs[i]['text_key'] != post['text_key']][:n]
        return [dict(text=self.refs[i]['text'], human_topics=self.refs[i]['topics'],
                     human_comment=self.refs[i]['comment'], reference_id=self.refs[i]['sample_id'])
                for i in selected]
