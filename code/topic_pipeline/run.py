"""Run production topic-decision logic on researcher-supplied local inputs."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    os.environ.update(FATIGUESCAN_WORK=str(args.work.resolve()), FATIGUESCAN_ENDPOINT=args.endpoint,
                      FATIGUESCAN_MODEL=args.model)
    import runtime as r
    from engine import Engine
    source = r.posts()
    for i, post in enumerate(source):
        assert post['position'] == i
        assert post['text_key'] == __import__('hashlib').sha256(post['text'].encode()).hexdigest()
    previous = {x['position']: x for x in r.read(r.WORK / 'prior_candidates.json')}
    assert set(previous) == set(range(len(source)))
    engine = Engine(source, previous)
    output = r.WORK / f'topic_run_{args.seed}.jsonl'
    if output.exists():
        raise SystemExit('Output exists; choose a fresh work directory to avoid mixing runs.')
    with output.open('x') as f:
        for pos in range(min(args.limit or len(source), len(source))):
            result = engine.classify(pos, args.endpoint, seed=args.seed)
            f.write(json.dumps(result, ensure_ascii=False) + '\n'); f.flush()
    print(output)


if __name__ == '__main__':
    main()
