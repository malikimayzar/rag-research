import json
import random
from pathlib import Path

random.seed(42)

with open('data/processed/ground_truth_qa.json') as f:
    data = json.load(f)

# Group by chunk_id
from collections import defaultdict
by_chunk = defaultdict(list)
for s in data:
    by_chunk[s['gold_chunk_id']].append(s)
    
from collections import defaultdict as dd

# Group chunk_ids by tier
chunks_by_tier = dd(list)
for chunk_id in by_chunk:
    tier = by_chunk[chunk_id][0]['tier']
    chunks_by_tier[tier].append(chunk_id)

train_chunks, holdout_chunks = set(), set()
for tier, cids in chunks_by_tier.items():
    random.shuffle(cids)
    split = int(len(cids) * 0.8)
    train_chunks.update(cids[:split])
    holdout_chunks.update(cids[split:])

train   = [s for s in data if s['gold_chunk_id'] in train_chunks]
holdout = [s for s in data if s['gold_chunk_id'] in holdout_chunks]

Path('data/processed/train_eval.json').write_text(
    json.dumps(train, indent=2, ensure_ascii=False))
Path('data/processed/holdout_eval.json').write_text(
    json.dumps(holdout, indent=2, ensure_ascii=False))

all_chunks = set(by_chunk.keys())

print(f'All chunks     : {len(all_chunks)}')
print(f'Train chunks   : {len(train_chunks)}')
print(f'Holdout chunks : {len(holdout_chunks)}')
print(f'Intersection   : {len(train_chunks & holdout_chunks)}')
print(f'Coverage OK    : {len(train_chunks | holdout_chunks) == len(all_chunks)}')