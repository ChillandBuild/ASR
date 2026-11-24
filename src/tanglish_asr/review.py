"""Import explicit human audio/reference review without changing originals."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .common import read_jsonl, timestamp, write_json, write_jsonl
from .evaluate import check_review


def import_review(output: Path, csv_path: Path) -> dict:
    rows = read_jsonl(output/'manifest.jsonl')
    with csv_path.open(newline='') as stream:
        reviews = list(csv.DictReader(stream))
    if len({r['clip_id'] for r in reviews}) != len(reviews):
        raise ValueError('Duplicate review clip IDs')
    indexed = {r['clip_id']:r for r in reviews}
    if set(indexed) != {r['clip_id'] for r in rows}:
        raise ValueError('Review CSV must contain exactly the prepared pilot IDs')
    for row in rows:
        review = indexed[row['clip_id']]
        if review['reference'] != row['reference']:
            raise ValueError('Original reference was edited; use adjudicated_reference instead')
        row.update({k:review.get(k,'') for k in ('review_status','reviewer','reviewed_at','review_notes')})
        speakers = json.loads(review.get('speaker_ids') or '[]')
        if not isinstance(speakers,list) or any(not isinstance(s,str) or not s.strip() for s in speakers):
            raise ValueError('speaker_ids must be a JSON list of nonempty strings')
        row['speaker_ids'] = speakers
        row['adjudicated_reference'] = review.get('adjudicated_reference') or None
        if row['review_status']=='approved':
            check_review([row])
    audit_name = timestamp().replace(':','').replace('+','_')
    write_json(output/'reviews'/(audit_name+'.json'),{'imported_at':timestamp(),'csv':str(csv_path),
                                                    'reviews':reviews})
    write_jsonl(output/'manifest.jsonl',rows)
    inference = read_jsonl(output/'inference_manifest.jsonl')
    for row in inference:
        reviewed = next(r for r in rows if r['clip_id']==row['clip_id'])
        row.update({k:reviewed[k] for k in ('review_status','reviewer','reviewed_at')})
    write_jsonl(output/'inference_manifest.jsonl',inference)
    return {'approved':sum(r['review_status']=='approved' for r in rows),'total':len(rows)}
