"""Source-aware transcript segmentation and deterministic pilot sampling."""
from __future__ import annotations

import collections
import hashlib
import math
import random
import re
import subprocess
import wave
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from .common import audio_info, safe_artifact, sha256, timestamp, write_csv, write_json, write_jsonl

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def read_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        tree = ElementTree.fromstring(archive.read('word/document.xml'))
    return [''.join(t.text or '' for t in p.findall('.//w:t', NS))
            for p in tree.findall('.//w:p', NS)]


def extract_segments(paragraphs: list[str], duration: float, line_labels: bool,
                     section_start: str | None = None, section_end: str | None = None,
                     max_segment_s: float = 30) -> tuple[list[dict], list[dict]]:
    begin, end = 0, len(paragraphs)
    if section_start:
        if paragraphs.count(section_start) != 1:
            raise ValueError(f'Section start not unique: {section_start}')
        begin = paragraphs.index(section_start) + 1
    if section_end:
        if paragraphs.count(section_end) != 1:
            raise ValueError(f'Section end not unique: {section_end}')
        end = paragraphs.index(section_end)
    if end <= begin:
        raise ValueError('Section end precedes start')
    anchors, rejected = [], []
    current = None
    for index in range(begin, end):
        raw = paragraphs[index]
        text = raw.strip()
        labelled = bool(re.match(r'^L\d+', text)) if line_labels else False
        prefix = re.match(r'^L\d+\s*', text) if labelled else None
        if prefix:
            text = text[prefix.end():]
        match = re.match(r'^\[(\d{1,3}):(\d{2})\]\s*', text)
        if not match and labelled:
            match = re.match(r'^(\d{1,3}):(\d{2})\s*', text)
        if match:
            if int(match[2]) >= 60:
                raise ValueError(f'Invalid seconds at paragraph {index}: {match[0]}')
            current = {'start_s': int(match[1])*60 + int(match[2]), 'pieces': [],
                       'source_paragraphs': [], 'paragraph_indices': [], 'structural_removals': []}
            anchors.append(current)
            current['structural_removals'].append({'paragraph': index,
                                                   'line_prefix': prefix[0] if prefix else '',
                                                   'timestamp_prefix': match[0]})
            text = text[match.end():]
        elif text.startswith('[') and re.match(r'^\[\d+:', text):
            raise ValueError(f'Malformed timestamp at paragraph {index}')
        if current is not None:
            current['source_paragraphs'].append(raw)
            current['paragraph_indices'].append(index)
            if text:
                current['pieces'].append(text)
            if prefix and not match:
                current['structural_removals'].append({'paragraph': index,
                                                       'line_prefix': prefix[0]})
    frequencies = collections.Counter(a['start_s'] for a in anchors)
    segments = []
    for i, anchor in enumerate(anchors):
        reference = ' '.join(anchor.pop('pieces'))
        row = {**anchor, 'reference': reference}
        reason = None
        if i == len(anchors)-1:
            reason = 'unverified_final_endpoint'
        else:
            row['end_s'] = anchors[i+1]['start_s']
            row['duration_s'] = row['end_s'] - row['start_s']
            if not 0 <= row['start_s'] < row['end_s'] <= duration:
                reason = 'invalid_or_nonincreasing_bounds'
            if frequencies[row['start_s']] > 1 or frequencies[row['end_s']] > 1:
                reason = 'duplicate_anchor'
            if row['duration_s'] > max_segment_s:
                reason = 'interval_exceeds_limit'
            if not reference.strip():
                reason = 'empty_reference'
        if reason:
            rejected.append({**row, 'reason': reason})
        else:
            segments.append(row)
    if not anchors:
        rejected.append({'reason': 'no_timestamp_anchors', 'paragraph_count': end-begin})
    return segments, rejected


def validate_manifest(rows: list[dict]) -> None:
    ids = set()
    groups = collections.defaultdict(list)
    for row in rows:
        if row['clip_id'] in ids:
            raise ValueError(f"Duplicate clip ID: {row['clip_id']}")
        ids.add(row['clip_id'])
        start, end, duration = (float(row[k]) for k in ('start_s', 'end_s', 'recording_duration_s'))
        if not all(math.isfinite(v) for v in (start, end, duration)) or not 0 <= start < end <= duration:
            raise ValueError(f"Invalid audio bounds: {row['clip_id']}")
        groups[row['source_audio_sha256']].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda r: r['start_s'])
        for previous, current in zip(ordered, ordered[1:]):
            if previous['end_s'] > current['start_s']:
                raise ValueError(f"Overlapping or duplicate audio: {previous['clip_id']} / {current['clip_id']}")


def select_pilot(rows: list[dict], target_seconds: float, seed: int) -> list[dict]:
    if target_seconds <= 0:
        raise ValueError('Pilot duration must be positive')
    if sum(r['duration_s'] for r in rows) < target_seconds:
        raise ValueError('Insufficient candidate duration for requested pilot')
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['recording_id']].append(row)
    queues = {}
    for recording, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda r: r['start_s'])
        buckets = [[] for _ in range(10)]
        for i, row in enumerate(ordered):
            buckets[min(9, i*10//len(ordered))].append(row)
        for bucket in buckets:
            rng.shuffle(bucket)
        queue = []
        while any(buckets):
            for bucket in buckets:
                if bucket:
                    queue.append(bucket.pop())
        queues[recording] = queue
    totals = dict.fromkeys(queues, 0.0)
    selected, total = [], 0.0
    while total < target_seconds:
        remaining = [k for k, queue in queues.items() if queue]
        if not remaining:
            raise ValueError('Sampling exhausted candidate clips')
        key = min(remaining, key=lambda k: (totals[k], k))
        row = queues[key].pop(0)
        selected.append(row)
        totals[key] += row['duration_s']
        total += row['duration_s']
    return selected


def prepare(cfg: dict, root: Path, output: Path, *, materialize: bool = True) -> dict:
    if (output / 'manifest.jsonl').exists():
        raise ValueError('Manifest already exists; use a new output_dir to preserve review records')
    all_rows, quarantined, inventories = [], [], []
    for recording in cfg['recordings']:
        audio = root / recording['audio']
        document = root / recording['transcript']
        info = audio_info(audio)
        audio_hash, document_hash = sha256(audio), sha256(document)
        inventories.append({'recording_id': recording['id'], 'audio': recording['audio'],
                            'source_audio_sha256': audio_hash, 'source_docx_sha256': document_hash,
                            'transcript': recording['transcript'], **info})
        segments, rejected = extract_segments(read_paragraphs(document), info['duration_s'],
                                             recording.get('line_labels', False),
                                             recording.get('section_start'), recording.get('section_end'),
                                             cfg['max_segment_s'])
        quarantined.extend({'recording_id': recording['id'], **row} for row in rejected)
        if not recording.get('pilot'):
            continue
        for segment in segments:
            clip_id = f"{recording['id']}_{segment['start_s']:05d}_{segment['end_s']:05d}"
            all_rows.append({**segment, 'clip_id': clip_id, 'recording_id': recording['id'],
                             'source_audio_path': recording['audio'], 'source_audio_sha256': audio_hash,
                             'source_docx_path': recording['transcript'], 'source_docx_sha256': document_hash,
                             'recording_duration_s': info['duration_s'],
                             'reference_sha256': hashlib.sha256(segment['reference'].encode()).hexdigest(),
                             'speaker_ids': [], 'split': 'pilot_exploratory',
                             'review_status': 'pending', 'reviewer': '', 'reviewed_at': '',
                             'review_notes': '', 'adjudicated_reference': None,
                             'permission_status': 'user_authorized_local_processing'})
    validate_manifest(all_rows)
    selected = select_pilot(all_rows, cfg['pilot_duration_s'], cfg['seed'])
    if materialize:
        for row in selected:
            wav_path = safe_artifact(output, 'clips/' + row['clip_id'] + '.wav')
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-ss', str(row['start_s']), '-i',
                            str(root / row['source_audio_path']),
                            '-t', str(row['duration_s']), '-ac', '1', '-ar', '16000',
                            '-c:a', 'pcm_s16le', str(wav_path)], check=True, capture_output=True)
            with wave.open(str(wav_path)) as wav:
                actual = wav.getnframes()/wav.getframerate()
                if wav.getframerate()!=16000 or wav.getnchannels()!=1 or abs(actual-row['duration_s'])>.002:
                    raise ValueError(f"Derived audio verification failed: {row['clip_id']}")
            row['audio_path'] = str(wav_path.relative_to(root))
            row['clip_audio_sha256'] = sha256(wav_path)
            row['actual_duration_s'] = actual
    validate_manifest(selected)
    write_jsonl(output / 'manifest.jsonl', selected)
    # The inference manifest deliberately excludes all transcripts and reviewer notes.
    fields = ['clip_id','recording_id','audio_path','clip_audio_sha256','duration_s','split','review_status']
    write_jsonl(output / 'inference_manifest.jsonl', [{k:r[k] for k in fields if k in r} for r in selected])
    write_csv(output / 'reference_review.csv', selected,
              ['clip_id','recording_id','audio_path','start_s','end_s','reference','adjudicated_reference',
               'speaker_ids','review_status','reviewer','reviewed_at','review_notes'])
    write_jsonl(output / 'quarantine.jsonl', quarantined)
    summary = {'prepared_at':timestamp(), 'selected_clips':len(selected),
               'selected_duration_s':sum(r['duration_s'] for r in selected),
               'candidate_clips':len(all_rows),'quarantined_entries':len(quarantined),
               'review_status':'human_audio_reference_review_pending',
               'by_recording':dict(collections.Counter(r['recording_id'] for r in selected)),
               'sources':inventories, 'seed':cfg['seed'], 'config':cfg}
    write_json(output / 'preparation.json', summary)
    return summary
