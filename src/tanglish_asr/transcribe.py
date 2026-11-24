"""Reference-blind official Whisper inference and reproducible smoke runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .audit import verify_sources
from .common import read_jsonl, safe_artifact, sha256, timestamp, write_csv, write_json, write_jsonl
from .evaluate import check_review, script_stats

LOG = logging.getLogger(__name__)


def inference_selection(rows: list[dict], smoke: bool, limit: int | None) -> list[dict]:
    if not rows:
        raise ValueError('Inference manifest is empty')
    if smoke:
        if limit is None or not 1 <= limit <= 3:
            raise ValueError('Unreviewed smoke mode requires a limit between one and three')
        # Choose short clips from different sources, without consulting references.
        selected, sources = [], set()
        ordered = sorted(rows, key=lambda r:(r['duration_s'],r['clip_id']))
        for row in ordered:
            if row['recording_id'] not in sources:
                selected.append(row)
                sources.add(row['recording_id'])
                if len(selected) == limit:
                    break
        return selected
    check_review(rows)
    return rows if limit is None else rows[:limit]


def validate_prediction_context(prediction: dict, config_hash: str, audio_hash: str) -> None:
    if prediction.get('config_sha256') != config_hash or prediction.get('clip_audio_sha256') != audio_hash:
        raise ValueError('Cannot resume predictions with changed configuration or audio content')


def package_versions() -> dict:
    return {d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}


def verify_inference_audio(rows: list[dict], root: Path, output: Path) -> None:
    for row in rows:
        path = (root / row['audio_path']).resolve()
        if not path.is_relative_to((output / 'clips').resolve()) or sha256(path) != row['clip_audio_sha256']:
            raise ValueError(f"Derived audio changed or escaped clip directory: {row['clip_id']}")


def inference_context(cfg: dict) -> dict:
    import torch
    import whisper
    backend = cfg['backend']
    if backend == 'cuda' and not torch.cuda.is_available():
        backend = 'cpu'
    if backend == 'mps' and not torch.backends.mps.is_available():
        backend = 'cpu'
    if backend not in {'cpu', 'cuda', 'mps'}:
        raise ValueError(f'Unsupported backend: {backend}')
    options = dict(cfg['inference'])
    if backend == 'cpu':
        options['fp16'] = False
    url = whisper._MODELS[cfg['model']]
    code_hashes = {}
    for label, directory in [('pipeline', Path(__file__).parent), ('whisper', Path(whisper.__file__).parent)]:
        for source in sorted(directory.rglob('*.py')):
            code_hashes[label + '/' + str(source.relative_to(directory))] = sha256(source)
    ffmpeg = subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    payload = {'backend': backend, 'options': options, 'model_url': url,
               'model_sha256': url.split('/')[-2], 'packages': package_versions(),
               'code_sha256': code_hashes, 'python': sys.version, 'ffmpeg': ffmpeg,
               'threads': cfg['threads'], 'seed': cfg['seed']}
    payload['inference_signature'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def transcribe(cfg: dict, root: Path, output: Path, *, smoke: bool = False,
               limit: int | None = None, retry_failed: bool = False) -> dict:
    preparation = json.loads((output/'preparation.json').read_text())
    verify_sources(root, preparation)
    rows = read_jsonl(output/'inference_manifest.jsonl')
    selected = inference_selection(rows, smoke, limit)
    config_hash = hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if preparation['config'] != cfg:
        raise ValueError('Configuration changed since preparation; create a new output directory')
    previous = read_jsonl(output/'predictions.jsonl') if (output/'predictions.jsonl').exists() else []
    if len({r['clip_id'] for r in previous}) != len(previous):
        raise ValueError('Duplicate prediction IDs in stored outputs')
    records = {r['clip_id']:r for r in previous}
    if set(records) - {r['clip_id'] for r in rows}:
        raise ValueError('Stored predictions contain IDs outside the manifest')
    verify_inference_audio(rows, root, output)
    context = inference_context(cfg)
    for row in rows:
        if row['clip_id'] in records:
            validate_prediction_context(records[row['clip_id']],config_hash,row['clip_audio_sha256'])
            if records[row['clip_id']].get('inference_signature') != context['inference_signature']:
                raise ValueError('Inference environment changed; use a separate output directory')
            if records[row['clip_id']].get('model_sha256') != context['model_sha256']:
                raise ValueError('Stored model checkpoint provenance changed')
    weights_dir = safe_artifact(output, 'models')
    model_path = safe_artifact(output, 'models/' + cfg['model'] + '.pt')
    if records and (not model_path.exists() or sha256(model_path) != context['model_sha256']):
        raise ValueError('Cached predictions require the original verified checkpoint')
    selected = [r for r in selected if r['clip_id'] not in records or
                (retry_failed and records[r['clip_id']]['status']!='ok')]
    if not selected:
        return {'status':'already_completed','completed_predictions':len(records)}
    import numpy as np
    import torch
    import whisper

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['threads'])
    requested = context['backend']
    if requested != cfg['backend']:
        LOG.warning('Requested backend unavailable; using CPU fp32')
    download_url = context['model_url']
    expected_sha256 = context['model_sha256']
    load_start = time.perf_counter()
    LOG.info('Loading official Whisper %s on %s',cfg['model'],requested)
    model = whisper.load_model(cfg['model'],device=requested,download_root=str(weights_dir))
    load_seconds = time.perf_counter()-load_start
    model_hash = sha256(model_path)
    if model_hash != expected_sha256:
        raise ValueError('Official checkpoint checksum mismatch')
    options = context['options']
    run_id = timestamp().replace(':','').replace('+','_')
    run_dir = safe_artifact(output, 'runs/' + run_id)
    metadata = {'run_id':run_id,'created_at':timestamp(),'mode':'pipeline_smoke' if smoke else 'reviewed_pilot',
                'backend':requested,'python':sys.version,'platform':platform.platform(),
                'packages':package_versions(),'model':'openai-whisper/'+cfg['model'],
                'weight_url':download_url,'model_sha256':model_hash,'expected_sha256':expected_sha256,
                'whisper_code_sha256':sha256(Path(whisper.__file__)),
                'config':cfg,'config_sha256':config_hash,'actual_inference_options':options,
                'inference_signature':context['inference_signature'],'inference_context':context,
                'model_load_seconds':load_seconds,'audio_manifest_sha256':sha256(output/'inference_manifest.jsonl'),
                'selected_clip_ids':[r['clip_id'] for r in selected],
                'mps_available':torch.backends.mps.is_available(),'cuda_available':torch.cuda.is_available(),
                'cloud_api_cost_usd':0,'local_electricity_cost':'unmeasured',
                'accuracy_status':'not_scored_without_human_review'}
    write_json(run_dir/'environment.json',metadata)
    command = subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
    if command.returncode == 0:
        (run_dir/'pip_freeze.txt').write_text(command.stdout)
    for row in selected:
        start = time.perf_counter()
        record = {'clip_id':row['clip_id'],'recording_id':row['recording_id'],'run_id':run_id,
                  'config_sha256':config_hash,'clip_audio_sha256':row['clip_audio_sha256'],
                  'inference_signature':context['inference_signature'],
                  'model_sha256':model_hash,'audio_duration_s':row['duration_s'],
                  'mode':metadata['mode'],'backend':requested,'started_at':timestamp()}
        try:
            LOG.info('Transcribing %s (%s seconds)',row['clip_id'],row['duration_s'])
            response = model.transcribe(str(root/row['audio_path']),**options)
            elapsed = time.perf_counter()-start
            if not isinstance(response.get('text'),str):
                raise ValueError('Whisper response has no text field')
            record.update({'status':'ok','text':response['text'],'raw_response':response,
                           'reported_language':response.get('language'),
                           'language_source':'forced_config' if options.get('language') else 'model_detected',
                           'inference_seconds':elapsed,'rtf':elapsed/row['duration_s'],
                           **script_stats(response['text'])})
        except Exception as error:
            elapsed = time.perf_counter()-start
            record.update({'status':'failed','text':None,'inference_seconds':elapsed,
                           'exception_type':type(error).__name__,'exception':str(error),
                           'traceback':traceback.format_exc()})
            LOG.exception('Clip failed: %s',row['clip_id'])
        records[row['clip_id']] = record
        write_json(run_dir/(row['clip_id']+'.json'),record)
        write_jsonl(output/'predictions.jsonl',list(records.values()))
    successful = [r for r in records.values() if r['status']=='ok']
    total_audio = sum(r['audio_duration_s'] for r in successful)
    total_inference = sum(r['inference_seconds'] for r in successful)
    result = {'status':'pipeline_smoke_completed' if smoke else 'reviewed_pilot_completed',
              'successful_clips':len(successful),'failed_clips':len(records)-len(successful),
              'successful_audio_s':total_audio,'inference_seconds':total_inference,
              'aggregate_rtf':total_inference/total_audio if total_audio else None,
              'model_load_seconds_this_run':load_seconds,
              'roman_compliant_clips':sum(r['roman_compliant'] for r in successful),
              'nonempty_clips':sum(not r['empty_output'] for r in successful),
              'accuracy_metrics':None,'accuracy_note':'Run evaluate only after human reference review.',
              'cloud_api_cost_usd':0}
    write_json(output/'inference_summary.json',result)
    write_csv(output/'baseline_outputs.csv',list(records.values()),
              ['clip_id','recording_id','status','mode','text','reported_language','language_source',
               'audio_duration_s','inference_seconds','rtf','roman_compliant','latin_letter_fraction',
               'model_sha256','exception_type','exception'])
    return result
