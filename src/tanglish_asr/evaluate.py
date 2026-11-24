"""Raw edit-count metrics and explicit review gates."""
from __future__ import annotations

import collections
import hashlib
import json
import unicodedata
from datetime import datetime
from pathlib import Path

from .common import read_jsonl, safe_artifact, sha256, write_csv, write_json
from .normalize import cleanup


def edit_counts(reference: list, hypothesis: list) -> dict:
    # Keep a single row of edit cost plus substitution/deletion/insertion counts.
    previous = [(j, 0, 0, j) for j in range(len(hypothesis)+1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            cost, sub, delete, insert = previous[j-1]
            diagonal = (cost+(ref!=hyp), sub+(ref!=hyp), delete, insert)
            cost, sub, delete, insert = previous[j]
            deletion = (cost+1, sub, delete+1, insert)
            cost, sub, delete, insert = current[j-1]
            insertion = (cost+1, sub, delete, insert+1)
            current.append(min((diagonal, deletion, insertion), key=lambda cell: cell[0]))
        previous = current
    _, sub, delete, insert = previous[-1]
    return {'substitutions':sub,'deletions':delete,'insertions':insert,'reference_units':len(reference)}


def score_pairs(pairs: list[tuple[str, str]]) -> dict:
    if not pairs or any(not reference.strip() for reference, _ in pairs):
        raise ValueError('Scoring requires nonempty references and at least one pair')
    word, char = collections.Counter(), collections.Counter()
    for reference, hypothesis in pairs:
        word.update(edit_counts(reference.split(), hypothesis.split()))
        char.update(edit_counts(list(reference), list(hypothesis)))
    def rate(counts: dict) -> float:
        return sum(counts[k] for k in ('substitutions','deletions','insertions')) / counts['reference_units']
    return {'wer':rate(word),'cer':rate(char),'word_errors':dict(word),'character_errors':dict(char)}


def script_stats(text: str) -> dict:
    letters = [c for c in text if c.isalpha()]
    latin = sum('LATIN' in unicodedata.name(c,'') for c in letters)
    non_latin = collections.Counter(c for c in letters if 'LATIN' not in unicodedata.name(c,''))
    return {'letter_count':len(letters),'latin_letter_count':latin,
            'latin_letter_fraction':latin/len(letters) if letters else None,
            'non_latin_letters':dict(non_latin),
            'roman_compliant':bool(letters) and latin==len(letters),
            'empty_output':not text.strip()}


def check_review(rows: list[dict]) -> None:
    for row in rows:
        if row.get('review_status') != 'approved' or not row.get('reviewer','').strip() or not row.get('reviewed_at','').strip():
            raise ValueError(f"Human audio/reference review missing: {row['clip_id']}")
        reviewed = datetime.fromisoformat(row['reviewed_at'])
        if reviewed.tzinfo is None:
            raise ValueError('Review timestamps must include timezone')


def validate_prediction_provenance(rows: list[dict], predictions: list[dict], config_hash: str,
                                   model_hash: str, signature: str) -> None:
    indexed = {r['clip_id']: r for r in rows}
    if len(indexed) != len(rows) or len({r['clip_id'] for r in predictions}) != len(predictions):
        raise ValueError('Duplicate manifest or prediction IDs')
    for prediction in predictions:
        row = indexed.get(prediction['clip_id'])
        if row is None or prediction.get('clip_audio_sha256') != row['clip_audio_sha256']:
            raise ValueError('Prediction does not correspond to the prepared audio')
        if prediction.get('config_sha256') != config_hash or prediction.get('model_sha256') != model_hash:
            raise ValueError('Prediction configuration or checkpoint provenance mismatch')
        if prediction.get('inference_signature') != signature:
            raise ValueError('Mixed inference environments cannot be evaluated together')


def evaluate_records(rows: list[dict], predictions: list[dict]) -> tuple[dict, list, list, list]:
    """Pure metric aggregation; the filesystem entry point verifies provenance first."""
    check_review(rows)
    if len({r['clip_id'] for r in predictions}) != len(predictions):
        raise ValueError('Duplicate prediction IDs')
    pred = {r['clip_id']:r for r in predictions}
    if set(pred)-{r['clip_id'] for r in rows}:
        raise ValueError('Predictions contain IDs outside the manifest')
    pairs, all_pairs, clips, review, cleanup_log = [], [], [], [], []
    by_recording = collections.defaultdict(list)
    for row in rows:
        item = pred.get(row['clip_id'])
        reference = row.get('adjudicated_reference')
        if reference is None:
            reference = row['reference']
        success = bool(item and item.get('status')=='ok')
        hypothesis = item['text'] if success else ''
        all_pairs.append((reference, hypothesis))
        if success:
            pairs.append((reference,hypothesis))
            by_recording[row['recording_id']].append((reference,hypothesis))
        metrics = score_pairs([(reference,hypothesis)])
        clips.append({'clip_id':row['clip_id'],'recording_id':row['recording_id'],
                      'status':item.get('status') if item else 'not_run', **metrics,
                      **script_stats(hypothesis)})
        review.append({'clip_id':row['clip_id'],'speaker_ids':row['speaker_ids'],
                       'reference':reference,'hypothesis':hypothesis,
                       'error_category':'unreviewed','reviewer_notes':'',
                       'word_substitutions':metrics['word_errors']['substitutions'],
                       'word_deletions':metrics['word_errors']['deletions'],
                       'word_insertions':metrics['word_errors']['insertions']})
        cleanup_log.append({'clip_id':row['clip_id'],'reference_before':reference,
                            'reference_after':cleanup(reference),'hypothesis_before':hypothesis,
                            'hypothesis_after':cleanup(hypothesis)})
    result = {'status':'human_reviewed_exploratory_accuracy','manifest_clips':len(rows),
              'successful_clips':len(pairs),'failed_or_missing_clips':len(rows)-len(pairs),
              'successful_only_raw':score_pairs(pairs) if pairs else None,
              'failures_as_empty_raw':score_pairs(all_pairs),
              'failures_as_empty_cleanup':score_pairs([(cleanup(a),cleanup(b)) for a,b in all_pairs]),
              'by_recording_successful_only':{k:score_pairs(v) for k,v in by_recording.items()},
              'speaker_independent':False,'confidence_intervals':None,
              'limitation':'Unknown speaker identities and exploratory sample; no generalization or significance claim.'}
    return result, clips, review, cleanup_log


def evaluate(output: Path, cfg: dict, root: Path) -> dict:
    from .audit import verify_sources
    from .manifest import validate_manifest
    from .transcribe import inference_context, verify_inference_audio

    rows = read_jsonl(output / 'manifest.jsonl')
    check_review(rows)
    preparation = json.loads((output / 'preparation.json').read_text())
    if preparation['config'] != cfg:
        raise ValueError('Configuration changed since preparation')
    verify_sources(root, preparation)
    validate_manifest(rows)
    verify_inference_audio(rows, root, output)
    for row in rows:
        if hashlib.sha256(row['reference'].encode()).hexdigest() != row['reference_sha256']:
            raise ValueError('Original reference was edited without adjudication')
    predictions = read_jsonl(output / 'predictions.jsonl')
    context = inference_context(cfg)
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    model_path = safe_artifact(output, 'models/' + cfg['model'] + '.pt')
    if sha256(model_path) != context['model_sha256']:
        raise ValueError('Model checkpoint changed before evaluation')
    validate_prediction_provenance(rows, predictions, config_hash, context['model_sha256'], context['inference_signature'])
    for prediction in predictions:
        run_id = prediction.get('run_id', '')
        run_path = safe_artifact(output, 'runs/' + run_id + '/environment.json')
        environment = json.loads(run_path.read_text())
        if (environment.get('config_sha256') != config_hash or
            environment.get('inference_signature') != context['inference_signature'] or
            environment.get('model_sha256') != context['model_sha256'] or
            prediction['clip_id'] not in environment.get('selected_clip_ids', [])):
            raise ValueError('Missing or incompatible run provenance')
    result, clips, review, cleanup_log = evaluate_records(rows, predictions)
    write_json(output / 'evaluation.json', result)
    write_csv(output / 'per_clip_scores.csv', clips,
              ['clip_id','recording_id','status','wer','cer','word_errors','character_errors',
               'roman_compliant','latin_letter_fraction','empty_output'])
    write_csv(output / 'error_review.csv', review,
              ['clip_id','speaker_ids','reference','hypothesis','error_category','reviewer_notes',
               'word_substitutions','word_deletions','word_insertions'])
    write_json(output / 'cleanup_transformations.json', cleanup_log)
    return result
