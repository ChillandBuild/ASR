"""Configuration, serialization and source provenance."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_artifact(output: Path, relative: str) -> Path:
    path = output / relative
    if not path.resolve().is_relative_to(output.resolve()):
        raise ValueError(f'Artifact path escapes output directory: {path}')
    return path


def protect_private_write(path: Path) -> None:
    absolute = path.absolute()
    for parent in absolute.parents:
        if parent.parent.name == 'private_outputs':
            safe_artifact(parent, str(absolute.relative_to(parent)))
            return


def write_json(path: Path, value: Any) -> None:
    protect_private_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    protect_private_write(tmp)
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    protect_private_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    protect_private_write(tmp)
    with tmp.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    protect_private_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def load_config(path: str | Path) -> tuple[dict, Path, Path]:
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    root = path.parent.parent
    output = (root / cfg['output_dir']).resolve()
    if output == root / 'private_outputs' or not output.is_relative_to(root / 'private_outputs'):
        raise ValueError('output_dir must be inside workspace private_outputs/')
    if not cfg.get('local_processing_authorized'):
        raise ValueError('Local audio processing is not authorized in config')
    ids = [r['id'] for r in cfg['recordings']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate recording IDs in configuration')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value) for value in ids):
        raise ValueError('Recording IDs must be safe filenames: letters, numbers, underscore or hyphen')
    for recording in cfg['recordings']:
        for key in ('audio', 'transcript'):
            source = (root / recording[key]).resolve()
            if not source.is_relative_to(root) or source.is_relative_to(output):
                raise ValueError(f'Invalid source path: {recording[key]}')
    return cfg, root, output


def audio_info(path: Path) -> dict:
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams',
                             '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    streams = [s for s in info['streams'] if s['codec_type'] == 'audio']
    if len(streams) != 1:
        raise ValueError(f'Expected exactly one audio stream: {path}')
    return {'duration_s': float(info['format']['duration']),
            'sample_rate': int(streams[0]['sample_rate']), 'channels': streams[0]['channels'],
            'codec': streams[0]['codec_name']}
