"""Re-runnable local inventory and original-file integrity checks."""
from __future__ import annotations

import collections
import platform
import subprocess
import sys
from pathlib import Path

from .common import audio_info, sha256, timestamp, write_json
from .manifest import read_paragraphs


def audit(cfg: dict, root: Path, output: Path, decode: bool = False) -> dict:
    paths = sorted({r['audio'] for r in cfg['recordings']} | set(cfg.get('duplicate_aliases', [])))
    inventory, seen = [], {}
    for name in paths:
        path = root/name
        digest = sha256(path)
        row = {'path':name,'sha256':digest,'bytes':path.stat().st_size,**audio_info(path),
               'duplicate_of':seen.get(digest)}
        if decode:
            checked = subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(path),
                                      '-map','0:a:0','-f','null','-'], capture_output=True, text=True)
            row['decode_exit_code'], row['decode_error'] = checked.returncode, checked.stderr
        seen.setdefault(digest, name)
        inventory.append(row)
    docs = []
    for name in sorted({r['transcript'] for r in cfg['recordings']}):
        path = root/name
        paragraphs = read_paragraphs(path)
        docs.append({'path':name,'sha256':sha256(path),'paragraphs':len(paragraphs),
                     'empty_paragraphs':sum(not p.strip() for p in paragraphs)})
    result = {'audited_at':timestamp(),'audio':inventory,'documents':docs,
              'unique_audio_hours':sum(r['duration_s'] for r in inventory if not r['duplicate_of'])/3600,
              'python':sys.version,'architecture':platform.machine(),'permission':cfg['authorization_note'],
              'license_status':cfg['dataset_license_status'],'speaker_ids_verified':False}
    write_json(output/'audit.json',result)
    return result


def verify_sources(root: Path, preparation: dict) -> None:
    for row in preparation['sources']:
        for path_field, digest_field in [('audio','source_audio_sha256'),('transcript','source_docx_sha256')]:
            if sha256(root/row[path_field]) != row[digest_field]:
                raise ValueError(f"Original source changed since preparation: {row[path_field]}")
