"""Command-line entry point; all private outputs stay inside the workspace."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .common import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['audit','prepare','transcribe','evaluate','review-import'])
    parser.add_argument('--config',default='configs/pilot.json')
    parser.add_argument('--decode',action='store_true',help='Fully decode each audio during audit')
    parser.add_argument('--smoke',action='store_true',help='Unreviewed reference-blind pipeline test; maximum three clips')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--reviews',type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    try:
        cfg,root,output = load_config(args.config)
        if args.command=='audit':
            from .audit import audit
            data = audit(cfg,root,output,args.decode)
            result = {'unique_audio_hours':data['unique_audio_hours'],'files':len(data['audio'])}
        elif args.command=='prepare':
            from .manifest import prepare
            data = prepare(cfg,root,output)
            result = {k:data[k] for k in ['selected_clips','selected_duration_s','candidate_clips','review_status','by_recording']}
        elif args.command=='transcribe':
            from .transcribe import transcribe
            result = transcribe(cfg,root,output,smoke=args.smoke,limit=args.limit,retry_failed=args.retry_failed)
        elif args.command=='evaluate':
            from .evaluate import evaluate
            result = evaluate(output, cfg, root)
        else:
            from .review import import_review
            if args.reviews is None:
                raise ValueError('--reviews CSV path is required')
            result = import_review(output,args.reviews)
        print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        return 0
    except (ValueError,FileNotFoundError,KeyError) as error:
        logging.error('%s',error)
        return 2


if __name__=='__main__':
    sys.exit(main())
