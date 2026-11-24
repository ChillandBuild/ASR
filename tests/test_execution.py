"""Tests for provenance and the inference/scoring execution boundary."""
import json
import tempfile
import unittest
from pathlib import Path

from tanglish_asr.common import load_config, sha256, write_jsonl, read_jsonl
from tanglish_asr.transcribe import inference_selection, validate_prediction_context
from tanglish_asr.evaluate import evaluate_records


class ExecutionTests(unittest.TestCase):
    def test_unreviewed_execution_is_limited_to_three_smoke_clips(self):
        rows = [{'clip_id':str(i),'review_status':'pending','recording_id':str(i%3),
                 'duration_s':10+i} for i in range(5)]
        with self.assertRaises(ValueError):
            inference_selection(rows, False, 3)
        with self.assertRaises(ValueError):
            inference_selection(rows, True, 4)
        self.assertEqual(len(inference_selection(rows, True, 3)), 3)

    def test_smoke_uses_different_recordings(self):
        rows = [{'clip_id':str(i),'review_status':'pending','recording_id':r,
                 'duration_s':d} for i,(r,d) in enumerate([('a',4),('a',5),('b',7),('c',8)])]
        chosen = inference_selection(rows, True, 3)
        self.assertEqual({r['recording_id'] for r in chosen}, {'a','b','c'})

    def test_resume_rejects_changed_configuration_or_clip_hash(self):
        pred = {'config_sha256':'old','clip_audio_sha256':'audio'}
        with self.assertRaises(ValueError):
            validate_prediction_context(pred, 'new', 'audio')
        with self.assertRaises(ValueError):
            validate_prediction_context(pred, 'old', 'different')

    def test_output_cannot_overwrite_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'configs').mkdir()
            cfg = root/'configs/pilot.json'
            cfg.write_text(json.dumps({'output_dir':'Data','local_processing_authorized':True,
                                       'recordings':[]}))
            with self.assertRaises(ValueError):
                load_config(cfg)

    def test_missing_and_failed_predictions_stay_in_sensitivity_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = [{'clip_id':'a','recording_id':'r','reference':'naan varen','speaker_ids':[],
                     'adjudicated_reference':None,'review_status':'approved','reviewer':'human',
                     'reviewed_at':'2026-10-12T00:00:00+05:30','review_notes':'Checked audio.'},
                    {'clip_id':'b','recording_id':'s','reference':'hello','speaker_ids':[],
                     'adjudicated_reference':None,'review_status':'approved','reviewer':'human',
                     'reviewed_at':'2026-10-12T00:00:00+05:30','review_notes':'Checked audio.'}]
            write_jsonl(out/'manifest.jsonl', rows)
            write_jsonl(out/'predictions.jsonl', [{'clip_id':'a','status':'ok','text':'naan varen'}])
            result, _, _, _ = evaluate_records(rows, read_jsonl(out/'predictions.jsonl'))
            self.assertEqual(result['successful_only_raw']['wer'], 0)
            self.assertEqual(result['failures_as_empty_raw']['wer'], 1/3)
            self.assertEqual(result['failed_or_missing_clips'], 1)

    def test_duplicate_predictions_raise_instead_of_silently_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_jsonl(out/'manifest.jsonl', [{'clip_id':'a','review_status':'approved',
                'reviewer':'human','reviewed_at':'2026-10-12T00:00:00+05:30'}])
            write_jsonl(out/'predictions.jsonl', [{'clip_id':'a'},{'clip_id':'a'}])
            with self.assertRaises(ValueError):
                evaluate_records(read_jsonl(out/'manifest.jsonl'), read_jsonl(out/'predictions.jsonl'))


if __name__=='__main__':
    unittest.main()

class ReviewRegressionTests(unittest.TestCase):
    def test_recording_id_cannot_escape_private_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs').mkdir()
            path=root/'configs/pilot.json'
            path.write_text(json.dumps({'output_dir':'private_outputs/run',
                'local_processing_authorized':True,'recordings':[
                    {'id':'../../../Data/overwrite','audio':'Data/a.mp3','transcript':'Data/a.docx'}]}))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_private_output_symlink_escape_rejected(self):
        from tanglish_asr.common import safe_artifact
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'private_outputs/run'; output.mkdir(parents=True)
            data=root/'Data'; data.mkdir()
            (output/'clips').symlink_to(data,target_is_directory=True)
            with self.assertRaises(ValueError):
                safe_artifact(output,'clips/test.wav')

    def test_cached_audio_is_checked_before_resume_skip(self):
        from tanglish_asr.transcribe import verify_inference_audio
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'private_outputs/run'; (output/'clips').mkdir(parents=True)
            clip=output/'clips/a.wav'; clip.write_bytes(b'original')
            row={'clip_id':'a','audio_path':str(clip.relative_to(root)),
                 'clip_audio_sha256':sha256(clip)}
            clip.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_inference_audio([row],root,output)

    def test_unrelated_predictions_rejected_before_scoring(self):
        from tanglish_asr.evaluate import validate_prediction_provenance
        row={'clip_id':'a','clip_audio_sha256':'audio'}
        pred={'clip_id':'a','clip_audio_sha256':'wrong','config_sha256':'cfg',
              'model_sha256':'model','inference_signature':'runtime'}
        with self.assertRaises(ValueError):
            validate_prediction_provenance([row],[pred],'cfg','model','runtime')

    def test_mixed_inference_signatures_cannot_be_combined(self):
        from tanglish_asr.evaluate import validate_prediction_provenance
        rows=[{'clip_id':'a','clip_audio_sha256':'audio'}]
        preds=[{'clip_id':'a','clip_audio_sha256':'audio','config_sha256':'cfg',
                'model_sha256':'model','inference_signature':'other_runtime'}]
        with self.assertRaises(ValueError):
            validate_prediction_provenance(rows,preds,'cfg','model','runtime')

class RootOutputRegressionTests(unittest.TestCase):
    def test_bare_private_outputs_directory_is_not_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs').mkdir()
            config=root/'configs/pilot.json'
            config.write_text(json.dumps({'output_dir':'private_outputs',
                'local_processing_authorized':True,'recordings':[]}))
            with self.assertRaises(ValueError):
                load_config(config)
