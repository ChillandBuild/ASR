"""Regression tests for real dataset hazards and metric validity."""
import tempfile
import unittest
from pathlib import Path

from tanglish_asr.manifest import extract_segments, validate_manifest, select_pilot
from tanglish_asr.evaluate import edit_counts, score_pairs, script_stats, check_review
from tanglish_asr.normalize import cleanup


class SegmentTests(unittest.TestCase):
    def test_continuations_and_spoken_clock_are_not_new_segments(self):
        segments, rejected = extract_segments(
            ['Transcript', '[00:00] Hello,', '3:30 meeting dhaan',
             '[00:10] Next', '[00:20] End'], 25, False)
        self.assertEqual(segments[0]['reference'], 'Hello, 3:30 meeting dhaan')
        self.assertEqual((segments[0]['start_s'], segments[0]['end_s']), (0, 10))
        self.assertEqual(len(segments), 2)
        self.assertEqual(rejected[-1]['reason'], 'unverified_final_endpoint')

    def test_glued_line_timestamp_preserves_lexical_digits(self):
        segments, _ = extract_segments(['L1 00:02naan 2026 la',
                                        'L2 00:12next', 'L3 00:22end'], 25, True)
        self.assertEqual(segments[0]['reference'], 'naan 2026 la')
        self.assertEqual(segments[0]['start_s'], 2)

    def test_section_resets_do_not_join_recordings(self):
        segments, _ = extract_segments(
            ['Transcript', '[00:00] A', '[00:10] B',
             'Section 2 - Transcript B: Before.txt', 'Transcript',
             '[00:00] C', '[00:10] D', 'Section 3 - Transcript C: Negative.txt'],
            20, False, 'Section 2 - Transcript B: Before.txt',
            'Section 3 - Transcript C: Negative.txt')
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]['reference'], 'C')

    def test_duplicate_anchors_and_long_intervals_are_quarantined(self):
        segments, rejected = extract_segments(
            ['[00:00] A', '[00:10] B', '[00:10] C', '[00:20] D',
             '[01:00] E', '[01:10] F'], 75, False)
        self.assertEqual([(s['start_s'], s['end_s']) for s in segments], [(60, 70)])
        self.assertIn('duplicate_anchor', {s['reason'] for s in rejected})
        self.assertIn('interval_exceeds_limit', {s['reason'] for s in rejected})

    def test_invalid_timestamp_is_not_treated_as_reference_text(self):
        with self.assertRaises(ValueError):
            extract_segments(['[00:00] A', '[00:99] B'], 120, False)

    def test_negative_or_out_of_bounds_segments_rejected(self):
        row = {'clip_id': 'a', 'recording_id': 'r', 'start_s': 2,
               'end_s': 20, 'recording_duration_s': 10, 'source_audio_sha256': 'x'}
        with self.assertRaises(ValueError):
            validate_manifest([row])

    def test_overlapping_duplicate_audio_cannot_hide_behind_recording_name(self):
        rows = [{'clip_id': 'a', 'recording_id': 'r', 'start_s': 0, 'end_s': 10,
                 'recording_duration_s': 30, 'source_audio_sha256': 'same'},
                {'clip_id': 'b', 'recording_id': 'alias', 'start_s': 5, 'end_s': 15,
                 'recording_duration_s': 30, 'source_audio_sha256': 'same'}]
        with self.assertRaises(ValueError):
            validate_manifest(rows)

    def test_pilot_sampling_is_reproducible_and_balances_sources(self):
        rows = [{'clip_id': f'{r}{i}', 'recording_id': r, 'start_s': i * 10,
                 'end_s': (i+1)*10, 'duration_s': 10} for r in ['a', 'b'] for i in range(20)]
        selected = select_pilot(rows, 100, 7)
        self.assertEqual(sum(s['duration_s'] for s in selected), 100)
        self.assertEqual([s['clip_id'] for s in selected],
                         [s['clip_id'] for s in select_pilot(rows, 100, 7)])
        self.assertEqual(sum(s['recording_id']=='a' for s in selected), 5)
        self.assertEqual(sum(s['recording_id']=='b' for s in selected), 5)


class MetricTests(unittest.TestCase):
    def test_raw_scoring_keeps_case_punctuation_and_spaces(self):
        score = score_pairs([('Hello, naan', 'hello naan')])
        self.assertEqual(score['wer'], .5)
        self.assertEqual(score['cer'], 2/11)

    def test_aggregate_weights_edit_counts_not_clip_percentages(self):
        score = score_pairs([('a', ''), ('b c d e', 'b c d e')])
        self.assertEqual(score['wer'], .2)
        self.assertEqual(score['word_errors']['deletions'], 1)

    def test_insert_delete_substitute_counts(self):
        self.assertEqual(edit_counts('a b'.split(), 'a c d'.split()),
                         {'substitutions': 1, 'deletions': 0, 'insertions': 1, 'reference_units': 2})

    def test_empty_reference_rejected_empty_prediction_scored(self):
        with self.assertRaises(ValueError):
            score_pairs([('', 'hello')])
        self.assertEqual(score_pairs([('naan varen', '')])['wer'], 1)

    def test_empty_output_cannot_pass_roman_compliance(self):
        self.assertFalse(script_stats('')['roman_compliant'])
        self.assertTrue(script_stats('Hello naan')['roman_compliant'])
        self.assertFalse(script_stats('வணக்கம் hello')['roman_compliant'])

    def test_cleanup_does_not_merge_spelling_or_negation(self):
        self.assertEqual(cleanup('  Naan   vara-maatten! '), 'naan vara maatten')
        self.assertNotEqual(cleanup('amma'), cleanup('aama'))

    def test_human_review_cannot_be_fabricated_by_flag_only(self):
        row = {'clip_id': 'a', 'review_status': 'approved', 'reviewer': '',
               'reviewed_at': '', 'review_notes': ''}
        with self.assertRaises(ValueError):
            check_review([row])

    def test_unreviewed_reference_blocks_accuracy(self):
        with self.assertRaises(ValueError):
            check_review([{'clip_id':'a','review_status':'pending'}])


if __name__ == '__main__':
    unittest.main()
