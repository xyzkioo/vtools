import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from model_visualization.adapters.ultralytics import UltralyticsAdapter


class Head(torch.nn.Module):
    nc = 1
    end2end = True
    def forward(self, tensor):
        return {'one2one': {'scores': tensor, 'boxes': tensor}}
    def _inference(self, branch):
        return torch.tensor([[[1.], [2.], [4.], [5.], [0.8]]])
    def get_topk_index(self, scores, limit):
        return scores, torch.zeros((1, 1, 1), dtype=torch.long), torch.zeros((1, 1), dtype=torch.long)


class StageExport(unittest.TestCase):
    def test_written_records_use_logical_id_and_original_probability(self):
        adapter = object.__new__(UltralyticsAdapter)
        adapter.model = torch.nn.Sequential(Head())
        adapter.head_name = "0"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = adapter.trace_stage(torch.zeros(1), {'stage_trace': {'save_raw': False}}, run_dir=root,
                image_stem='safe_output_hash', image_id='images/a.jpg',
                transform_meta=SimpleNamespace(original_width=20, original_height=20, source_path='/dataset/images/a.jpg'))
            self.assertEqual(result['status'], 'ok')
            self.assertAlmostEqual(result['raw_rows'][0]['class_scores'][0], 0.8)
            for path in (root/'canonical').glob('*.json'):
                record = json.loads(path.read_text())['records'][0]
                self.assertEqual(record['image_id'], 'images/a.jpg')
                self.assertEqual(record['predictions'][0]['image_id'], 'images/a.jpg')
            stage = json.loads((root/'stage_trace/safe_output_hash/final_detections.json').read_text())
            self.assertEqual(stage['image_id'], 'images/a.jpg')

    def test_failed_topk_is_not_reported_as_empty_success(self):
        from unittest.mock import patch
        adapter = object.__new__(UltralyticsAdapter)
        adapter.model = torch.nn.Sequential(Head())
        adapter.head_name = '0'
        with patch.object(adapter.head, 'get_topk_index', side_effect=RuntimeError('broken topk')):
            result = adapter.trace_stage(torch.zeros(1), {})
        self.assertEqual(result['status'], 'unsupported')
        self.assertIn('broken topk', result['reason'])
        self.assertNotIn('canonical_final_record', result)
