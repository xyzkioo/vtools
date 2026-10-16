"""Compression modes use the full workbench's existing CLI and isolate inputs."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import run_tools
from model_compression.core.module_selection import COMPRESSION_MODES
from vtools_ui.webapp import core


class CompressionModeTests(unittest.TestCase):
    def build(self, root, mode, *, task='detect', edits=None, frozen=False):
        config = root / 'config.yaml'
        config.write_text(yaml.safe_dump({
            'project': {'root': str(root)}, 'run': {'root': 'runs'},
            'model': {'task': task},
            'modules': {'baseline.evaluate': True, 'compression.prune.structured': True,
                        'compression.prune.unstructured': True},
        }))
        values = {'task': task, 'processing_mode': mode,
                  'resource': str(root / 'stale-public.pt'), 'data': str(root / 'data.yaml'),
                  'train': str(root / 'stale-train'), 'val': str(root / 'stale-val'),
                  'teacher_weights': str(root / 'teacher.pt'),
                  'student_weights': str(root / 'student.pt')}
        request = {'tool_id': 'compression', 'config_path': str(config),
                   'python_executable': sys.executable, 'values': values,
                   'module_edits': edits or {}}
        app_executable = root / 'workbench'
        app_executable.touch()
        with patch.object(core, 'read_settings', return_value={
                'python_executable': sys.executable, 'results_root': str(root / 'results')}), \
                patch.object(sys, 'frozen', frozen, create=True), \
                patch.object(sys, 'executable', str(app_executable) if frozen else sys.executable):
            return core.build_command(request)

    def test_detection_distillation_reaches_backend_without_public_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, args, _ = self.build(root, 'detection_distillation')
            self.assertNotIn('--weights', args)
            self.assertNotIn('--train', args)
            self.assertNotIn('--val', args)
            self.assertIn('--teacher-weights', args)
            self.assertIn('--student-weights', args)
            captured = []

            def execute(module_id, config, *_args):
                captured.append((module_id, config))
                return {'module': module_id, 'status': 'succeeded'}

            with patch.object(sys, 'argv', args), \
                    patch('model_compression.core.detection.run_detection_operation', side_effect=execute):
                self.assertEqual(run_tools.main(), 0)
            self.assertEqual([item[0] for item in captured], ['distillation.detection'])
            effective = captured[0][1]
            self.assertEqual(effective['distillation']['student_weights'], str(root / 'student.pt'))
            self.assertFalse(effective['modules']['compression.prune.structured'])
            report = json.loads((root / 'runs/run1/summary.json').read_text())
            self.assertEqual(report['status'], 'succeeded')

    def test_modes_override_yaml_and_filter_hidden_weight_and_dataset_inputs(self):
        for mode, definition in COMPRESSION_MODES.items():
            for task in definition['tasks']:
                with self.subTest(mode=mode, task=task), tempfile.TemporaryDirectory() as directory:
                    _, args, _ = self.build(Path(directory), mode, task=task)
                    selected = [args[i + 1] for i, arg in enumerate(args) if arg == '--only']
                    self.assertEqual(selected, definition['modules'])
                    self.assertEqual('--weights' in args, definition['public_weights'])
                    self.assertEqual('--teacher-weights' in args, mode == 'detection_distillation')
                    self.assertEqual('--student-weights' in args, mode == 'detection_distillation')
                    self.assertEqual('--data' in args, task == 'detect')
                    self.assertEqual('--train' in args, task == 'classify')
                    self.assertNotIn('--enable', args)

    def test_unsupported_modes_and_conflicting_module_edits_are_rejected(self):
        for mode, task, edits in [('structured', 'classify', {}),
                                  ('quantization', 'detect', {}), ('missing', 'detect', {}),
                                  ('', 'detect', {}),
                                  ('baseline', 'detect', {'compression.prune.structured': False})]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.build(Path(directory), mode, task=task, edits=edits)

    def test_frozen_workbench_keeps_external_model_interpreter_and_writable_run_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable, args, _ = self.build(root, 'detection_distillation', frozen=True)
            self.assertEqual(executable, sys.executable)
            self.assertEqual(Path(args[0]).name, 'run_tools.py')
            self.assertNotIn('--run-script', args)
            output = Path(args[args.index('--run-dir') + 1])
            self.assertEqual(output.parent, root / 'results/compression')


if __name__ == '__main__':
    unittest.main()
