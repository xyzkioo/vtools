import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
import yaml
from vtools_ui.app import QApplication, ConfigEditorDialog, ResultsPage, ToolPage, ToolSpec, _ResultScanWorker
from PySide6.QtWidgets import QToolButton


class UIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_tabs_and_save_keep_latest_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('benchmark:\n  device: cpu\n')
            dialog = ConfigEditorDialog(path, 'diagnostics')
            dialog.tabs.setCurrentIndex(0)
            dialog.field_widgets['benchmark.device'][0].setCurrentText('cuda:0')
            dialog.tabs.setCurrentIndex(1)
            self.assertEqual(dialog.all_field_widgets['benchmark.device'][0].text(), 'cuda:0')
            dialog.all_field_widgets['benchmark.device'][0].setText('auto')
            dialog.tabs.setCurrentIndex(2)
            self.assertEqual(yaml.safe_load(dialog.editor.toPlainText())['benchmark']['device'], 'auto')
            dialog.editor.setPlainText('benchmark:\n  device: cpu\nextra: [1, 2]\n')
            dialog.tabs.setCurrentIndex(0)
            self.assertEqual(dialog.field_widgets['benchmark.device'][0].currentText(), 'cpu')
            dialog._save()
            self.assertEqual(yaml.safe_load(path.read_text()), {'benchmark': {'device': 'cpu'}, 'extra': [1, 2]})
            dialog.deleteLater()

    def test_diagnostics_graphical_fields_match_named_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text(
                'mode: ultralytics_model\n'
                'predictions_file:\n  predictions: predictions.json\n'
                'ultralytics_model:\n  weights: model.pt\n  task: detect\n'
                'custom_adapter:\n  adapter: adapter.py\n',
                encoding='utf-8',
            )
            dialog = ConfigEditorDialog(path, 'diagnostics')
            self.assertNotIn('mode_a.predictions', dialog.field_widgets)
            self.assertNotIn('mode_b.weights', dialog.field_widgets)
            self.assertEqual(dialog.field_widgets['mode'][0].currentText(), 'ultralytics_model')
            self.assertEqual(dialog.field_widgets['ultralytics_model.weights'][0].text(), 'model.pt')
            dialog.tabs.setCurrentIndex(0)
            dialog.field_widgets['mode'][0].setCurrentText('predictions_file')
            dialog._save()
            saved = yaml.safe_load(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['mode'], 'predictions_file')
            self.assertEqual(saved['predictions_file']['predictions'], 'predictions.json')
            dialog.deleteLater()

    def test_tool_page_has_single_config_editor_button(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text('mode: ultralytics_model\n', encoding='utf-8')
            page = ToolPage(ToolSpec('diagnostics', 'test', '', 'diagnostics', path))
            self.assertEqual(page.findChildren(QToolButton), [])
            page.deleteLater()

    def test_result_scan_includes_updated_output_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [
                root / 'runs' / 'run1' / 'summary.json',
                root / 'runs' / 'run1' / 'raw_data' / 'details.csv',
                root / 'runs' / 'run1' / 'artifacts' / 'quantization' / 'model.pt',
                root / 'runs' / 'run1' / 'images' / 'case.png',
                root / 'runs' / 'run1' / 'canonical' / 'final_img.json',
                root / 'runs' / 'run1' / 'index.html',
            ]
            for file in files:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b'0')
            stale_registry = root / 'store' / 'registry.json'
            stale_registry.parent.mkdir(parents=True, exist_ok=True)
            stale_registry.write_bytes(b'{}')
            found = []
            worker = _ResultScanWorker(
                root,
                root,
                ResultsPage._SKIP_DIRS,
                ResultsPage._IMAGE_SUFFIXES | ResultsPage._TEXT_SUFFIXES | ResultsPage._MODEL_SUFFIXES,
            )
            worker.finished.connect(lambda _root, result: found.append(result))
            worker.run()
            self.assertEqual(set(Path(item) for item in found[0]), set(files))

    def test_manual_path_change_and_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.yaml'
            path.write_text('diagnostics: {}\n')
            page = ToolPage(ToolSpec('diagnostics', 'test', '', 'diagnostics', path))
            calls = []
            page.run_requested.connect(lambda *args: calls.append(args))
            page._run()
            self.assertEqual(len(calls), 1)
            self.assertNotIn('--only', calls[0][2])
            other = Path(tmp) / 'other.yaml'
            other.write_text('modules:\n  predict.generate: true\n  diagnostics.missed: true\n')
            page.config_edit.setText(str(other))
            self.assertTrue(page._configured_modules['predict.generate'])
            page.module_buttons[1].setChecked(True)
            page._run()
            self.assertIn('predict.generate', calls[-1][2])
            page.deleteLater()

    def test_classification_config_preserves_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'classification.yaml'
            path.write_text('model:\n  task: classify\nmodules:\n  compression.quantize.dynamic_int8: true\n')
            page = ToolPage(ToolSpec('compression', 'test', '', 'compression', path))
            calls = []
            page.run_requested.connect(lambda *args: calls.append(args))
            self.assertEqual(page.compression_task.currentIndex(), 1)
            self.assertTrue(page.module_buttons[1].isEnabled())
            self.assertTrue(page.module_buttons[1].isChecked())
            page._run()
            self.assertEqual(calls[0][2][calls[0][2].index('--task') + 1], 'classify')
            page.deleteLater()

    def test_empty_benchmark_selection_does_not_restore_defaults(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'benchmark.yaml'
            path.write_text('modules:\n  speed.pytorch_call: true\n')
            page = ToolPage(ToolSpec('benchmark', 'test', '', 'pytorch', path))
            calls = []
            page.run_requested.connect(lambda *args: calls.append(args))
            page.module_buttons[0].setChecked(False)
            with patch('vtools_ui.app.QMessageBox.warning') as warning:
                page._run()
                warning.assert_called_once()
            self.assertEqual(calls, [])
            page.deleteLater()

    def test_checkpoint_only_tensorrt_does_not_enable_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'benchmark.yaml'
            path.write_text('modules:\n  checkpoint.inspect: true\n  speed.tensorrt_call: true\n')
            page = ToolPage(ToolSpec('benchmark', 'test', '', 'pytorch', path))
            calls = []
            page.run_requested.connect(lambda *args: calls.append(args))
            page.backend_combo.setCurrentText('TensorRT 测速')
            page.module_buttons[1].setChecked(False)
            page._run()
            self.assertIn('checkpoint.inspect', calls[0][2])
            self.assertNotIn('build.tensorrt', calls[0][2])
            self.assertNotIn('export.onnx', calls[0][2])
            page.deleteLater()
