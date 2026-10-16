"""Detection distillation boundaries: dispatch, input protection and failures."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
import torch

import run_tools
from model_compression.core.detection import (
    _check_distillation_architecture, _isolated_distillation_data,
    _prepare_distillation_student, _run_detection_distillation,
)
from model_compression.modules.detection_distillation import detection_distillation_settings, train_detection_distillation
from vtools_ui.webapp import core


class TinyHead(torch.nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.nc = nc
        self.cv3 = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv2d(4, nc, 1))])


class TinyDetectorModel(torch.nn.Module):
    def __init__(self, cfg, ch=3, nc=None, verbose=False):
        super().__init__()
        nc = cfg['nc'] if nc is None else nc
        self.yaml = {**cfg, 'nc': nc}
        self.model = torch.nn.ModuleList([torch.nn.Conv2d(ch, 4, 1), TinyHead(nc)])
        self.names = dict(enumerate(str(index) for index in range(nc)))
        self.args = {'task': 'detect'}


def tiny_detector(names):
    network = TinyDetectorModel({'nc': len(names)})
    network.names = dict(enumerate(names))
    return types.SimpleNamespace(model=network, ckpt={'ema': 'stale', 'model': 'stale', 'optimizer': 'stale'})


class DetectionDistillationTests(unittest.TestCase):
    def dataset(self, root, names, **extra):
        path = root / 'data.yaml'
        path.write_text(yaml.safe_dump({'names': names, **extra}))
        return path

    def test_generic_student_head_is_rebuilt_without_changing_source_or_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.dataset(Path(directory), ['red', 'black'])
            teacher = tiny_detector(['red', 'black'])
            student = tiny_detector([f'coco-{i}' for i in range(80)])
            original = student.model
            original_state = {key: value.clone() for key, value in original.state_dict().items()}
            rng = torch.random.get_rng_state().clone()
            result = _prepare_distillation_student(teacher, student, data, 7)
            self.assertTrue(result['adapted'])
            self.assertEqual((result['source_nc'], result['target_nc']), (80, 2))
            self.assertEqual(student.model.model[-1].cv3[0][-1].out_channels, 2)
            self.assertEqual(student.model.names, {0: 'red', 1: 'black'})
            self.assertTrue(torch.equal(student.model.model[0].weight, original.model[0].weight))
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
            self.assertEqual(original.model[-1].nc, 80)
            for key, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, original_state[key]))
            self.assertNotIn('ema', student.ckpt)
            self.assertNotIn('model', student.ckpt)

    def test_same_count_different_student_class_order_remaps_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.dataset(Path(directory), ['red', 'black'])
            student = tiny_detector(['black', 'red'])
            with torch.no_grad():
                student.model.model[-1].cv3[0][-1].weight[0].fill_(10)
                student.model.model[-1].cv3[0][-1].weight[1].fill_(20)
            _prepare_distillation_student(tiny_detector(['red', 'black']), student, data, 0)
            weights = student.model.model[-1].cv3[0][-1].weight
            self.assertTrue(torch.all(weights[0] == 20))
            self.assertTrue(torch.all(weights[1] == 10))

    def test_aligned_student_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.dataset(Path(directory), ['red', 'black'])
            student = tiny_detector(['red', 'black'])
            original = student.model
            result = _prepare_distillation_student(tiny_detector(['red', 'black']), student, data, 0)
            self.assertFalse(result['adapted'])
            self.assertIs(student.model, original)

    def test_unrelated_classes_keep_fresh_classification_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.dataset(Path(directory), ['red', 'black'])
            student = tiny_detector(['cat', 'dog'])
            with torch.no_grad():
                student.model.model[-1].cv3[0][-1].weight.fill_(999)
            _prepare_distillation_student(tiny_detector(['red', 'black']), student, data, 0)
            self.assertFalse(torch.any(student.model.model[-1].cv3[0][-1].weight == 999))

    def test_unreconstructable_pruned_backbone_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.dataset(Path(directory), ['red', 'black'])
            student = tiny_detector(['generic'])
            student.model.model[0] = torch.nn.Conv2d(3, 5, 1)
            with self.assertRaisesRegex(ValueError, '主干结构与 model.yaml 不一致'):
                _prepare_distillation_student(tiny_detector(['red', 'black']), student, data, 0)

    def test_wrong_teacher_count_or_order_is_rejected_before_student_adaptation(self):
        for names, error in [(['lychee'], '教师与数据集类别数不一致'),
                             (['black', 'red'], '类别名称或顺序不一致')]:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                data = self.dataset(Path(directory), names)
                student = tiny_detector([f'coco-{i}' for i in range(80)])
                original = student.model
                with self.assertRaisesRegex(ValueError, error):
                    _prepare_distillation_student(tiny_detector(['red', 'black']), student, data, 0)
                self.assertIs(student.model, original)

    def test_invalid_dataset_class_metadata_is_rejected(self):
        for names, extra in [({1: 'red'}, {}), (['red', 'red'], {}), (['red', 'black'], {'nc': 80})]:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                data = self.dataset(Path(directory), names, **extra)
                with self.assertRaises(ValueError):
                    _prepare_distillation_student(tiny_detector(['red', 'black']), tiny_detector(['generic']), data, 0)

    def test_prepared_checkpoint_is_used_for_training_and_provenance(self):
        distill = types.ModuleType('ultralytics.nn.distill_model')
        distill.DistillationModel = type('DistillationModel', (), {})
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'ultralytics.nn.distill_model': distill}):
            root = Path(directory)
            data = self.dataset(root, ['red', 'black'])
            teacher_path, student_path = root / 'teacher.pt', root / 'student.pt'
            teacher_path.write_bytes(b'teacher'); student_path.write_bytes(b'student')
            teacher, student = tiny_detector(['red', 'black']), tiny_detector(['generic'])
            saved = {}
            def save(path):
                path.write_bytes(b'adapted')
                saved[path] = student
            student.save = save
            final = root / 'student-best.pt'
            final.write_bytes(b'trained')
            def load(path):
                if path == teacher_path:
                    return teacher
                if path == student_path:
                    return student
                return saved.get(path, student)
            state = types.SimpleNamespace(current=lambda *_args: (None, None),
                                          add_version=unittest.mock.Mock(return_value={'id': 'version'}),
                                          advance=unittest.mock.Mock())
            config = {'dataset': {'data': str(data)}, 'distillation': {
                'teacher_weights': str(teacher_path), 'student_weights': str(student_path), 'imgsz': 64}}
            metrics = {'evaluation_map50': .5, 'evaluation_map50_95': .4}
            with patch('model_compression.core.detection.load_detector', side_effect=load), \
                 patch('model_compression.core.detection._check_distillation_architecture'), \
                 patch('model_compression.core.detection._isolated_distillation_data', return_value=data), \
                 patch('model_compression.core.detection.evaluate_detector', return_value=metrics), \
                 patch('model_compression.modules.detection_distillation.train_detection_distillation',
                       return_value={'best_checkpoint': str(final)}) as train:
                result = _run_detection_distillation(config, state, root / 'run', 'run')
            initial_path = root / 'run/artifacts/distillation/student-initialized.pt'
            self.assertEqual(train.call_args.kwargs['student_weights'], initial_path)
            self.assertEqual(state.add_version.call_args_list[0].kwargs['artifact'], initial_path)
            self.assertTrue(result['metadata']['student_initialization']['adapted'])
            self.assertEqual(student_path.read_bytes(), b'student')

    def test_workbench_request_reaches_effective_backend_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({"project": {"root": "."}, "run": {"root": "runs"},
                                             "model": {"task": "detect"}, "modules": {}}))
            teacher, student = root / "teacher.pt", root / "student.pt"
            request = {"tool_id": "compression", "config_path": str(config), "python_executable": sys.executable,
                       "values": {"task": "detect", "teacher_weights": str(teacher), "student_weights": str(student)},
                       "module_edits": {"distillation.detection": True}}
            _executable, args, _config = core.build_command(request)
            captured = []

            def backend(module_id, effective, _state, _run_dir, _run_id):
                captured.append((module_id, effective["distillation"]))
                return {"module": module_id, "status": "succeeded", "metadata": {}}

            with patch.object(sys, "argv", args), patch("model_compression.core.detection.run_detection_operation", side_effect=backend):
                self.assertEqual(run_tools.main(), 0)
            self.assertEqual(captured, [("distillation.detection", {"teacher_weights": str(teacher), "student_weights": str(student)})])
            effective = json.loads((root / "runs/run1/effective_config.json").read_text())
            self.assertEqual(effective["distillation"]["teacher_weights"], str(teacher))
            self.assertTrue(effective["modules"]["distillation.detection"])

    def test_invalid_training_settings_fail_before_loading_models(self):
        for field, value in (("epochs", 0), ("epochs", True), ("workers", -1), ("dis", 0),
                             ("dis", float("nan")), ("learning_rate", float("inf")),
                             ("imgsz", [64, 96]), ("amp", "true"), ("optimizer", "invalid")):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                detection_distillation_settings({"distillation": {field: value}})

    def test_dataset_mirror_omits_caches_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val"):
                images, labels = root / "images" / split, root / "labels" / split
                images.mkdir(parents=True); labels.mkdir(parents=True)
                (images / "image.jpg").write_bytes(b"image bytes")
                (images / "image.npy").write_bytes(b"existing image cache")
                (labels / "image.txt").write_text("0 .5 .5 .2 .2\n")
                (labels / "labels.cache").write_bytes(b"existing label cache")
            config = root / "data.yaml"
            config.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "object"}}))
            original = config.read_bytes()
            output = _isolated_distillation_data(config, root / "job")
            mirror = Path(yaml.safe_load(output.read_text())["path"])
            self.assertEqual((mirror / "labels/train/image.txt").read_text(), "0 .5 .5 .2 .2\n")
            self.assertFalse(list(mirror.rglob("*.cache")))
            self.assertFalse(list(mirror.rglob("*.npy")))
            (mirror / "labels/train/labels.cache").write_bytes(b"new cache")
            self.assertEqual((root / "labels/train/labels.cache").read_bytes(), b"existing label cache")
            self.assertEqual(config.read_bytes(), original)

    def test_teacher_student_class_mismatch_is_rejected(self):
        def detector(nc):
            return types.SimpleNamespace(model=types.SimpleNamespace(model=[types.SimpleNamespace(nc=nc)]))
        with self.assertRaisesRegex(ValueError, "类别数不一致"):
            _check_distillation_architecture(detector(2), detector(1), None, 64)

    def test_classification_task_records_failure_without_dispatching_detection(self):
        from model_compression import run_model_compression as entry
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = root / "config.yaml"
            config.write_text(yaml.safe_dump({"project": {"root": "."}, "run": {"root": "runs"}, "model": {"task": "classify"}}))
            with patch.object(sys, "argv", ["compression", "--config", str(config), "--only", "distillation.detection"]):
                self.assertEqual(entry.main(), 1)
            report = json.loads((root / "runs/run1/summary.json").read_text())
            self.assertEqual(report["status"], "failed")

    def test_training_without_best_checkpoint_is_failure(self):
        ultralytics = types.ModuleType("ultralytics")
        ultralytics.YOLO = lambda *_args, **_kwargs: types.SimpleNamespace(train=lambda **_kwargs: None, trainer=None)
        cfg = types.ModuleType("ultralytics.cfg"); cfg.DEFAULT_CFG_DICT = {"distill_model": None, "dis": 6.0}
        distill = types.ModuleType("ultralytics.nn.distill_model"); distill.DistillationModel = object
        modules = {"ultralytics": ultralytics, "ultralytics.cfg": cfg, "ultralytics.nn.distill_model": distill}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "未生成 best.pt"):
                train_detection_distillation(student_weights=root / "student.pt", teacher_weights=root / "teacher.pt",
                                             data=root / "data.yaml", config={}, output_dir=root / "output")


if __name__ == "__main__":
    unittest.main()
