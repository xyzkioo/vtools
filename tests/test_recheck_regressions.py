import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from model_visualization.run_visualization import _logical_image_id
from model_diagnostics.diagnostics.engine import _ultralytics_image_id
from model_diagnostics.diagnostics.engine import _find_image


class ReviewRegressions(unittest.TestCase):
    def test_image_identity_across_subsets_and_layouts(self):
        for folder in ('images', 'photos'):
            root = Path('/dataset')
            for suffix in ('.jpg', '.png'):
                image = root / folder / ('a' + suffix)
                single = _logical_image_id(image, image, root)
                directory = _logical_image_id(image, image.parent, root)
                diagnostic = _ultralytics_image_id(image, root, set())
                self.assertEqual(single, directory)
                self.assertEqual(single, diagnostic)
                self.assertEqual(single, folder + '/a' + suffix)

    def test_custom_layout_requires_root(self):
        with self.assertRaises(ValueError):
            _logical_image_id(Path('/dataset/photos/a.jpg'), Path('/dataset/photos'))

    def test_image_lookup_accepts_canonical_id_with_extension(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'images' / 'val' / 'a.jpg'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'image')
            self.assertEqual(_find_image(root, 'images/val/a.jpg'), image)

    def test_page_edits_commit_in_navigation_order(self):
        # Execute the actual non-Qt state methods with widget value substitutes.
        tree = ast.parse(Path('vtools_ui/app.py').read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ConfigEditorDialog')
        methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in {'_set', '_parse_yaml', '_commit_config_tab'}]
        cls = ast.ClassDef(name='Editor', bases=[], keywords=[], body=methods, decorator_list=[])
        namespace = {'copy': copy}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<editor>', 'exec'), namespace)
        editor = namespace['Editor']()
        editor.data = {'device': 'cpu'}
        editor.parse_error = None
        editor._field_value = lambda widget, kind: widget
        editor._active_config_tab = 0
        editor.field_widgets = {'device': ('cuda:0', 'text')}
        editor._field_initial = {'device': 'cpu'}
        editor._commit_config_tab()
        editor._active_config_tab = 1
        editor.all_field_widgets = {'device': ('auto', 'text')}
        editor._all_initial = {'device': 'cuda:0'}
        editor._commit_config_tab()
        self.assertEqual(editor.data['device'], 'auto')
        editor._active_config_tab = 2
        editor.editor = SimpleNamespace(toPlainText=lambda: 'device: cpu')
        editor._commit_config_tab()
        self.assertEqual(editor.data['device'], 'cpu')

    def test_switch_config_replaces_hidden_modules(self):
        import tempfile
        tree = ast.parse(Path('vtools_ui/app.py').read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ToolPage')
        method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == '_load_module_defaults')
        cls = ast.ClassDef(name='Page', bases=[], keywords=[], body=[method], decorator_list=[])
        namespace = {'copy': copy, 'Path': Path}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<page>', 'exec'), namespace)
        page = namespace['Page']()
        page.spec = SimpleNamespace(key='diagnostics')
        page.module_buttons = [SimpleNamespace(setChecked=lambda value: None)]
        page._module_ids = lambda: ['diagnostics.missed']
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.yaml'
            page.config_edit = SimpleNamespace(text=lambda: str(path))
            path.write_text('modules:\n  diagnostics.missed: true\n  predict.generate: false\n')
            page._load_module_defaults()
            self.assertFalse(page._configured_modules['predict.generate'])
            path.write_text('modules:\n  diagnostics.missed: false\n  predict.generate: true\n')
            page._load_module_defaults()
            self.assertTrue(page._configured_modules['predict.generate'])
            self.assertFalse(page._module_selection_touched)
