"""Download controls must reach the converter and enforce limits on real HTTP requests."""
import asyncio
import contextlib
import io
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from transform_tools import ndjson_to_yolo as entry
from vtools_ui.webapp import core


class DownloadSettingsTests(unittest.TestCase):
    def test_saved_settings_and_command(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(core, 'SETTINGS_PATH', Path(directory) / 'settings.json'):
            config = {'workers': 2, 'auto_retry': False, 'retries': 5, 'timeout': 90}
            core.save_settings({'ndjson_download': config})
            self.assertEqual(core.read_settings()['ndjson_download'], config)
            _, args, _ = core.build_command({'tool_id': 'transform', 'variant': 'ndjson', 'values': {'input': '/tmp/example.ndjson', 'output': directory}})
            self.assertIn('--no-auto-retry', args)
            self.assertEqual(args[args.index('--workers') + 1], '2')
            self.assertEqual(args[args.index('--timeout') + 1], '90')

    def test_reject_invalid_download_settings(self):
        for config in ({'workers': 0}, {'workers': 1.5}, {'retries': -1}, {'timeout': 0}, {'auto_retry': 'false'}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                core._ndjson_settings(config)


class DownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_disables_retries(self):
        async def converter(source, output, *, download_workers, download_retries, download_timeout, retry_delay):
            return Path(output) / 'data.yaml'
        mock = AsyncMock(spec=converter, wraps=converter)
        mock.__signature__ = inspect.signature(converter)
        with patch.object(entry, '_load_converter', return_value=mock), contextlib.redirect_stdout(io.StringIO()):
            await entry.main(['--input', 'data.ndjson', '--output', '/tmp/yolo', '--workers', '2', '--no-auto-retry'])
        self.assertEqual(mock.await_args.kwargs['download_retries'], 0)
        self.assertEqual(mock.await_args.kwargs['download_workers'], 2)

    async def test_http_concurrency_retry_resume_and_permanent_failure(self):
        from aiohttp import web
        converter = entry._load_converter('/tmp/data.ndjson')
        if 'download_workers' not in inspect.signature(converter).parameters:
            self.skipTest('Requires Ultralytics Fork with download controls')
        counts = {}
        active = peak = 0

        async def image(request):
            nonlocal active, peak
            key = request.match_info['name']
            counts[key] = counts.get(key, 0) + 1
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(.03)
                if key == 'missing':
                    return web.Response(status=404)
                if counts[key] == 1:
                    return web.Response(status=503)
                return web.Response(body=b'complete image payload')
            finally:
                active -= 1

        app = web.Application(); app.router.add_get('/{name}', image)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root / 'data.ndjson'
                def write(names):
                    rows = [{'task': 'detect', 'class_names': {'0': 'fruit'}}]
                    rows += [{'type': 'image', 'file': f'{name}.jpg', 'split': 'train' if i % 2 == 0 else 'val', 'url': f'http://127.0.0.1:{port}/{name}', 'annotations': {}} for i, name in enumerate(names)]
                    source.write_text('\n'.join(json.dumps(row) for row in rows))
                write(['a', 'b', 'c', 'd'])
                options = dict(download_workers=2, download_retries=1, download_timeout=2, retry_delay=.001)
                result = await converter(source, root / 'output', **options)
                self.assertTrue(result.is_file())
                self.assertEqual(peak, 2)
                self.assertEqual(list(counts.values()), [2] * 4)
                # Force conversion past its completion marker, exercising per-file resume.
                result.unlink()
                await converter(source, root / 'output', **options)
                self.assertEqual(list(counts.values()), [2] * 4)
                write(['missing', 'a'])
                with self.assertRaisesRegex(RuntimeError, 'Downloaded 1/2'):
                    await converter(source, root / 'failed', **options)
                self.assertEqual(counts['missing'], 1)
                self.assertFalse(list((root / 'failed').rglob('1.jpg')))
                write(['slow1', 'slow2'])
                with self.assertRaisesRegex(RuntimeError, 'Downloaded 0/2'):
                    await converter(source, root / 'timeout', download_workers=1, download_retries=0,
                                    download_timeout=.001, retry_delay=.001)
                self.assertFalse(list((root / 'timeout').rglob('*.jpg')))
                self.assertFalse(list((root / 'timeout').rglob('data.yaml')))
        finally:
            await runner.cleanup()
