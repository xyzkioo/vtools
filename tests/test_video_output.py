import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from others.video_frame_extractor import collect_videos, extract_video


class VideoOutput(unittest.TestCase):
    def test_single_unsupported_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.touch()
            with self.assertRaisesRegex(ValueError, "不支持的视频格式"):
                collect_videos(path, recursive=False)

    def test_collision_is_detected_before_decoding_or_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frame_00000008.jpg").write_bytes(b"existing")
            capture = Mock()
            capture.isOpened.return_value = True
            capture.get.side_effect = [30.0, 16]
            with patch("cv2.VideoCapture", return_value=capture):
                with self.assertRaises(FileExistsError):
                    extract_video(root / "video.mp4", root, 8, 0, None, ".jpg", 95, 3, False)
            capture.read.assert_not_called()
            capture.release.assert_called_once()

    def test_zero_decoded_target_frames_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = Mock()
            capture.isOpened.return_value = True
            capture.get.side_effect = [30.0, 10]
            capture.read.return_value = (False, None)
            with patch("cv2.VideoCapture", return_value=capture):
                with self.assertRaises(OSError):
                    extract_video(root / "video.mp4", root, 1, 0, None, ".jpg", 95, 3, False)
            capture.release.assert_called_once()

    def test_failed_frame_write_preserves_previous_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); target=root/'frame_00000000.jpg'; target.write_bytes(b'old frame')
            capture=Mock(); capture.isOpened.return_value=True; capture.get.return_value=1; capture.read.return_value=(True, object())
            def fail(path, *args):
                Path(path).write_bytes(b'partial')
                return False
            with patch('cv2.VideoCapture', return_value=capture), patch('cv2.imwrite', side_effect=fail):
                with self.assertRaises(OSError):
                    extract_video(root/'video.mp4', root, 1, 0, 0, '.jpg', 95, 3, True)
            self.assertEqual(target.read_bytes(), b'old frame')
            self.assertEqual(list(root.iterdir()), [target])
            capture.release.assert_called_once()
