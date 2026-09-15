# 视频解帧工具

## 入口

```bash
python others/video_frame_extractor.py --help
python others/video_frame_extractor.py --input /path/to/video.mp4 --output-dir /path/to/frames
```

也可以修改脚本顶部的 `PYCHARM_CONFIG` 后在 PyCharm 直接运行。

`video_frame_extractor.py` 将视频保存成图片，支持：

- `every_n=1`：保存每一帧。
- `every_n=N`：从起始帧开始，每隔 N 帧保存一帧。例如 `N=10` 保存第 0、10、20……帧。
- 处理单个视频或目录中的多个视频。
- JPG / PNG 输出、起止帧范围、避免覆盖已有图片。

## 安装依赖

```bash
python -m pip install opencv-python
```

## 命令行示例

保存每一帧：

```bash
python others/video_frame_extractor.py \
  --input /path/to/video.mp4 \
  --output-dir /path/to/frames
```

每隔 10 帧保存一帧：

```bash
python others/video_frame_extractor.py \
  -i /path/to/video.mp4 \
  -o /path/to/frames \
  --every-n 10
```

只截取第 100 到 500 帧，并每隔 5 帧保存：

```bash
python others/video_frame_extractor.py \
  -i /path/to/video.mp4 \
  -o /path/to/frames \
  --start-frame 100 \
  --end-frame 500 \
  --every-n 5 \
  --extension png
```

处理目录下所有视频：

```bash
python others/video_frame_extractor.py \
  -i /path/to/videos \
  -o /path/to/frames \
  --recursive \
  --every-n 10
```

批量处理时会自动在输出目录下按视频名建立子目录。默认不覆盖已有帧，如需覆盖请加 `--overwrite`。

## PyCharm 运行

打开脚本顶部的 `PYCHARM_CONFIG`，修改 `input`、`output_dir` 和 `every_n` 后直接运行：

```python
PYCHARM_CONFIG = {
    "input": r"/path/to/video.mp4",
    "output_dir": r"/path/to/frames",
    "every_n": 8,       # 默认每隔 8 帧；1=每一帧
    "start_frame": 0,
    "end_frame": None,
    "extension": ".jpg",
    "overwrite": False,
}
```

帧文件名示例：`frame_00000000.jpg`、`frame_00000010.jpg`。编号对应原视频帧编号，便于和视频时间位置对应。
