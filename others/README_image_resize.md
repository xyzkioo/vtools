# 图片改分辨率工具

## 入口

```bash
python others/image_resize.py --help
python others/image_resize.py -i /path/to/images -o /path/to/resized --width 1920 --height 1080
```

也可以修改脚本顶部的 `PYCHARM_CONFIG` 后在 PyCharm 直接运行。

`image_resize.py` 支持单张图片或目录批量调整分辨率。默认 `fit` 模式保持宽高比，图片会完整放入目标尺寸框内；`stretch` 强制变成指定宽高；`crop` 保持比例并从中心裁剪到指定尺寸。

统一输出格式时会先检查同名文件冲突，即使指定 `--overwrite` 也不会让两张输入相互覆盖。单张图片编码成功后才替换已有输出，编码失败会保留旧文件。

## 安装依赖

```bash
python -m pip install Pillow
```

## 命令行示例

```bash
python others/image_resize.py -i /path/to/images -o /path/to/resized --width 1920 --height 1080
python others/image_resize.py -i input.jpg --width 800 --height 600 --mode crop --output-format jpg
python others/image_resize.py -i /path/to/images --width 512 --height 512 --mode stretch --recursive --overwrite
```

默认输出到输入目录下的 `resized` 文件夹，不覆盖已有文件。也可以修改脚本顶部的 `PYCHARM_CONFIG` 后直接在 PyCharm 中运行。
