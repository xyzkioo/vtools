# 图片、视频与文件名工具

[返回模块导航](../README.md)

提供不加载模型的数据准备功能：批量缩放、视频抽帧及图片与标注同步改名。

## 入口

从仓库根目录执行，`python` 指向已准备好的目标环境；系统仅提供 `python3` 时可替换命令名。

```bash
python others/image_resize/image_resize.py --help
python others/video_frame_extractor/video_frame_extractor.py --help
python others/filename_transform/image_filename_converter.py --help
```

## 输入与功能

| 工具 | 功能 | 依赖 |
| --- | --- | --- |
| [图片缩放](../../others/image_resize/README_image_resize.md) | 单图/目录；fit 保持比例、stretch 拉伸、crop 中心裁剪 | Pillow |
| [视频抽帧](../../others/video_frame_extractor/README_video_frame_extractor.md) | 单视频/目录；抽帧间隔、起止帧、JPG/PNG | OpenCV |
| [文件名转换](../../others/filename_transform/README_image_filename_converter.md) | 普通图片、YOLO 同名标签、COCO file_name 同步改名；模板与 CSV 清单 | Python 标准库 |

参数由各 CLI 提供，相对路径按当前工作目录解析。缩放和抽帧使用独立输出目录；文件名转换默认只预览，显式 `--apply` 才修改输入。

## 输出

缩放默认在输入目录下生成 `resized/`。抽帧写入指定输出目录，多视频按视频名分目录，帧编号对应原始视频帧。文件名转换可用 `--plan` 导出 CSV，COCO 修改前保留 JSON 备份。

## 使用边界

图片缩放不自动同步 YOLO/COCO 标签；裁剪或拉伸后须检查标注坐标。抽取每帧可能产生大量文件。改名前检查预览和引用关系，工具只维护它支持的标签格式。

详细说明：[原模块文档](../../others/filename_transform/README_image_filename_converter.md)。
