import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="快速测试 memoryos VideoConverter（基于 VideoRAG 的真实视频理解）"
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="待处理视频文件路径，例如 /root/repo/VideoRAG/VideoRAG-algorithm/files/BigBuckBunny_320x180.mp4",
        default="/root/repo/uni-mem/files/BigBuckBunny_320x180.mp4"
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("./videorag-workdir"),
        help="VideoRAG 的工作目录，缓存 KV / 向量库等文件",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="可选：直接向 VideoRAG 提问并返回额外总结 chunk",
    )
    parser.add_argument(
        "--auto-summary",
        action="store_true",
        help="如果未提供 question，启用后会使用 VideoConverter 的默认总结问题",
    )
    parser.add_argument(
        "--debug-caption",
        action="store_true",
        help="调试时跳过加载 MiniCPM caption 模型，加快 query 阶段（insert 仍需模型）",
    )
    parser.add_argument(
        "--deepseek-key",
        type=str,
        default=None,
        help="DeepSeek API Key（可选，若已在环境变量中配置则无需传入）",
    )
    parser.add_argument(
        "--siliconflow-key",
        type=str,
        default=None,
        help="硅基流动 API Key，用于 bge-m3 embedding（可选）",
    )
    return parser.parse_args()


def progress_callback(progress: float, message: str) -> None:
    pct = f"{progress * 100:5.1f}%"
    print(f"[{pct}] {message}")


def main() -> None:
    args = parse_args()

    if args.deepseek_key:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_key
    if args.siliconflow_key:
        os.environ["SILICONFLOW_API_KEY"] = args.siliconflow_key

    if not args.video.exists():
        print(f"视频文件不存在：{args.video}", file=sys.stderr)
        sys.exit(1)

    args.working_dir.mkdir(parents=True, exist_ok=True)

    from multimodal.converters.video_converter import VideoConverter

    converter = VideoConverter(
        progress_callback=progress_callback,
        working_dir=str(args.working_dir),
    )

    convert_kwargs = {
        "auto_summary": args.auto_summary,
        "debug_caption": args.debug_caption,
    }
    if args.question:
        convert_kwargs["question"] = args.question

    result = converter.convert(
        str(args.video),
        source_type="file_path",
        **convert_kwargs,
    )

    print("\n=== Conversion Result ===")
    print(f"Status: {result.status}")
    if result.error:
        print(f"Error: {result.error}")

    print("\nMetadata:")
    print(json.dumps(result.metadata, ensure_ascii=False, indent=2))

    print(f"\nChunks ({len(result.chunks)} total):")
    for chunk in result.chunks:
        meta = json.dumps(chunk.metadata, ensure_ascii=False)
        preview = chunk.text.strip().replace("\n", " ")[:160]
        print(f"- idx={chunk.chunk_index} meta={meta}")
        print(f"  text={preview}{'...' if len(chunk.text) > 160 else ''}")


if __name__ == "__main__":
    main()

