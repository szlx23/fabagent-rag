from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabagent_rag.cli import main


if __name__ == "__main__":
    # 复用 `rag eval` 的 Click 命令树，但把脚本自身当成一键入口直接映射到 `eval` 子命令。
    main(args=["eval", *sys.argv[1:]], prog_name="rag")
