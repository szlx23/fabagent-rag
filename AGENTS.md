# Project Codex Notes

This project uses the `rag` conda environment for development and verification.

When running Python, pip, tests, CLI commands, or the FastAPI server in this
repository, prefer executables from:

```bash
/home/szlx23/conda/envs/rag/bin/
```

Examples:

```bash
/home/szlx23/conda/envs/rag/bin/python -m pip install -e .
/home/szlx23/conda/envs/rag/bin/python -m uvicorn fabagent_rag.api:app --reload
/home/szlx23/conda/envs/rag/bin/rag ask "OPC有哪些类型？"
```
