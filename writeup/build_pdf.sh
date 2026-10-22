#!/bin/zsh
# Render algorithmic-contribution.md -> PDF without LaTeX: python-markdown -> HTML (+writeup.css) -> WeasyPrint.
# Needs: uv/uvx (present), brew pango+cairo (present). Run from anywhere.
set -eu
cd "$(dirname "$0")"
uvx --python 3.14 --with markdown python - <<'PY'
import markdown, pathlib
src = pathlib.Path("algorithmic-contribution.md").read_text()
body = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
css = pathlib.Path("writeup.css").read_text()
pathlib.Path("algorithmic-contribution.html").write_text(
    "<!doctype html><html><head><meta charset='utf-8'><title>Write-up</title>"
    f"<style>{css}</style></head><body>{body}</body></html>")
print("html ok")
PY
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uvx --python 3.14 --from weasyprint weasyprint \
    algorithmic-contribution.html algorithmic-contribution.pdf
ls -la algorithmic-contribution.pdf
