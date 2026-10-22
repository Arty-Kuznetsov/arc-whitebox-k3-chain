#!/bin/zsh
# Assemble write-up v4 from its section files and render it to PDF (python-markdown -> HTML + writeup.css -> WeasyPrint).
set -eu
cd "$(dirname "$0")"
uvx --python 3.14 --with markdown python - <<'PY'
import markdown, pathlib, re
parts = []
s57 = pathlib.Path("v4-s5-s7.md").read_text()
i7 = s57.index("## 8. Limitations")
for f in ("v4-s0-summary.md", "v4-s1-setting.md", "v4-s2-error.md", "v4-s3-method.md", "v4-s4-cost.md"):
    parts.append(pathlib.Path(f).read_text())
parts.append(s57[:i7])
parts.append(pathlib.Path("v4-s6-boundary.md").read_text())
parts.append(pathlib.Path("v4-s7-order.md").read_text())
parts.append(s57[i7:])
parts.append(pathlib.Path("v4-s8-s9.md").read_text())
parts.append(pathlib.Path("v4-appendix.md").read_text())
parts.append("""## References and credits

- Wu et al., *Estimating the expected activations of random neural networks* (arXiv:2605.05179), and the MIT reference
  implementation `mlp_cumulant_propagation` — the SIMPLE and AUGMENT algorithms this chain is built on.
- 504aldo, AIcrowd forum topic 18218 — an independent catalogue of the same walls (cited in §6.2); their post also
  named the κ₄ → κ₃ channel (§3.3). No code of theirs was used.
- jamesrahenry, AIcrowd forum topics 18097 / 18157 — the coherence mechanism, first published in Phase 1 (§6.3).
- hyojun_kwon, AIcrowd forum topic 18154 (Phase 1) — accuracy against stability of the iterated closure, the
  precursor of the consistency principle (§3.5).
- mliston, AIcrowd forum topic 18170 — the idea of an offline-trained corrector of the chain's own moments (§3.7).
- il.miscusi, AIcrowd forum topic 18248, and kai_leal_gonzalez, topic 18246 (both 2026-10-15) — the scale mode of a
  positively homogeneous network and a conditional skew identity, found independently of our §6.4 (concurrent work).
""")
src = "\n\n".join(parts)
# python-markdown needs a blank line before a list that follows a paragraph line; and no TeX-style subscripts
lines, outl = src.split("\n"), []
islist = lambda t: bool(re.match(r"^(\s*)([-*] |\d+\. )", t))
for k, ln in enumerate(lines):
    if islist(ln) and k > 0 and lines[k - 1].strip() and not islist(lines[k - 1]) and not lines[k - 1].startswith((">", "|", " ")):
        outl.append("")
    outl.append(ln)
src = "\n".join(outl).replace("W_{l+1}", "W(l+1)").replace("W_l,", "W(l),")
pathlib.Path("v4.md").write_text(src)
body = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
body = re.sub(r"<p><em>([^<]*)</em></p>", r'<p class="note"><em>\1</em></p>', body)  # italic-only paragraphs = notes
css = pathlib.Path("writeup.css").read_text()
pathlib.Path("v4.html").write_text("<!doctype html><html><head><meta charset='utf-8'><title>Write-up v4</title>"
                                   f"<style>{css}</style></head><body>{body}</body></html>")
print("v4.md + v4.html:", len(src.split()), "words")
PY
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uvx --python 3.14 --from weasyprint weasyprint v4.html v4.pdf
ls -la v4.pdf
