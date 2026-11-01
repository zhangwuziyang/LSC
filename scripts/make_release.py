"""Build the anonymous public release of this project.

    python scripts/make_release.py OUT_DIR

Copies an allowlist of paths (code, experiments and their results, the pre-registration,
the LaTeX paper) into OUT_DIR with a release README and licence, then scans every copied
text file for company, personal and session identifiers and fails if any is found.
Internal planning documents (docs/ other than the pre-registration) are never copied.
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["gpsdid", "sim", "agent", "cases", "tests", "experiments", "pyproject.toml", ".gitignore",
           ".env.example", "docs/E2V2_PLAN.md", "paper/latex/main.tex", "paper/latex/refs.bib",
           "paper/latex/figures"]
SKIP = re.compile(r"(__pycache__|\.egg-info|\.pyc$|^\.env$|\.pytest_cache)")
# Hard failures: anything that names the company, the author, or this development session.
FORBIDDEN = re.compile(r"alibaba|阿里|速卖通|aliexpress|1\.4 ?million|140 ?万|实习|internship|"
                       r"zhangwuziyang|@gmail|claude\.ai/code|session_0|Claude-Session", re.I)
RELEASE = ROOT / "scripts" / "release"


def copy(src, dst):
    if src.is_dir():
        for p in src.rglob("*"):
            rel = p.relative_to(src)
            if p.is_file() and not any(SKIP.search(part) for part in rel.parts):
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst / rel)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    out = Path(sys.argv[1]).resolve()
    if out.exists():
        sys.exit(f"{out} exists; choose a new directory")
    for item in INCLUDE:
        copy(ROOT / item, out / item)
    for name in ("README.md", "LICENSE"):
        shutil.copy2(RELEASE / name, out / name)
    hits = []
    for p in out.rglob("*"):
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, ValueError):
                continue
            for m in FORBIDDEN.finditer(text):
                hits.append(f"{p.relative_to(out)}: …{text[max(0, m.start() - 40):m.end() + 40]!r}…")
    if hits:
        print("\n".join(hits[:50]))
        sys.exit(f"{len(hits)} forbidden matches; release not usable")
    n = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"release written to {out}: {n} files, scan clean")


if __name__ == "__main__":
    main()
