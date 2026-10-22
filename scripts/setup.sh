#!/usr/bin/env bash
# setup.sh — rebuild this project's working environment on a fresh machine
# (a Claude Code cloud session, a rented box, or a second laptop).
#
# Idempotent: re-running re-uses whatever is already in place.
# It never touches AIcrowd (no key, no submit) and writes only into this repo,
# into --prefix, and into the Hugging Face cache.
#
#   scripts/setup.sh                          # Mac: externals + whest env + torch env + baked/ (10 MLPs)
#   scripts/setup.sh --prefix /opt/arc --data none   # cloud SETUP SCRIPT: envs only, outside the repo
#   scripts/setup.sh --data mini10            # cloud, in-session: fetch baked/ (~1.1 GB, ~1-2 min)
#   scripts/setup.sh --data full              # + materialise the whole mini split (~14 GB disk)
#   scripts/setup.sh --arc-env                # + the full ARC env (torch/mpi4py/quimb: mlp_kprop oracle)
#
# --prefix DIR builds external/ and the venvs in DIR and symlinks them into the repo. A cloud
# session re-clones the repo every time but keeps DIR in its filesystem cache, so the envs survive.
#
# What to run after it: see README.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PREFIX="$REPO"       # where external/ and the venvs physically live
DATA=mini10          # none | mini10 | full
N_MLPS=10
WANT_WHEST=yes
WANT_TORCH=yes
WANT_ARC_ENV=no
RUN_CHECK=yes
REVISION=v2-phase2

# Pins. The whest env is the one we are graded by: a drifted meter invalidates every FLOP
# number in notes/, so these are asserted, not hoped for.
WHESTBENCH=0.16.1
FLOPSCOPE=0.12.1
TORCH=2.8.0
NUMPY=2.3.3          # the ARC/torch chain's numpy
WHEST_NUMPY=2.2.6    # what the starter kit's uv.lock resolves to — pinned so the PyPI
                     # fallback env is comparable with the Mac's, not merely working

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix)      PREFIX="$2"; shift 2 ;;
    --data)        DATA="$2"; shift 2 ;;
    --n-mlps)      N_MLPS="$2"; shift 2 ;;
    --revision)    REVISION="$2"; shift 2 ;;
    --arc-env)     WANT_ARC_ENV=yes; shift ;;
    --no-torch)    WANT_TORCH=no; shift ;;
    --no-whest)    WANT_WHEST=no; shift ;;
    --no-check)    RUN_CHECK=no; shift ;;
    -h|--help)     usage ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"

# A cloud session cannot recover how long its setup script took: the script's own output is not
# kept and the filesystem snapshot normalises every mtime to one stamp. So keep a log with
# wall-clock stamps and elapsed seconds; $PREFIX survives into the session.
LOG="$PREFIX/setup.log"
printf '\n########## %s  %s\n' "$(date -u '+%Y-%m-%d %H:%M:%SZ')" "$0 $*" >> "$LOG"
stamp() { printf '[%s +%3ds]' "$(date -u '+%H:%M:%S')" "$SECONDS"; }
log()  { printf '\n=== %s %s\n' "$(stamp)" "$*" | tee -a "$LOG"; }
note() { printf '    %s\n' "$*" | tee -a "$LOG"; }
warn() { printf '    WARNING: %s\n' "$*" | tee -a "$LOG" >&2; }
die()  { printf '\nFAILED %s %s\n' "$(stamp)" "$*" | tee -a "$LOG" >&2; exit 1; }

# Make $REPO/<name> resolve to $PREFIX/<name>: a symlink when they differ, nothing when they
# do not. An existing real directory in the repo always wins — never clobber a working tree.
link_in() {
  local name="$1" target="$PREFIX/$1" link="$REPO/$1"
  [ "$PREFIX" = "$REPO" ] && return 0
  if [ -L "$link" ]; then
    [ "$(readlink "$link")" = "$target" ] || { rm "$link"; ln -s "$target" "$link"; }
  elif [ -e "$link" ]; then
    warn "$link exists and is not a symlink — leaving it alone, $target is unused"
    return 1
  else
    ln -s "$target" "$link"
  fi
  note "$name -> $target"
}

# ---------------------------------------------------------------- 0. preflight
log "0. preflight"
note "repo:    $REPO"
note "prefix:  $PREFIX"
note "os:      $(uname -s) $(uname -m)"
if [ "$(uname -s)" = "Linux" ]; then
  note "cpus:    $(nproc)"
  note "mem:     $(awk '/MemTotal/ {printf "%.1f GB", $2/1048576}' /proc/meminfo)"
else
  note "cpus:    $(sysctl -n hw.ncpu)"
  note "mem:     $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
fi
note "disk:    $(df -h "$PREFIX" | awk 'NR==2 {print $4" free"}')"
for t in git curl; do command -v "$t" >/dev/null || die "$t is not installed"; done

# ---------------------------------------------------------------------- 1. uv
log "1. uv"
if ! command -v uv >/dev/null; then
  note "installing uv ..."                      # pre-installed in cloud sessions; this is for other boxes
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || die "uv still not on PATH (add \$HOME/.local/bin)"
note "uv:      $(uv --version)   (developed against uv 0.11.14)"

# ------------------------------------------------------- 2. vendored externals
# Public repos, read-only. In a cloud session GitHub goes through a proxy scoped to the
# attached repository, so these clones may be refused — that is survivable: the whest env
# then comes from PyPI (step 3), and only the mlp_kprop oracle (step 5) truly needs a clone.
log "2. external/ (pinned clones — see scripts/external_pins.txt)"
mkdir -p "$PREFIX/external"
link_in external || true
while read -r dir url commit; do
  case "$dir" in ''|\#*) continue ;; esac
  d="$PREFIX/external/$dir"
  if [ ! -d "$d/.git" ]; then
    note "cloning $dir ..."
    git clone --quiet --filter=blob:none "$url" "$d" || { warn "clone of $dir failed (GitHub proxy?)"; continue; }
  fi
  if [ "$(git -C "$d" rev-parse HEAD)" != "$commit" ]; then
    git -C "$d" fetch --quiet origin "$commit" 2>/dev/null || git -C "$d" fetch --quiet origin || true
    git -C "$d" checkout --quiet --detach "$commit" || warn "could not pin $dir to ${commit:0:12}"
  fi
  note "$dir @ $(git -C "$d" rev-parse --short HEAD)"
done < scripts/external_pins.txt

# ------------------------------------------------------ 3. whest (grader) env
# Two ways to the same CLI: the starter kit's uv project (what the Mac uses), or the same
# pinned wheels straight from PyPI. `whest` is a console script of whestbench either way.
# Whichever is built, .venv-whest/bin/whest is the one command form that works everywhere.
if [ "$WANT_WHEST" = yes ]; then
  log "3. whest env (the CLI we are graded by: validate / run / package)"
  if [ -d "$PREFIX/external/whest-starterkit/.git" ]; then
    note "via the starter kit's uv project"
    uv sync --quiet --project "$PREFIX/external/whest-starterkit"
    WHEST_VENV="$PREFIX/external/whest-starterkit/.venv"
  else
    note "starter kit not cloned — installing the same pins from PyPI"
    [ -d "$PREFIX/.venv-whest-pypi" ] || uv venv --quiet --python 3.12 "$PREFIX/.venv-whest-pypi"
    uv pip install --quiet --python "$PREFIX/.venv-whest-pypi" \
      "whestbench==$WHESTBENCH" "flopscope==$FLOPSCOPE" "numpy==$WHEST_NUMPY"
    WHEST_VENV="$PREFIX/.venv-whest-pypi"
  fi
  rm -f "$REPO/.venv-whest"
  ln -s "$WHEST_VENV" "$REPO/.venv-whest"
  note ".venv-whest -> $WHEST_VENV"
  "$REPO/.venv-whest/bin/python" - "$WHESTBENCH" "$FLOPSCOPE" <<'PY'
import importlib.metadata as md, sys
want = {"whestbench": sys.argv[1], "flopscope": sys.argv[2]}
bad = []
for pkg, exp in want.items():
    got = md.version(pkg)
    print(f"    {pkg:12s} {got}" + ("" if got == exp else f"   <-- EXPECTED {exp}"))
    if got != exp:
        bad.append(pkg)
print(f"    {'numpy':12s} {md.version('numpy')}   (mac dev env: 2.2.6)")
if bad:
    raise SystemExit("version drift in " + ", ".join(bad) + ": the FLOP meter or the grader moved. "
                     "Do not trust any score from this env until notes/SPEC.md is re-checked.")
PY
else
  log "3. whest env — skipped (--no-whest)"
fi

# ------------------------------------------- 4. torch env (research chain only)
# chain.py / ablate.py need nothing but torch + numpy; the heavy ARC env
# (mpi4py, quimb, jupyter) is only needed for the mlp_kprop reference oracle.
if [ "$WANT_TORCH" = yes ]; then
  log "4. torch env (.venv-torch: chain.py / ablate.py)"
  [ -d "$PREFIX/.venv-torch" ] || uv venv --quiet --python 3.12 "$PREFIX/.venv-torch"
  if [ "$(uname -s)" = "Linux" ]; then
    note "CPU-only torch (the CUDA-bundled wheel is ~6.4 GB against ~1 GB, and no GPU is attached)"
    # Both CPU arms resolve wheels from download-r2.pytorch.org: the index on
    # download.pytorch.org answers 200 but every href points there. With that host missing from a
    # cloud environment's allow-list they fail with "CONNECT tunnel failed, response 403" and leave
    # an empty venv (measured, cloud session #1). PyPI is the arm that always works.
    uv pip install --quiet --python "$PREFIX/.venv-torch" --torch-backend=cpu "torch==$TORCH" "numpy==$NUMPY" \
      || uv pip install --quiet --python "$PREFIX/.venv-torch" \
           --index-url https://download.pytorch.org/whl/cpu "torch==$TORCH" "numpy==$NUMPY" \
      || { warn "the CPU wheel index is unreachable — add download-r2.pytorch.org (or *.pytorch.org)"
           warn "to the environment's allowed domains. Falling back to PyPI: same torch $TORCH, but"
           warn "the CUDA-bundled build, ~6.4 GB of a 30 GB disk."
           uv pip install --quiet --python "$PREFIX/.venv-torch" "torch==$TORCH" "numpy==$NUMPY"; }
  else
    uv pip install --quiet --python "$PREFIX/.venv-torch" "torch==$TORCH" "numpy==$NUMPY"
  fi
  # Only publish the venv once it actually imports: a half-built one masquerading as ready is
  # what made cloud session #1 look healthy while step 4 was dead.
  "$PREFIX/.venv-torch/bin/python" -c "import torch, numpy; print(f'    torch {torch.__version__}  numpy {numpy.__version__}  threads {torch.get_num_threads()}')" \
    || die "torch did not install into $PREFIX/.venv-torch"
  link_in .venv-torch || true
else
  log "4. torch env — skipped (--no-torch)"
fi

# ----------------------------------------------- 5. full ARC env (mlp_kprop)
if [ "$WANT_ARC_ENV" = yes ]; then
  log "5. ARC env (mlp_cumulant_propagation: the offline kprop oracle)"
  note "heavy (~4 GB) and mpi4py may need system MPI headers; skip unless you need the oracle"
  [ -d "$PREFIX/external/mlp_cumulant_propagation/.git" ] || die "the ARC repo is not cloned (step 2 failed)"
  uv sync --quiet --project "$PREFIX/external/mlp_cumulant_propagation" \
    || die "uv sync of the ARC env failed — on Debian/Ubuntu try: apt-get install -y libopenmpi-dev"
  note "ok"
else
  log "5. ARC env — skipped (pass --arc-env if you need mlp_kprop cross-checks)"
fi

# --------------------------------------------------------------------- 6. data
# Deliberately NOT part of a cloud setup script: ~1.1 GB would eat the 5-minute budget,
# and baked/ lives inside the repo, which a cloud session re-clones anyway. Run this
# step in-session, once, when a torch-chain run actually needs it.
case "$DATA" in
  none)
    log "6. data — skipped (--data none).  In a cloud session run, when needed:"
    note "scripts/setup.sh --data mini10 --no-whest --no-torch --no-check"
    ;;
  mini10)
    log "6. data: baked/ = first $N_MLPS MLPs of the mini split"
    [ -x "$REPO/.venv-whest/bin/python" ] || die "--data needs the whest env (run without --no-whest first)"
    "$REPO/.venv-whest/bin/python" scripts/fetch_baked.py --n-mlps "$N_MLPS" --revision "$REVISION"
    ;;
  full)
    log "6. data: materialise the whole mini split (100 MLPs; ~7 GB download, ~14 GB disk)"
    [ -x "$REPO/.venv-whest/bin/python" ] || die "--data needs the whest env (run without --no-whest first)"
    "$REPO/.venv-whest/bin/python" - "$REVISION" <<'PY'
import sys
import whestbench as wb
ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision=sys.argv[1], split="mini")
print(f"    materialised: {len(ds)} MLPs, columns {ds.column_names}")
PY
    "$REPO/.venv-whest/bin/python" scripts/fetch_baked.py --n-mlps "$N_MLPS" --revision "$REVISION"
    ;;
  *) die "--data must be one of: none | mini10 | full" ;;
esac

# ------------------------------------------------------------------- 7. check
if [ "$RUN_CHECK" = yes ] && [ "$WANT_WHEST" = yes ]; then
  log "7. contract check (no dataset needed, ~1 min)"
  "$REPO/.venv-whest/bin/whest" validate --estimator src/estimator_k3z.py
fi

log "done"
cat <<'OUT'
    Next (see README.md for the full list, and for what a cloud number proves):

    # accuracy on the 100 public MLPs — residual gate OFF, because residual time
    # is a property of THIS machine and must be re-measured on the Mac before shipping
    .venv-whest/bin/whest run --estimator src/estimator_k3z.py \
        --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
        --runner local --max-threads 4 --format json \
        --no-residual-wall-time-limit --wall-time-limit 600 > /tmp/run.json

    # torch research chain (must print 3.7533e-08 on 3 MLPs)
    cd experiments/2026-09-24_k3-chain && PYTHONPATH=. ../../.venv-torch/bin/python ablate.py k4memo 3
OUT
