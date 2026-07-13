#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REPO_URL="${MEDICAL_SAM3_REPO_URL:-https://github.com/AIM-Research-Lab/Medical-SAM3.git}"
TARGET_DIR="${MEDICAL_SAM3_REPO_DIR:-$ROOT_DIR/external/Medical-SAM3}"
GIT_REF="${MEDICAL_SAM3_GIT_REF:-main}"
FORCE=0
UPDATE=0

usage() {
  cat <<USAGE
Usage: $0 [--force] [--update]

Clone the official Medical-SAM3 source code into:
  external/Medical-SAM3

Options:
  --force   Delete an existing non-empty target and clone again.
  --update  If the target is already a Git repository, fetch and fast-forward it.
  -h, --help

Environment overrides:
  MEDICAL_SAM3_REPO_URL   Git URL (default: official AIM-Research-Lab repository)
  MEDICAL_SAM3_REPO_DIR   Destination directory
  MEDICAL_SAM3_GIT_REF    Branch or tag (default: main)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --update) UPDATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] git is not installed. Install it first:" >&2
  echo "        Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y git" >&2
  exit 1
fi

mkdir -p "$(dirname "$TARGET_DIR")"

if [[ -d "$TARGET_DIR/.git" ]]; then
  if [[ "$UPDATE" -eq 1 ]]; then
    echo "[INFO] Updating existing Medical-SAM3 repository: $TARGET_DIR"
    git -C "$TARGET_DIR" fetch --depth 1 origin "$GIT_REF"
    git -C "$TARGET_DIR" checkout "$GIT_REF" 2>/dev/null || true
    git -C "$TARGET_DIR" pull --ff-only origin "$GIT_REF"
  else
    echo "[OK] Medical-SAM3 repository already exists: $TARGET_DIR"
    echo "     Use --update to fetch the newest revision or --force to clone again."
  fi
elif [[ -e "$TARGET_DIR" ]] && [[ -n "$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  if [[ "$FORCE" -eq 1 ]]; then
    echo "[WARN] Removing existing target because --force was supplied: $TARGET_DIR"
    rm -rf "$TARGET_DIR"
  else
    echo "[ERROR] Target exists and is not an empty Git repository: $TARGET_DIR" >&2
    echo "        Move it away, or rerun with --force." >&2
    exit 1
  fi
fi

if [[ ! -d "$TARGET_DIR/.git" ]]; then
  echo "[INFO] Cloning Medical-SAM3 source code..."
  echo "       URL: $REPO_URL"
  echo "       Ref: $GIT_REF"
  echo "       Dir: $TARGET_DIR"

  # First try a small shallow clone. If a tag/branch is unavailable, clone the
  # default branch and then try to check out the requested ref.
  if ! git clone --depth 1 --branch "$GIT_REF" "$REPO_URL" "$TARGET_DIR"; then
    echo "[WARN] Could not shallow-clone ref '$GIT_REF'; trying the default branch."
    rm -rf "$TARGET_DIR"
    git clone --depth 1 "$REPO_URL" "$TARGET_DIR"
    if [[ "$GIT_REF" != "main" ]]; then
      git -C "$TARGET_DIR" fetch --depth 1 origin "$GIT_REF"
      git -C "$TARGET_DIR" checkout FETCH_HEAD
    fi
  fi
fi

# The official repository bundles the SAM3 implementation under sam3/.
if [[ ! -d "$TARGET_DIR/sam3" ]]; then
  echo "[WARN] Clone completed, but '$TARGET_DIR/sam3' was not found." >&2
  echo "       The upstream repository layout may have changed." >&2
  echo "       Inspect its README and adjust teacher.builder in the YAML if needed." >&2
else
  echo "[OK] Found bundled SAM3 package: $TARGET_DIR/sam3"
fi

if [[ -f "$TARGET_DIR/requirements.txt" ]]; then
  echo "[INFO] Upstream requirements file detected: $TARGET_DIR/requirements.txt"
  echo "       It is not installed automatically to avoid changing your PyTorch/CUDA stack."
  echo "       Install manually only if Medical-SAM3 reports missing packages:"
  echo "       python -m pip install -r external/Medical-SAM3/requirements.txt"
fi

printf '\n[OK] Medical-SAM3 source is ready at:\n%s\n' "$TARGET_DIR"
printf 'Next: ./scripts/download_medical_sam3_checkpoint.sh\n'
printf 'Then: python scripts/check_medical_sam3_setup.py\n'
