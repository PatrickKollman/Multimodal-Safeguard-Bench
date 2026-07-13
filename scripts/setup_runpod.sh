#!/usr/bin/env bash
# RunPod RTX 4090 bootstrap — run once after pod creation.
# Assumes: Ubuntu pod with Python 3.11+, CUDA 12.x, internet access.
# Usage: bash scripts/setup_runpod.sh
set -euo pipefail

REPO_URL="https://github.com/PatrickKollman/Multimodal-Safeguard-Bench.git"
REPO_DIR="/workspace/Multimodal-Safeguard-Bench"

# Direct all HF downloads to the persistent volume, not ephemeral container disk.
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"

echo "=== 1. System packages ==="
apt-get update -qq && apt-get install -y -q git git-lfs python3-pip

echo "=== 2. Clone or update repo ==="
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "=== 3. Install Python deps ==="
pip install -e . --quiet

echo "=== 4. HuggingFace login ==="
# Check existing auth first (respects HF_HOME, HF_TOKEN, HUGGING_FACE_HUB_TOKEN).
# || true inside the subshell prevents set -e from aborting on unauthenticated state.
_hf_user=$(python3 -c "
from huggingface_hub import whoami
try:
    print(whoami()['name'])
except Exception:
    pass
" 2>/dev/null || true)

if [ -n "$_hf_user" ]; then
    echo "  ✓ Already authenticated as: $_hf_user"
elif [ -t 0 ]; then
    echo "  → Paste your HF token when prompted (needs read access to meta-llama + google gated repos)"
    hf auth login
else
    echo "  ✗ Not authenticated and no interactive terminal available."
    echo "    Export HF_TOKEN=hf_... and re-run, or run 'hf auth login' manually first."
    exit 1
fi

echo "=== 5. Verify model access (all 4 models the harness loads) ==="
python3 - <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download
try:
    from huggingface_hub import GatedRepoError
except ImportError:
    from huggingface_hub.utils import GatedRepoError

MODELS = [
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "meta-llama/Llama-Guard-4-12B",
    "google/shieldgemma-2-4b-it",
    "allenai/wildguard",
]

failed = []
for mid in MODELS:
    try:
        hf_hub_download(mid, "config.json")
        print(f"  ✓ {mid}")
    except GatedRepoError:
        print(f"  ✗ {mid}: access denied — request access at https://huggingface.co/{mid}")
        failed.append(mid)
    except Exception as e:
        print(f"  ✗ {mid}: {e}")
        failed.append(mid)

if failed:
    print(f"\n  {len(failed)} model(s) inaccessible. Fix access above, then re-run setup.")
    sys.exit(1)
PYEOF

echo "=== 6. Pin model revision SHAs ==="
echo "  → Run this to get SHAs, then paste into configs/mvp.yaml:"
python3 - <<'EOF'
from huggingface_hub import model_info
models = [
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "meta-llama/Llama-Guard-4-12B",
    "google/shieldgemma-2-4b-it",
    "allenai/wildguard",
]
for mid in models:
    try:
        info = model_info(mid)
        sha = info.sha
        print(f"  {mid}: {sha}")
    except Exception as e:
        print(f"  {mid}: ERROR — {e}")
EOF

echo ""
echo "=== Setup complete ==="
echo "Edit configs/mvp.yaml to fill in the revision SHAs printed above, then run:"
echo "  python -m msbench.run --config configs/mvp.yaml --smoke"
echo "  # Check results, then stop the pod: RunPod dashboard → Stop"
