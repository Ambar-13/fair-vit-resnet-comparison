#!/bin/bash
# Run once on the login node: venv, the two pretrained weight files, and the Food-101 tarball (5 GB).
# Nothing here imports torch: the login node caps virtual memory at 1 GB, which is too
# small to load torch's CUDA library, and compute nodes may have no internet. Weights come
# through huggingface_hub into HF_HOME, the dataset through curl; jobs then run offline.
set -e
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.hf_cache"
export HF_HUB_DISABLE_XET=1   # the Rust downloader needs more virtual memory than the login node allows
export PIP_CACHE_DIR="$PWD/.pip-cache"
export TMPDIR="$PWD/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" data

if [ ! -x venv/bin/python ]; then python3 -m venv venv; fi
source venv/bin/activate
pip install --upgrade pip -q
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q -r requirements.txt

python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["timm/vit_small_patch16_224.augreg_in1k", "timm/resnet34.a1_in1k"]:
    print("weights:", snapshot_download(repo_id=repo))
PY

if [ ! -f data/food-101.tar.gz ]; then
    curl -L --retry 5 -o data/food-101.tar.gz https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
fi
echo "85eeb15f3717b99a5da872d97d918f87  data/food-101.tar.gz" | md5sum -c -
# not extracted here: the jobs unpack the tarball to node-local disk themselves
echo "weights and data ready"
