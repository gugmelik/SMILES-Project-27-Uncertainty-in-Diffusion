#!/usr/bin/env bash
# Download ImageNet-1k validation set only from Kaggle (~6.2 GB).
#
# Source dataset: https://www.kaggle.com/datasets/titericz/imagenet1k-val
# Layout after extract: imagenet-val/<wnid>/*.JPEG  (1000 class folders)
#
# Prerequisites:
#   1. Kaggle account + API token at ~/.kaggle/kaggle.json
#      (Account → Settings → Create New Token)
#   2. Accept the dataset ToS in the browser (open the link above once)
#   3. kaggle CLI installed:  pip install kaggle
#
# Usage:
#   bash scripts/download_imagenet_val_kaggle.sh
#   bash scripts/download_imagenet_val_kaggle.sh /path/to/datasets/imagenet
#
# Then point the notebook at:
#   IMAGENET_VAL_ROOT = Path("/path/to/datasets/imagenet/val")

set -euo pipefail

OUT_ROOT="${1:-$PWD/data/imagenet}"
DATASET="titericz/imagenet1k-val"
TMP_DIR="${OUT_ROOT}/_download"
VAL_DIR="${OUT_ROOT}/val"

echo "==> Output root: ${OUT_ROOT}"
mkdir -p "${TMP_DIR}" "${OUT_ROOT}"

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "ERROR: missing ${HOME}/.kaggle/kaggle.json"
  echo "Download an API token from https://www.kaggle.com/settings and place it there."
  echo "Then:  chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi
chmod 600 "${HOME}/.kaggle/kaggle.json" 2>/dev/null || true

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not found. Install with:  pip install kaggle"
  exit 1
fi

if [[ -d "${VAL_DIR}" ]] && [[ "$(find "${VAL_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)" -ge 1000 ]]; then
  echo "==> Already present: ${VAL_DIR} (1000 class folders)."
  echo "    Skipping download. Delete that folder to re-download."
  exit 0
fi

echo "==> Downloading ${DATASET} ..."
echo "    If this fails with 403, open the dataset page in a browser and Accept rules:"
echo "    https://www.kaggle.com/datasets/${DATASET}"
kaggle datasets download -d "${DATASET}" -p "${TMP_DIR}" --force

ZIP_FILE="$(find "${TMP_DIR}" -maxdepth 1 -type f \( -name '*.zip' -o -name '*.7z' \) | head -n 1)"
if [[ -z "${ZIP_FILE}" ]]; then
  echo "ERROR: no archive found in ${TMP_DIR}"
  ls -la "${TMP_DIR}"
  exit 1
fi

echo "==> Extracting ${ZIP_FILE} ..."
EXTRACT_DIR="${TMP_DIR}/extracted"
mkdir -p "${EXTRACT_DIR}"
unzip -q -o "${ZIP_FILE}" -d "${EXTRACT_DIR}"

# Dataset ships as imagenet-val/<wnid>/... — normalize to val/<wnid>/
SRC="$(find "${EXTRACT_DIR}" -type d -name 'n01440764' | head -n 1)"
if [[ -z "${SRC}" ]]; then
  echo "ERROR: could not find class folders (e.g. n01440764) after extract"
  find "${EXTRACT_DIR}" -maxdepth 3 -type d | head -50
  exit 1
fi
SRC_PARENT="$(dirname "${SRC}")"

echo "==> Organizing into ${VAL_DIR}"
rm -rf "${VAL_DIR}"
mkdir -p "${OUT_ROOT}"
mv "${SRC_PARENT}" "${VAL_DIR}"

N_CLASSES="$(find "${VAL_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
N_IMAGES="$(find "${VAL_DIR}" -type f \( -iname '*.JPEG' -o -iname '*.jpg' -o -iname '*.png' \) | wc -l)"
echo "==> Done."
echo "    classes : ${N_CLASSES}"
echo "    images  : ${N_IMAGES}"
echo "    path    : ${VAL_DIR}"
echo
echo "Use in the notebook:"
echo "  IMAGENET_VAL_ROOT = Path(\"${VAL_DIR}\")"

# Optional cleanup (keeps disk free; comment out if you want to keep the zip)
rm -rf "${TMP_DIR}"
echo "==> Cleaned temporary download files."
