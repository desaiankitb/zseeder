#!/usr/bin/env bash

# Downloader that fetches the full validation split without requiring Python packages.

set -euo pipefail

REPO_ID="chinmays18/medical-prescription-dataset"
API_BASE="https://huggingface.co/api/datasets/${REPO_ID}/tree/main"
RESOLVE_BASE="https://huggingface.co/datasets/${REPO_ID}/resolve/main"
DATASET_DIR="data/validation_dataset/test"
IMAGES_DIR="${DATASET_DIR}/images"
ANNOTATIONS_DIR="${DATASET_DIR}/annotations"
REMOTE_IMAGES_ROOT="test/images"
REMOTE_ANN_ROOT="test/annotations"

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: '$1' is required but not installed."
        exit 1
    fi
}

require_cmd curl

echo "=========================================="
echo "Downloading dataset from HuggingFace"
echo "=========================================="
echo "Repository : ${REPO_ID}"
echo "Destination: ${DATASET_DIR}"
echo ""

mkdir -p "${IMAGES_DIR}" "${ANNOTATIONS_DIR}"
echo "✓ Created directories"

download_file() {
    local url="$1"
    local output="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$output"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$url" -O "$output"
    else
        echo "Error: Neither curl nor wget is available."
        exit 1
    fi
}

parse_paths_without_jq() {
    local json_file="$1"

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Error: jq is missing and python3 is not available to parse JSON."
        exit 1
    fi

    python3 - "$json_file" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
for entry in data:
    if entry.get("type") == "file":
        print(entry["path"])
PY
}

fetch_file_list() {
    local remote_path="$1"
    local tmp_file
    tmp_file=$(mktemp)

    if ! curl -fsSL "${API_BASE}/${remote_path}?recursive=1&expand=1" -o "$tmp_file"; then
        echo "✗ Failed to read file list for '${remote_path}'."
        rm -f "$tmp_file"
        exit 1
    fi

    if command -v jq >/dev/null 2>&1; then
        jq -r '.[] | select(.type=="file") | .path' "$tmp_file"
    else
        parse_paths_without_jq "$tmp_file"
    fi

    rm -f "$tmp_file"
}

read_into_array() {
    local __target="$1"
    eval "$__target=()"
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        eval "$__target+=(\"\$line\")"
    done
}

download_group() {
    local label="$1"
    local remote_root="$2"
    local dest_dir="$3"
    shift 3
    local files=("$@")
    local total="${#files[@]}"
    local downloaded=0

    echo ""
    echo "→ Downloading ${label} (${total} files)"

    for remote_path in "${files[@]}"; do
        local relative="${remote_path#${remote_root}/}"
        local destination="${dest_dir}/${relative}"

        mkdir -p "$(dirname "$destination")"
        if [[ -f "$destination" ]]; then
            continue
        fi

        local url="${RESOLVE_BASE}/${remote_path}"
        if download_file "$url" "$destination"; then
            ((downloaded++))
        else
            echo "  ✗ Failed: ${relative}"
        fi
    done

    echo "✓ ${label} ready (${downloaded}/${total} new files)"
}

echo "→ Fetching file list from HuggingFace..."
read_into_array "IMAGE_FILES" < <(fetch_file_list "$REMOTE_IMAGES_ROOT")
read_into_array "ANNOTATION_FILES" < <(fetch_file_list "$REMOTE_ANN_ROOT")

if [[ ${#IMAGE_FILES[@]} -eq 0 || ${#ANNOTATION_FILES[@]} -eq 0 ]]; then
    echo "✗ Did not receive file list. Please verify the repository paths."
    exit 1
fi

download_group "images" "$REMOTE_IMAGES_ROOT" "$IMAGES_DIR" "${IMAGE_FILES[@]}"
download_group "annotations" "$REMOTE_ANN_ROOT" "$ANNOTATIONS_DIR" "${ANNOTATION_FILES[@]}"

images_present=$(find "$IMAGES_DIR" -type f | wc -l | tr -d '[:space:]')
annotations_present=$(find "$ANNOTATIONS_DIR" -type f -name "*.txt" | wc -l | tr -d '[:space:]')

echo ""
echo "=========================================="
echo "Download complete"
echo "=========================================="
echo "Images present     : ${images_present}"
echo "Annotations present: ${annotations_present}"
echo ""
echo "Files stored under: ${DATASET_DIR}"
echo "You can now run: python validate_dataset.py"
