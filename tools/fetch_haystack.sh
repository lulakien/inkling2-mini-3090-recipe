#!/usr/bin/env bash
# needle.py builds its long prompt from Python source files. Copy vLLM's model sources out of the pinned image
# into tests/data/haystack (44 MB, not committed) so every run uses the same text.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); . "$R/launch/image.env"
rm -rf "$R/tests/data/haystack"; mkdir -p "$R/tests/data/haystack"
cid=$(docker create "$IMAGE"); trap 'docker rm "$cid" >/dev/null' EXIT
docker cp "$cid:/usr/local/lib/python3.14/dist-packages/vllm/models/." "$R/tests/data/haystack/"
echo "haystack: $(find "$R/tests/data/haystack" -name '*.py' | wc -l) files"
