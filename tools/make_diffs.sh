#!/usr/bin/env bash
# Regenerate patches/diffs/*.diff: every overlay mounted by launch/serve.sh against the stock file in the pinned image.
# Run after changing any file in patches/ (CONTRIBUTING.md asks for it in every PR that touches a patch).
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); . "$R/launch/image.env"; cd "$R"
PY=/usr/local/lib/python3.14/dist-packages
cid=$(docker create "$IMAGE"); tmp=$(mktemp -d); trap 'docker rm "$cid" >/dev/null; rm -rf "$tmp"' EXIT
mkdir -p patches/diffs
grep -oE '\$P/[a-z0-9_]+\.py:\$[A-Z]+[^:]*' launch/serve.sh | while IFS=: read -r src dst; do
  f=${src#\$P/}
  d=$(echo "$dst" | sed "s|\$VM|$PY/vllm/models/inkling2|; s|\$TR|$PY/triton|; s|\$VF|$PY/vllm/model_executor/layers/fused_moe|; s|\$VL|$PY/vllm/model_executor/layers|; s|\$VB|$PY/vllm/v1/attention/backends|; s|\$VD|$PY/vllm/distributed/device_communicators|; s|\$PY|$PY|")
  if [ "$f" = dspark_boot.py ]; then d=$PY/vllm/models/inkling2/nvidia/dspark.py; fi   # new module derived from dspark.py
  if docker cp "$cid:$d" "$tmp/$f" >/dev/null 2>&1; then
    diff -u --label "a/${d#$PY/}" --label "b/${d#$PY/}" "$tmp/$f" "patches/$f" > "patches/diffs/$f.diff" || true
    echo "$f: $(grep -c '^[+-][^+-]' "patches/diffs/$f.diff") changed lines"
  else
    rm -f "patches/diffs/$f.diff"; echo "$f: new file (no stock counterpart)"
  fi
done
