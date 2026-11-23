#!/usr/bin/env bash
# Prints the environment details the issue templates ask for. Paste the output into your issue.
#   bash tools/env_report.sh [container-name]
R=$(cd "$(dirname "$0")/.." && pwd); . "$R/launch/image.env"; C=${1:-}
echo '```'
echo "recipe commit: $(git -C "$R" rev-parse --short HEAD 2>/dev/null) $(git -C "$R" status --porcelain 2>/dev/null | grep -q . && echo '(local changes)')"
echo "pinned image:  $IMAGE"
echo "local image:   $(docker image inspect "$IMAGE" -f '{{.Id}}' 2>/dev/null || echo 'not pulled')"
nvidia-smi --query-gpu=index,name,memory.total,power.limit,pcie.link.gen.max,pcie.link.width.current,driver_version --format=csv
nvidia-smi topo -m 2>/dev/null | head -4
echo "kernel: $(uname -r)   docker: $(docker --version 2>/dev/null | cut -d, -f1)"
echo "gpu-unlock module: $(lsmod 2>/dev/null | grep -q gpu_unlock && echo loaded || echo 'not loaded')"
echo "host RAM: $(free -g | awk '/Mem:/{print $2}') GiB"
if [ -z "$C" ]; then C=$(docker ps --format '{{.Names}}' | grep -m1 inkling2); fi
if [ -n "$C" ]; then
  echo "--- container $C"
  docker inspect "$C" -f '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^VLLM_|^HOSTAR' | sort
  docker inspect "$C" -f '{{join .Args " "}}' | tr -s ' ' | fold -w 160
  echo "--- last log lines"
  docker logs --tail 40 "$C" 2>&1 | sed -E 's/^.*\] //' | cut -c1-200
fi
echo '```'
