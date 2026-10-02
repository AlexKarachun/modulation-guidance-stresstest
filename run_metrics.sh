#!/bin/bash
# Waits for generation (main.py) to finish, then computes all Table-2 metrics on the GPU.
# Usage (from anywhere, e.g. in tmux): bash run_metrics.sh
set -eo pipefail
cd "$(dirname "$0")"
PY=/venv/metrics/bin/python
IMG=generations/coco5000

while pgrep -f '^(/opt/miniforge3/bin/)?python3? main.py' >/dev/null; do sleep 60; done
n=$(ls $IMG/*.png | wc -l)
echo "$(date -u) main.py finished, $n pngs"
[ "$n" -eq 25000 ] || { echo "expected 25000 pngs, got $n -- not starting"; exit 1; }

echo "$(date -u) light metrics"; $PY metrics.py light 2>&1 | grep --line-buffered -v -i warn
echo "$(date -u) hpsv3";         $PY metrics.py hpsv3 2>&1 | grep --line-buffered -v -i warn
echo "$(date -u) analyze";       $PY metrics.py analyze
echo "$(date -u) ALL METRICS DONE"
