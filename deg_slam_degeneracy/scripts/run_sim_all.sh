#!/usr/bin/env bash
# Run the full synthetic sweep: 3 scenarios x 2 lidar ranges.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../src"

for s in garage corridor glass; do
  for r in 20 8; do
    echo "=== ${s} (max_range=${r}) ==="
    python3 run_sim_experiment.py --scenario "$s" --max-range "$r" --laps 1 --out ../results \
      2>&1 | tail -4
  done
done
echo "ALL DONE"
