#!/usr/bin/env bash
set -euo pipefail

make
./batch_processor -c 256 -w 64 -h 64 -s 42 -o results
cat results/execution_log.txt
head -n 10 results/frame_means.csv