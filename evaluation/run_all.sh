#!/bin/sh
# Runs the four tasks (from the repository root).
set -e
export PYTHONPATH=evaluation
for t in task1_coverage task2_agreement task3_overlap task4_detection; do
  echo "=== $t"; python evaluation/$t.py
done
