#!/bin/bash

# Example: Generate overlap samples with speaker 84 as the base speaker in all overlaps
# This means speaker 84 will be present in every generated sample, with other speakers mixed on top

python generate_non_target_overlap_utterances.py \
    --base-speaker 84 \
    --other-speakers 174 251 422 652 1272 1462 1673 1919 1988 2035 \
    --n-overlapping-speakers 2 \
    --n-samples 10 \
    --output-dir test_outputs/overlap_with_84_base

echo ""
echo "Generated overlap samples with speaker 84 as base speaker!"
echo "Each sample contains speaker 84 mixed with one other speaker."
