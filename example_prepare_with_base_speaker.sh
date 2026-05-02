#!/bin/bash
#
# Example: Generate overlap dataset with speaker 84 as base speaker
# This demonstrates the new BASE_SPEAKER feature
#

# Make a temporary copy of the script
cp prepare_overlap_dataset.sh prepare_overlap_dataset_84.sh

# Modify the BASE_SPEAKER setting
sed -i 's/BASE_SPEAKER=""/BASE_SPEAKER="84"/' prepare_overlap_dataset_84.sh

# Optionally change output directory name
sed -i 's/feature_dir_name=overlap_50pct_50/feature_dir_name=overlap_50pct_50_base84/' prepare_overlap_dataset_84.sh

echo "================================================"
echo "Generating overlap dataset with speaker 84 as base"
echo "================================================"
echo ""
echo "This will create samples where speaker 84 is the"
echo "target speaker in ALL generated overlaps."
echo ""

# Run the modified script
bash prepare_overlap_dataset_84.sh 0

echo ""
echo "Done! Check data/overlap_50pct_50_base84/ for results"
echo ""
echo "To use different settings, edit prepare_overlap_dataset.sh:"
echo "  - Change BASE_SPEAKER=\"84\" to your desired speaker"
echo "  - Or set BASE_SPEAKER=\"\" for random speaker selection"
