#!/usr/bin/python
"""Test that looped NTSS audio is correctly detected and labeled."""

import numpy as np

def test_looped_detection():
    """Test the looped audio detection threshold."""
    
    test_cases = [
        # (ntss_duration, alignment_duration, expected_looped)
        (4.0, 3.0, True),   # Looped: 4.0/3.0 = 1.33 > 1.05
        (3.2, 3.0, True),   # Slightly extended: 3.2/3.0 = 1.067 > 1.05
        (3.05, 3.0, False), # Nearly same: 3.05/3.0 = 1.017 < 1.05
        (3.0, 3.0, False),  # Exact match: 3.0/3.0 = 1.0 < 1.05
        (2.0, 3.0, False),  # Trimmed: 2.0/3.0 = 0.67 < 1.05
        (6.0, 3.0, True),   # Doubled: 6.0/3.0 = 2.0 > 1.05
    ]
    
    threshold = 1.05
    
    print("Testing looped audio detection threshold:")
    print(f"Threshold: {threshold}x alignment duration\n")
    
    all_passed = True
    for ntss_dur, align_dur, expected_looped in test_cases:
        ratio = ntss_dur / align_dur
        detected_looped = ntss_dur > align_dur * threshold
        
        status = "✓" if detected_looped == expected_looped else "✗"
        all_passed = all_passed and (detected_looped == expected_looped)
        
        print(f"{status} NTSS: {ntss_dur}s, Alignment: {align_dur}s, Ratio: {ratio:.3f}")
        print(f"   Expected: {'LOOPED' if expected_looped else 'NOT_LOOPED'}, "
              f"Got: {'LOOPED' if detected_looped else 'NOT_LOOPED'}")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    return all_passed

if __name__ == '__main__':
    success = test_looped_detection()
    exit(0 if success else 1)
