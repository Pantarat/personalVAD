#!/usr/bin/env python3
"""
Test script to verify overlap percentage is respected.
Reads generated samples and calculates actual overlap percentage.
"""
import os
import sys

def parse_labels_file(text_path):
    """Parse the text file with labels and timestamps."""
    results = []
    
    with open(text_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            utt_id = parts[0]
            labels_str = parts[1]
            timestamps = [float(t) for t in parts[2:]]
            labels = labels_str.split(',')
            
            # Calculate overlap statistics
            tss_duration = 0.0
            overlap_duration = 0.0
            
            for i, label in enumerate(labels):
                if i < len(timestamps):
                    segment_start = 0.0 if i == 0 else timestamps[i-1]
                    segment_end = timestamps[i]
                    segment_duration = segment_end - segment_start
                    
                    if 'TSS' in label:
                        tss_duration += segment_duration
                        if 'NTSS' in label:  # Overlap
                            overlap_duration += segment_duration
            
            overlap_pct = 0.0
            if tss_duration > 0:
                overlap_pct = (overlap_duration / tss_duration) * 100
            
            results.append({
                'utt_id': utt_id,
                'tss_duration': tss_duration,
                'overlap_duration': overlap_duration,
                'overlap_pct': overlap_pct
            })
    
    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_overlap_percentage.py <overlap_dir>")
        print("Example: python test_overlap_percentage.py data/overlap")
        sys.exit(1)
    
    overlap_dir = sys.argv[1]
    text_path = os.path.join(overlap_dir, 'text')
    
    if not os.path.exists(text_path):
        print(f"Error: {text_path} not found")
        sys.exit(1)
    
    print(f"Analyzing overlap percentages from: {text_path}")
    print("=" * 70)
    
    results = parse_labels_file(text_path)
    
    if not results:
        print("No samples found!")
        sys.exit(1)
    
    # Calculate statistics
    overlap_pcts = [r['overlap_pct'] for r in results]
    avg_overlap = sum(overlap_pcts) / len(overlap_pcts)
    max_overlap = max(overlap_pcts)
    min_overlap = min(overlap_pcts)
    
    print(f"\nTotal samples: {len(results)}")
    print(f"Average overlap: {avg_overlap:.2f}%")
    print(f"Min overlap: {min_overlap:.2f}%")
    print(f"Max overlap: {max_overlap:.2f}%")
    print("=" * 70)
    
    # Show first 10 samples
    print("\nFirst 10 samples:")
    for i, r in enumerate(results[:10]):
        print(f"{i+1:2d}. {r['utt_id'][:50]:50s} | TSS: {r['tss_duration']:5.2f}s | Overlap: {r['overlap_duration']:5.2f}s ({r['overlap_pct']:5.2f}%)")
    
    # Check if overlap percentage is close to zero for 0% overlap dataset
    if max_overlap > 1.0:
        print(f"\n⚠️  WARNING: Found samples with overlap > 1% (max: {max_overlap:.2f}%)")
        print("    This suggests overlap percentage is not being respected!")
        
        # Show problematic samples
        print("\n    Samples with highest overlap:")
        sorted_results = sorted(results, key=lambda x: x['overlap_pct'], reverse=True)
        for r in sorted_results[:5]:
            print(f"    - {r['utt_id'][:50]:50s} | Overlap: {r['overlap_pct']:5.2f}%")
    else:
        print(f"\n✓ All samples have overlap < 1% - overlap percentage is respected!")
