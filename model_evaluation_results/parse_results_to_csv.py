#!/usr/bin/env python3
"""Parse baseline_eval_results.txt and generate a CSV for Google Sheets import."""

import re
import csv

def parse_results(input_file, output_file):
    """Parse the results file and write to CSV."""
    
    results = []
    current_model = None
    current_data = {}
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect model name (lines ending with .pt)
        if line.endswith('.pt'):
            # Save previous model if exists
            if current_model and current_data:
                results.append(current_data)
            
            # Start new model (remove "models/" prefix if present)
            current_model = line.replace('models/', '')
            current_data = {'model': current_model}
            i += 1
            continue
        
        # Parse AP scores (array of 3 values)
        if line.startswith('[') and line.endswith(']'):
            ap_match = re.findall(r'[\d.]+', line)
            if len(ap_match) == 3:
                current_data['AP_class0'] = float(ap_match[0])
                current_data['AP_class1'] = float(ap_match[1])
                current_data['AP_class2'] = float(ap_match[2])
            i += 1
            continue
        
        # Parse mAP
        if line.startswith('mAP:'):
            map_val = re.search(r'[\d.]+', line)
            if map_val:
                current_data['mAP'] = float(map_val.group())
            i += 1
            continue
        
        # Parse confusion matrix (3 lines of 3 values each)
        if line == 'confusion':
            i += 1
            confusion = []
            for row_idx in range(3):
                if i < len(lines):
                    row_line = lines[i].strip()
                    row_match = re.findall(r'[\d.]+', row_line)
                    if len(row_match) == 3:
                        confusion.extend([float(x) for x in row_match])
                    i += 1
            if len(confusion) == 9:
                for idx, val in enumerate(confusion):
                    current_data[f'conf_{idx//3}_{idx%3}'] = val
            continue
        
        # Parse accuracy
        if line.startswith('accuracy'):
            acc_match = re.search(r'[\d.]+', line)
            if acc_match:
                current_data['accuracy'] = float(acc_match.group())
            i += 1
            continue
        
        i += 1
    
    # Don't forget the last model
    if current_model and current_data:
        results.append(current_data)
    
    # Write to CSV
    if results:
        fieldnames = ['model', 'mAP', 'accuracy', 
                     'AP_class0', 'AP_class1', 'AP_class2',
                     'conf_0_0', 'conf_0_1', 'conf_0_2',
                     'conf_1_0', 'conf_1_1', 'conf_1_2',
                     'conf_2_0', 'conf_2_1', 'conf_2_2']
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        
        print(f"✓ Parsed {len(results)} models")
        print(f"✓ CSV written to: {output_file}")
    else:
        print("No results found in input file")

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        output_file = input_file.replace('.txt', '.csv')
    else:
        input_file = 'eval_overlap_0pct_100.txt'
        output_file = 'eval_overlap_0pct_100.csv'

    parse_results(input_file, output_file)
