#!/usr/bin/env python3
"""
Quick script to generate Newton convergence plots from existing CSV files.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..',)))
from fol.tools.newton_residual_tracker import NewtonResidualTracker

def main():
    case_dir = './nn_output_mechanical_2D_neohooke_pi_fno'
    
    # Find all CSV files matching the pattern
    csv_files = []
    if os.path.exists(case_dir):
        for fname in os.listdir(case_dir):
            if fname.startswith('newton_residuals_') and fname.endswith('.csv'):
                # Extract sample_tag from filename: newton_residuals_{sample_tag}.csv
                sample_tag = fname.replace('newton_residuals_', '').replace('.csv', '')
                csv_files.append(sample_tag)
    
    if not csv_files:
        print(f"No Newton residual CSV files found in {case_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s). Generating plots...")
    
    for sample_tag in csv_files:
        print(f"  Generating plot for: {sample_tag}")
        try:
            plotter = NewtonResidualTracker(case_dir, sample_tag)
            plotter.plot_convergence()
        except Exception as e:
            print(f"    Error: {e}")
    
    print("\nDone! Check the output directory for PNG files.")

if __name__ == "__main__":
    main()
