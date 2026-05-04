#!/usr/bin/env python3
"""
Standalone script to plot Newton convergence from CSV files.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_newton_csv(csv_path, output_dir=None):
    """Plot Newton convergence from a CSV file."""
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    # Read CSV
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    
    if data.size == 0:
        print(f"Warning: No data in {csv_path}")
        return
    
    # Handle single row case
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    load_steps = data[:, 0].astype(int)
    iterations = data[:, 1].astype(int)
    res_l2 = data[:, 2]
    res_rms = data[:, 3]
    
    # Create cumulative iteration count
    cumulative_iters = np.arange(1, len(iterations) + 1)
    
    # Extract sample tag from filename
    basename = os.path.basename(csv_path)
    sample_tag = basename.replace('newton_residuals_', '').replace('.csv', '')
    
    # Create figure
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # Plot 1: RMS residual vs cumulative iteration
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(cumulative_iters, res_rms, 'o-', linewidth=2, markersize=4, label='RMS residual')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Cumulative Newton Iteration', fontsize=11)
    ax1.set_ylabel('RMS Residual', fontsize=11)
    ax1.set_title('Convergence History', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Plot 2: Residual per load step (grouped)
    ax2 = fig.add_subplot(gs[0, 1])
    unique_steps = np.unique(load_steps)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_steps)))
    
    for i, step in enumerate(unique_steps):
        mask = load_steps == step
        step_iters = iterations[mask]
        step_rms = res_rms[mask]
        ax2.semilogy(step_iters, step_rms, 'o-', color=colors[i], 
                    linewidth=2, markersize=5, label=f'Step {step}')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Inner Iteration', fontsize=11)
    ax2.set_ylabel('RMS Residual', fontsize=11)
    ax2.set_title('Per Load Step', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=8, ncol=2)
    
    # Plot 3: Iterations per load step
    ax3 = fig.add_subplot(gs[0, 2])
    
    iters_per_step = []
    final_res_per_step = []
    for step in unique_steps:
        mask = load_steps == step
        iters_per_step.append(np.sum(mask))
        final_res_per_step.append(res_rms[mask][-1])
    
    x_pos = np.arange(len(unique_steps))
    bars = ax3.bar(x_pos, iters_per_step, color=colors, alpha=0.7, edgecolor='black')
    
    for i, (bar, final_res) in enumerate(zip(bars, final_res_per_step)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{final_res:.1e}',
                ha='center', va='bottom', fontsize=8)
    
    ax3.set_xlabel('Load Step', fontsize=11)
    ax3.set_ylabel('Newton Iterations', fontsize=11)
    ax3.set_title('Iterations per Load Step', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{s}' for s in unique_steps])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    total_iters = len(iterations)
    final_rms = res_rms[-1]
    fig.suptitle(f'Newton Solver Convergence: {sample_tag} | '
                f'Total Iterations: {total_iters} | Final RMS: {final_rms:.2e}',
                fontsize=13, fontweight='bold', y=0.98)
    
    # Save
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    output_path = os.path.join(output_dir, f'newton_convergence_{sample_tag}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)

def main():
    import glob
    
    # Find all CSV files in current directory
    csv_pattern = './nn_output_mechanical_2D_neohooke_pi_fno/newton_residuals_*.csv'
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching: {csv_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s)")
    
    for csv_file in sorted(csv_files):
        print(f"Processing: {csv_file}")
        plot_newton_csv(csv_file)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
