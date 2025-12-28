#!/usr/bin/env python3
"""
NLSE Wireless Coupling - Magnet Effect Analysis
================================================
Place this script in the folder containing MASTER/ and RECEIVER/ subdirectories.

Outputs:
  - magnet_effect_analysis.png
  - magnet_effect_results.json
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from datetime import datetime
import sys

# =============================================================================
# CONFIGURATION - Edit these for your experimental timeline
# =============================================================================
PHASE_CONFIG = {
    'no_magnet': (0, 310),           # seconds: 0:00 - 5:10
    'magnet_stationary': (310, 610), # seconds: 5:10 - 10:10
    'magnet_oscillating': (610, 730),# seconds: 10:10 - 12:10
    'magnet_out': (730, None)        # seconds: 12:10 - end
}

# Folder names (adjust if yours differ)
MASTER_FOLDER = "MASTER"
RECEIVER_FOLDER = "RECEIVER"  # Note: your spelling

# =============================================================================
# DATA LOADING
# =============================================================================
def find_prosecutor_log(folder, prefix):
    """Find the prosecutor log file (handles timestamped names)"""
    pattern = folder / f"{prefix}prosecutor_log_*.jsonl"
    matches = list(glob(str(pattern)))
    if not matches:
        raise FileNotFoundError(f"No prosecutor log found matching {pattern}")
    if len(matches) > 1:
        matches.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        print(f"  Multiple logs found, using most recent: {Path(matches[0]).name}")
    return Path(matches[0])

def load_jsonl(path):
    """Load JSONL prosecutor log"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_experiment_data(base_dir):
    """Load MASTER and RECEIVER data from subdirectories"""
    base = Path(base_dir)
    master_dir = base / MASTER_FOLDER
    receiver_dir = base / RECEIVER_FOLDER
    
    if not master_dir.exists():
        raise FileNotFoundError(f"MASTER directory not found: {master_dir}")
    if not receiver_dir.exists():
        raise FileNotFoundError(f"RECEIVER directory not found: {receiver_dir}")
    
    print("Loading data...")
    
    master_log = find_prosecutor_log(master_dir, "MASTER")
    receiver_log = find_prosecutor_log(receiver_dir, "RECEIVER")
    
    print(f"  MASTER: {master_log.name}")
    print(f"  RECEIVER: {receiver_log.name}")
    
    master_data = load_jsonl(master_log)
    receiver_data = load_jsonl(receiver_log)
    
    m_feat = np.array([d['features'] for d in master_data])
    r_feat = np.array([d['features'] for d in receiver_data])
    m_times = np.array([(d['mono_ns'] - master_data[0]['mono_ns'])/1e9 for d in master_data])
    r_times = np.array([(d['mono_ns'] - receiver_data[0]['mono_ns'])/1e9 for d in receiver_data])
    
    return {
        'master_features': m_feat,
        'receiver_features': r_feat,
        'master_times': m_times,
        'receiver_times': r_times,
        'master_log_name': master_log.name,
        'receiver_log_name': receiver_log.name
    }

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================
def compute_all_correlations(m_feat, r_feat, start_step, end_step):
    end_step = min(end_step, len(m_feat), len(r_feat))
    if end_step - start_step < 20:
        return None
    
    m = m_feat[start_step:end_step]
    r = r_feat[start_step:end_step]
    
    results = {'n_samples': end_step - start_step}
    
    for mq in range(4):
        for rq in range(4):
            key = f'M{mq+1}-R{rq+1}'
            corr, pval = stats.pearsonr(m[:, mq], r[:, rq])
            results[key] = {'r': float(corr), 'p': float(pval)}
    
    direct_keys = ['M1-R1', 'M2-R2', 'M3-R3', 'M4-R4']
    diagonal_keys = ['M1-R3', 'M3-R1', 'M2-R4', 'M4-R2']
    
    results['direct_mean'] = float(np.mean([results[k]['r'] for k in direct_keys]))
    results['diagonal_mean'] = float(np.mean([results[k]['r'] for k in diagonal_keys]))
    
    return results

def analyze_phases(m_feat, r_feat, m_times, phases):
    rate = len(m_feat) / m_times[-1]
    min_len = min(len(m_feat), len(r_feat))
    
    results = {
        'step_rate': float(rate),
        'total_steps': min_len,
        'duration_seconds': float(m_times[-1]),
        'phases': {}
    }
    
    for phase_name, (t_start, t_end) in phases.items():
        start_step = int(t_start * rate)
        end_step = int(t_end * rate) if t_end else min_len
        
        phase_corrs = compute_all_correlations(m_feat, r_feat, start_step, end_step)
        if phase_corrs:
            phase_corrs['start_step'] = start_step
            phase_corrs['end_step'] = end_step
            phase_corrs['start_time'] = t_start
            phase_corrs['end_time'] = t_end if t_end else float(m_times[-1])
            results['phases'][phase_name] = phase_corrs
    
    return results

def compute_phase_deltas(results):
    phases = results['phases']
    deltas = {}
    
    phase_names = list(phases.keys())
    baseline = phase_names[0] if phase_names else None
    
    if not baseline:
        return deltas
    
    keys_to_compare = ['M1-R1', 'M2-R2', 'M3-R3', 'M4-R4', 
                       'M1-R3', 'M3-R1', 'M2-R4', 'M4-R2']
    
    for phase_name in phase_names[1:]:
        delta_key = f"{phase_name}_vs_{baseline}"
        deltas[delta_key] = {}
        for k in keys_to_compare:
            if k in phases[baseline] and k in phases[phase_name]:
                delta = phases[phase_name][k]['r'] - phases[baseline][k]['r']
                deltas[delta_key][k] = float(delta)
    
    for i in range(1, len(phase_names)):
        prev = phase_names[i-1]
        curr = phase_names[i]
        delta_key = f"{curr}_vs_{prev}"
        if delta_key not in deltas:
            deltas[delta_key] = {}
            for k in keys_to_compare:
                if k in phases[prev] and k in phases[curr]:
                    delta = phases[curr][k]['r'] - phases[prev][k]['r']
                    deltas[delta_key][k] = float(delta)
    
    return deltas

# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualization(results, output_path):
    phases = results['phases']
    phase_names = list(phases.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('NLSE Wireless Coupling - Magnet Effect Analysis', fontsize=14, fontweight='bold')
    
    colors = {
        'no_magnet': '#808080',
        'magnet_stationary': '#FF6B6B', 
        'magnet_oscillating': '#4ECDC4',
        'magnet_out': '#45B7D1'
    }
    
    # Plot 1: Direct correlations
    ax1 = axes[0, 0]
    direct_keys = ['M1-R1', 'M2-R2', 'M3-R3', 'M4-R4']
    x = np.arange(len(direct_keys))
    width = 0.2
    
    for i, phase in enumerate(phase_names):
        if phase in phases:
            vals = [phases[phase][k]['r'] for k in direct_keys]
            color = colors.get(phase, f'C{i}')
            ax1.bar(x + i*width, vals, width, label=phase.replace('_', ' '), 
                   color=color, alpha=0.8)
    
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xticks(x + width * (len(phase_names)-1) / 2)
    ax1.set_xticklabels(direct_keys)
    ax1.set_ylabel('Correlation (r)')
    ax1.set_title('Direct Quadrant Correlations')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(-0.5, 0.6)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Diagonal correlations
    ax2 = axes[0, 1]
    diag_keys = ['M1-R3', 'M3-R1', 'M2-R4', 'M4-R2']
    
    for i, phase in enumerate(phase_names):
        if phase in phases:
            vals = [phases[phase][k]['r'] for k in diag_keys]
            color = colors.get(phase, f'C{i}')
            ax2.bar(x + i*width, vals, width, label=phase.replace('_', ' '),
                   color=color, alpha=0.8)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x + width * (len(phase_names)-1) / 2)
    ax2.set_xticklabels(diag_keys)
    ax2.set_ylabel('Correlation (r)')
    ax2.set_title('Diagonal (90° Phase) Correlations')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(-0.5, 0.4)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Summary means
    ax3 = axes[1, 0]
    direct_means = [phases[p]['direct_mean'] for p in phase_names if p in phases]
    diag_means = [phases[p]['diagonal_mean'] for p in phase_names if p in phases]
    
    x3 = np.arange(len(phase_names))
    ax3.bar(x3 - 0.15, direct_means, 0.3, label='Direct mean', color='#2E86AB', alpha=0.8)
    ax3.bar(x3 + 0.15, diag_means, 0.3, label='Diagonal mean', color='#A23B72', alpha=0.8)
    
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([p.replace('_', '\n') for p in phase_names], fontsize=9)
    ax3.set_ylabel('Mean Correlation (r)')
    ax3.set_title('Direct vs Diagonal Mean Correlation by Phase')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Key channel evolution
    ax4 = axes[1, 1]
    m2r4_vals = [phases[p]['M2-R4']['r'] for p in phase_names if p in phases]
    m4r2_vals = [phases[p]['M4-R2']['r'] for p in phase_names if p in phases]
    m2r2_vals = [phases[p]['M2-R2']['r'] for p in phase_names if p in phases]
    
    ax4.plot(range(len(phase_names)), m2r4_vals, 'o-', markersize=10, 
             linewidth=2, label='M2-R4', color='#E63946')
    ax4.plot(range(len(phase_names)), m4r2_vals, 's-', markersize=10,
             linewidth=2, label='M4-R2', color='#457B9D')
    ax4.plot(range(len(phase_names)), m2r2_vals, '^-', markersize=10,
             linewidth=2, label='M2-R2 (direct)', color='#2A9D8F', alpha=0.7)
    
    ax4.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax4.set_xticks(range(len(phase_names)))
    ax4.set_xticklabels([p.replace('_', '\n') for p in phase_names], fontsize=9)
    ax4.set_ylabel('Correlation (r)')
    ax4.set_title('Key Channel Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.5, 0.5)
    
    for i, (v, phase) in enumerate(zip(m2r4_vals, phase_names)):
        ax4.annotate(f'{v:+.2f}', (i, v), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9, color='#E63946')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_path}")
    return fig

# =============================================================================
# MAIN
# =============================================================================
def main():
    script_dir = Path(__file__).parent.resolve()
    
    print("="*70)
    print("NLSE Wireless Coupling - Magnet Effect Analysis")
    print("="*70)
    print(f"Script directory: {script_dir}")
    
    try:
        data = load_experiment_data(script_dir)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print(f"\nLooking for: {MASTER_FOLDER}/ and {RECEIVER_FOLDER}/")
        sys.exit(1)
    
    m_feat = data['master_features']
    r_feat = data['receiver_features']
    m_times = data['master_times']
    
    print(f"\nData loaded:")
    print(f"  MASTER:   {len(m_feat)} steps, {m_times[-1]:.1f}s")
    print(f"  RECEIVER: {len(r_feat)} steps")
    print(f"  Step rate: {len(m_feat)/m_times[-1]:.2f} steps/sec")
    
    print(f"\nPhase configuration:")
    for name, (start, end) in PHASE_CONFIG.items():
        end_str = f"{end}s" if end else "end"
        print(f"  {name}: {start}s - {end_str}")
    
    results = analyze_phases(m_feat, r_feat, m_times, PHASE_CONFIG)
    results['deltas'] = compute_phase_deltas(results)
    results['metadata'] = {
        'analysis_time': datetime.now().isoformat(),
        'master_log': data['master_log_name'],
        'receiver_log': data['receiver_log_name'],
        'phase_config': {k: list(v) for k, v in PHASE_CONFIG.items()}
    }
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS BY PHASE")
    print("="*70)
    
    for phase_name, phase_data in results['phases'].items():
        print(f"\n{phase_name.upper().replace('_', ' ')}:")
        print(f"  Steps {phase_data['start_step']}-{phase_data['end_step']} "
              f"({phase_data['n_samples']} samples)")
        
        print("  Direct:")
        for k in ['M1-R1', 'M2-R2', 'M3-R3', 'M4-R4']:
            r = phase_data[k]['r']
            p = phase_data[k]['p']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {k}: r = {r:+.4f} {sig}")
        
        print("  Diagonal (90°):")
        for k in ['M1-R3', 'M3-R1', 'M2-R4', 'M4-R2']:
            r = phase_data[k]['r']
            p = phase_data[k]['p']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {k}: r = {r:+.4f} {sig}")
        
        print(f"  Means: direct={phase_data['direct_mean']:+.4f}, "
              f"diagonal={phase_data['diagonal_mean']:+.4f}")
    
    # Deltas
    print("\n" + "="*70)
    print("PHASE TRANSITIONS")
    print("="*70)
    
    for delta_name, delta_vals in results['deltas'].items():
        print(f"\n{delta_name.replace('_', ' ').upper()}:")
        for k, v in delta_vals.items():
            marker = " <<<" if abs(v) > 0.2 else ""
            print(f"  {k}: Δr = {v:+.4f}{marker}")
    
    # Save
    output_json = script_dir / "magnet_effect_results.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_json}")
    
    output_png = script_dir / "magnet_effect_analysis.png"
    create_visualization(results, output_png)
    
    print("\nDone.")

if __name__ == '__main__':
    main()