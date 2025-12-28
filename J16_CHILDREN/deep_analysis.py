#!/usr/bin/env python3
"""
Deeper Statistical Analysis - Beyond Pearson Correlation
=========================================================
- Variance structure
- Mutual information
- Entropy
- Cross-correlation lags
- Joint distribution analysis
"""

import json
import numpy as np
from scipy import stats
from scipy.signal import correlate
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
PHASE_CONFIG = {
    'no_magnet': (0, 310),
    'magnet_stationary': (310, 610),
    'magnet_oscillating': (610, 730),
    'magnet_out': (730, None)
}

MASTER_FOLDER = "MASTER"
RECEIVER_FOLDER = "RECEIVER"

# =============================================================================
# DATA LOADING
# =============================================================================
def find_prosecutor_log(folder, prefix):
    pattern = folder / f"{prefix}prosecutor_log_*.jsonl"
    matches = list(glob(str(pattern)))
    if not matches:
        raise FileNotFoundError(f"No log found: {pattern}")
    matches.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return Path(matches[0])

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_data(base_dir):
    base = Path(base_dir)
    master_log = find_prosecutor_log(base / MASTER_FOLDER, "MASTER")
    receiver_log = find_prosecutor_log(base / RECEIVER_FOLDER, "RECEIVER")
    
    m_data = load_jsonl(master_log)
    r_data = load_jsonl(receiver_log)
    
    m_feat = np.array([d['features'] for d in m_data])
    r_feat = np.array([d['features'] for d in r_data])
    m_times = np.array([(d['mono_ns'] - m_data[0]['mono_ns'])/1e9 for d in m_data])
    
    return m_feat, r_feat, m_times

# =============================================================================
# MUTUAL INFORMATION (binned estimator)
# =============================================================================
def mutual_information(x, y, bins=20):
    """Estimate MI using binned histogram method"""
    c_xy = np.histogram2d(x, y, bins)[0]
    c_x = np.histogram(x, bins)[0]
    c_y = np.histogram(y, bins)[0]
    
    # Normalize to probabilities
    p_xy = c_xy / c_xy.sum()
    p_x = c_x / c_x.sum()
    p_y = c_y / c_y.sum()
    
    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i,j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i,j] * np.log2(p_xy[i,j] / (p_x[i] * p_y[j]))
    return mi

def entropy(x, bins=20):
    """Shannon entropy of distribution"""
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist)) * (x.max() - x.min()) / bins

# =============================================================================
# CROSS-CORRELATION WITH LAGS
# =============================================================================
def xcorr_peak(x, y, max_lag=50):
    """Find peak cross-correlation and its lag"""
    x_norm = (x - x.mean()) / (x.std() + 1e-10)
    y_norm = (y - y.mean()) / (y.std() + 1e-10)
    
    xcorr = correlate(x_norm, y_norm, mode='full')
    xcorr = xcorr / len(x)  # Normalize
    
    mid = len(xcorr) // 2
    lags = np.arange(-mid, mid + 1)
    
    # Look within max_lag
    valid = (np.abs(lags) <= max_lag)
    xcorr_valid = xcorr[valid]
    lags_valid = lags[valid]
    
    peak_idx = np.argmax(np.abs(xcorr_valid))
    return xcorr_valid[peak_idx], lags_valid[peak_idx]

# =============================================================================
# ANALYSIS
# =============================================================================
def analyze_phase(m, r, label):
    """Comprehensive analysis of one phase"""
    results = {'label': label, 'n': len(m)}
    
    # Variance per quadrant
    results['var_M'] = [float(m[:, q].var()) for q in range(4)]
    results['var_R'] = [float(r[:, q].var()) for q in range(4)]
    results['var_M_total'] = float(np.sum(results['var_M']))
    results['var_R_total'] = float(np.sum(results['var_R']))
    
    # Entropy per quadrant
    results['entropy_M'] = [float(entropy(m[:, q])) for q in range(4)]
    results['entropy_R'] = [float(entropy(r[:, q])) for q in range(4)]
    
    # Mutual information - direct and cross
    results['MI_direct'] = {}
    results['MI_cross'] = {}
    for mq in range(4):
        for rq in range(4):
            mi = mutual_information(m[:, mq], r[:, rq])
            key = f'M{mq+1}-R{rq+1}'
            if mq == rq:
                results['MI_direct'][key] = float(mi)
            else:
                results['MI_cross'][key] = float(mi)
    
    results['MI_direct_mean'] = float(np.mean(list(results['MI_direct'].values())))
    results['MI_cross_mean'] = float(np.mean(list(results['MI_cross'].values())))
    
    # Cross-correlation peaks and lags
    results['xcorr_peaks'] = {}
    results['xcorr_lags'] = {}
    for mq in range(4):
        for rq in range(4):
            peak, lag = xcorr_peak(m[:, mq], r[:, rq])
            key = f'M{mq+1}-R{rq+1}'
            results['xcorr_peaks'][key] = float(peak)
            results['xcorr_lags'][key] = int(lag)
    
    # Covariance matrix structure
    combined = np.hstack([m, r])
    cov = np.cov(combined.T)
    results['cov_trace'] = float(np.trace(cov))
    results['cov_det'] = float(np.linalg.det(cov))
    
    # Cross-covariance block (M with R)
    cross_cov = cov[:4, 4:]
    results['cross_cov_norm'] = float(np.linalg.norm(cross_cov))
    
    return results

def print_comparison(ctrl, exp, metric_name, keys=None):
    """Print side-by-side comparison"""
    print(f"\n{metric_name}:")
    print(f"  {'':20} | {'CONTROL':>10} | {'EXPERIMENTAL':>12} | {'Δ':>8}")
    print(f"  {'-'*55}")
    
    if keys is None:
        keys = ctrl.keys() if isinstance(ctrl, dict) else range(len(ctrl))
    
    for k in keys:
        c = ctrl[k] if isinstance(ctrl, dict) else ctrl[k]
        e = exp[k] if isinstance(exp, dict) else exp[k]
        d = e - c
        marker = " <--" if abs(d) > 0.05 else ""
        if isinstance(k, int):
            label = f"Q{k+1}"
        else:
            label = k
        print(f"  {label:20} | {c:10.4f} | {e:12.4f} | {d:+8.4f}{marker}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    script_dir = Path(__file__).parent.resolve()
    
    # Detect if we're in control or experimental based on folder name
    if 'control' in str(script_dir).lower():
        run_type = "CONTROL"
    else:
        run_type = "EXPERIMENTAL"
    
    print("="*70)
    print(f"DEEP STATISTICAL ANALYSIS - {run_type}")
    print("="*70)
    
    m_feat, r_feat, m_times = load_data(script_dir)
    rate = len(m_feat) / m_times[-1]
    min_len = min(len(m_feat), len(r_feat))
    
    print(f"Loaded: {min_len} usable steps, {rate:.2f} steps/sec")
    
    all_results = {}
    
    for phase_name, (t_start, t_end) in PHASE_CONFIG.items():
        s = int(t_start * rate)
        e = int(t_end * rate) if t_end else min_len
        e = min(e, min_len)
        
        if e - s < 50:
            continue
            
        results = analyze_phase(m_feat[s:e], r_feat[s:e], phase_name)
        all_results[phase_name] = results
        
        print(f"\n{'='*70}")
        print(f"{phase_name.upper().replace('_', ' ')}")
        print(f"{'='*70}")
        
        print(f"\nVARIANCE:")
        print(f"  MASTER total:   {results['var_M_total']:.6f}")
        print(f"  RECEIVER total: {results['var_R_total']:.6f}")
        print(f"  Per quadrant M: {[f'{v:.4f}' for v in results['var_M']]}")
        print(f"  Per quadrant R: {[f'{v:.4f}' for v in results['var_R']]}")
        
        print(f"\nENTROPY:")
        print(f"  MASTER:   {[f'{v:.2f}' for v in results['entropy_M']]}")
        print(f"  RECEIVER: {[f'{v:.2f}' for v in results['entropy_R']]}")
        
        print(f"\nMUTUAL INFORMATION:")
        print(f"  Direct (Mx-Rx) mean:  {results['MI_direct_mean']:.4f}")
        print(f"  Cross mean:           {results['MI_cross_mean']:.4f}")
        print(f"  Direct channels: {results['MI_direct']}")
        
        print(f"\nCROSS-CORRELATION PEAKS & LAGS:")
        print(f"  {'Channel':<10} | {'Peak':>8} | {'Lag':>5}")
        print(f"  {'-'*30}")
        for k in ['M1-R1', 'M2-R2', 'M3-R3', 'M4-R4', 'M2-R4', 'M4-R2']:
            peak = results['xcorr_peaks'][k]
            lag = results['xcorr_lags'][k]
            print(f"  {k:<10} | {peak:+8.4f} | {lag:>5}")
        
        print(f"\nCOVARIANCE STRUCTURE:")
        print(f"  Cross-cov norm: {results['cross_cov_norm']:.6f}")
    
    # Save results
    output_file = script_dir / f"deep_analysis_{run_type.lower()}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_file}")

if __name__ == '__main__':
    main()
