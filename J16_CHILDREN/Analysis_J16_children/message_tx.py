#!/usr/bin/env python3
"""
HERBIE MESSAGE TRANSMISSION
===========================
Analyzes prosecutor logs after a transmission attempt.

Usage:
  python message_tx.py master_prosecutor.jsonl receiver_prosecutor.jsonl
"""

import json
import numpy as np
import sys

def load_log(path):
    with open(path) as f:
        return [json.loads(l) for l in f.readlines()]

def pearson_r(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    if denom < 1e-10:
        return 0.0
    return (x * y).sum() / denom

def analyze_transmission(master_path, receiver_path):
    print("Loading logs...")
    master = load_log(master_path)
    receiver = load_log(receiver_path)
    
    m_feat = np.array([e['features'] for e in master])
    r_feat = np.array([e['features'] for e in receiver])
    
    min_len = min(len(m_feat), len(r_feat))
    m_feat = m_feat[:min_len]
    r_feat = r_feat[:min_len]
    
    if len(master) > 1:
        dt = (master[1]['mono_ns'] - master[0]['mono_ns']) / 1e9
        sample_rate = 1.0 / dt if dt > 0 else 10.0
    else:
        sample_rate = 10.0
    
    print(f"Aligned samples: {min_len}")
    print(f"Estimated sample rate: {sample_rate:.1f} Hz")
    print(f"Duration: {min_len / sample_rate:.1f} seconds")
    print()
    
    window_sec = 10
    step_sec = 5
    window = int(window_sec * sample_rate)
    step = int(step_sec * sample_rate)
    
    times = []
    q3_corr = []
    
    for i in range(0, min_len - window, step):
        t = (i + window // 2) / sample_rate
        r = pearson_r(m_feat[i:i+window, 3], r_feat[i:i+window, 3])
        times.append(t)
        q3_corr.append(r)
    
    print("=" * 60)
    print("Q3 CORRELATION TRACE")
    print("=" * 60)
    print()
    print("Time(s)  Corr   Visual")
    print("-" * 60)
    
    for t, r in zip(times, q3_corr):
        bar_pos = int((r + 1) * 25)
        bar = ['-'] * 51
        bar[25] = '|'
        bar[bar_pos] = '#'
        bar_str = ''.join(bar)
        sign = '+' if r > 0.2 else '-' if r < -0.2 else '0'
        print(f"{t:6.1f}   {r:+.3f}  [{bar_str}] {sign}")
    
    print()
    print("=" * 60)
    print("BIT DECODE (threshold = +/-0.2)")
    print("=" * 60)
    print()
    
    bits = []
    for t, r in zip(times, q3_corr):
        if r > 0.2:
            bits.append(('0', t))
        elif r < -0.2:
            bits.append(('1', t))
        else:
            bits.append(('?', t))
    
    collapsed = []
    if bits:
        current_bit = bits[0][0]
        current_start = bits[0][1]
        for bit, t in bits[1:]:
            if bit != current_bit:
                collapsed.append((current_bit, current_start, t))
                current_bit = bit
                current_start = t
        collapsed.append((current_bit, current_start, bits[-1][1]))
    
    print("Raw bit stream:")
    print(''.join(b[0] for b in bits))
    print()
    
    print("Collapsed (run-length decoded):")
    for bit, start, end in collapsed:
        duration = end - start
        print(f"  {bit} : {start:.1f}s - {end:.1f}s ({duration:.1f}s)")
    
    bit_string = ''.join(b[0] for b in collapsed if b[0] in '01')
    print()
    print(f"Bit string: {bit_string}")
    
    if len(bit_string) >= 8:
        print()
        print("ASCII decode attempt:")
        for i in range(0, len(bit_string) - 7, 8):
            byte = bit_string[i:i+8]
            try:
                char = chr(int(byte, 2))
                if 32 <= ord(char) <= 126:
                    print(f"  {byte} = '{char}'")
                else:
                    print(f"  {byte} = 0x{ord(char):02x} (non-printable)")
            except:
                print(f"  {byte} = ???")
    
    print()
    print("=" * 60)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python message_tx.py <master.jsonl> <receiver.jsonl>")
        sys.exit(1)
    analyze_transmission(sys.argv[1], sys.argv[2])
