#!/usr/bin/env python3
"""Summarise a results/<label>/gpu.csv (nvidia-smi 1 Hz log written by tests/run_suite.sh): per GPU max/mean power
while busy (>50% utilisation), max temperature, peak VRAM."""
import csv, json, sys
rows = list(csv.reader(open(sys.argv[1])))[1:]
by = {}
for r in rows:
    try:
        by.setdefault(int(r[1]), []).append((float(r[2].split()[0]), float(r[4]), float(r[5].split()[0]), float(r[7].split()[0])))
    except (ValueError, IndexError):
        continue
out = {}
for i, v in sorted(by.items()):
    busy = [x for x in v if x[3] > 50] or v
    out[f"gpu{i}"] = {"seconds": len(v), "max_power_w": max(x[0] for x in v),
                      "mean_power_busy_w": round(sum(x[0] for x in busy) / len(busy), 1),
                      "max_temp_c": max(x[1] for x in v), "peak_vram_mib": max(x[2] for x in v)}
print(json.dumps(out, indent=1))
