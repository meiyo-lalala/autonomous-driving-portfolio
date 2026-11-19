import csv
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results_quick/garage_r20/frames.csv"
rows = list(csv.DictReader(open(path)))
naive = [r for r in rows if r["mode"] == "naive"]
print(f"{'frame':>6}{'cond':>9}{'lammin':>10}{'inlier':>8}{'rmse':>8}{'dir':>7}")
for i in range(0, len(naive), max(1, len(naive) // 24)):
    r = naive[i]
    print(f"{r['frame']:>6}{float(r['cond']):>9.1f}{float(r['lam_min']):>10.2f}"
          f"{float(r['inlier_ratio']):>8.2f}{float(r['rmse']):>8.3f}"
          f"{float(r['direction_deg']):>7.1f}")
print("\ncond pct", np.round(np.percentile([float(r['cond']) for r in naive], [50, 90, 99]), 1))
print("rmse pct", np.round(np.percentile([float(r['rmse']) for r in naive], [50, 90, 99]), 3))
print("inlier pct", np.round(np.percentile([float(r['inlier_ratio']) for r in naive], [10, 50, 90]), 3))
