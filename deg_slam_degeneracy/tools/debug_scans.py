import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "src")
from degslam.scan_sim import GarageWorld, make_truth_trajectory
from degslam.geometry import se2, transform_points

w = GarageWorld()
t = make_truth_trajectory()
s0 = w.to_scan(t[0])
s1 = w.to_scan(t[1])
print("npts", s0.shape, s1.shape)
print("local bbox0", np.round(s0.min(0), 2), np.round(s0.max(0), 2))
W0 = transform_points(se2(*t[0]), s0)
W1 = transform_points(se2(*t[1]), s1)
print("world bbox0", np.round(W0.min(0), 2), np.round(W0.max(0), 2))
d, _ = cKDTree(W0).query(W1)
print("NN pct", np.round(np.percentile(d, [50, 80, 90, 99]), 3),
      "frac<0.3:", round(float((d < 0.3).mean()), 3))
