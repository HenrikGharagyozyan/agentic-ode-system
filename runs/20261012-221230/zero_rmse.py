import numpy as np
import scipy.io

# Load data
mat_path = '/home/henrik/Projects/agentic-ode-system/data/SNLS80mV.mat'
data = scipy.io.loadmat(mat_path)
V2 = data['V2'].flatten()  # ensure 1D

# Test excerpt
excerpt = V2[42650:46746]
excerpt_mean = np.mean(excerpt)
y_zero_mean = excerpt - excerpt_mean

# Last 1024 samples
test_segment = y_zero_mean[-1024:]

# Zero prediction
y_pred = np.zeros_like(test_segment)

# RMSE
rmse = np.sqrt(np.mean((test_segment - y_pred) ** 2))
print(rmse)