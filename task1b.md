Write and run a script that computes the zero-prediction baseline RMSE on the Silverbox test segment and report it.

The test segment: take samples [42650:46746] of SNLS80mV.mat, subtract the mean of V2 (y) on this
excerpt, and use the last 1024 samples. The zero prediction is y_hat = 0 for every sample.
