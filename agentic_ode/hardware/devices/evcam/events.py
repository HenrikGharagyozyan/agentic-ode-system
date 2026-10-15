"""The event format: one structured numpy array per recording. No SDK needed, so the fake camera, the store
and the analysis can use it on any machine."""

import numpy as np

EVENT_DTYPE = np.dtype({"names": ["x", "y", "p", "t"], "formats": ["<u2", "<u2", "<i2", "<i8"]})  # = CD decoder output
