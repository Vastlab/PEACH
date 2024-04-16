from PEACH import PEACH
import numpy as np
import time

t0 = time.time()
features = np.random.rand(5000,3072)

gpu = 0
result = PEACH(features, gpu, no_singleton = False) # 0 means GPU0


print(result)

t1 = time.time()
print("runining time = ", t1-t0)
