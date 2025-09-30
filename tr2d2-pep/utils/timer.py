import time, torch
from collections import defaultdict
from contextlib import contextmanager

class StepTimer:
    def __init__(self, device=None):
        self.times = defaultdict(list)
        self.device = device
        self._use_cuda_sync = (
            isinstance(device, torch.device) and device.type == "cuda"
        ) or (isinstance(device, str) and "cuda" in device)

    @contextmanager
    def section(self, name):
        if self._use_cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._use_cuda_sync:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.times[name].append(dt)

    def summary(self, top_k=None):
        # returns (name, count, total, mean, p50, p95)
        import numpy as np
        rows = []
        for k, v in self.times.items():
            a = np.array(v, dtype=float)
            rows.append((k, len(a), a.sum(), a.mean(), np.median(a), np.percentile(a, 95)))
        rows.sort(key=lambda r: r[2], reverse=True)  # by total time
        return rows[:top_k] if top_k else rows
