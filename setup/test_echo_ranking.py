"""Check identical descending score/token-ID order, including tied scores."""
import numpy as np

rng = np.random.default_rng(42)
for n in [1, 32, 262144]:
    for values in [rng.normal(size=n).astype(np.float32),
                   rng.integers(-2, 3, size=n).astype(np.float32)]:
        for k in [1, 10]:
            expected = sorted(zip(values, range(n)), reverse=True)[:k]
            ids = np.lexsort((np.arange(n), values))[::-1][:k]
            assert expected == [(values[i], int(i)) for i in ids]
print('12 exact ordering comparisons passed')
