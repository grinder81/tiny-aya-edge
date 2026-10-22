"""CPU parity/benchmark of the exact new patch; never import/load model weights."""
import argparse
import json
from pathlib import Path
import textwrap
import time

import numpy as np


def patch_block():
    patch = Path(__file__).resolve().parents[1] / 'patches/python-echo-top1.patch'
    added = ''.join(line[1:] + '\n' for line in patch.read_text().splitlines()
                    if line.startswith('+') and not line.startswith('+++'))
    return compile(textwrap.dedent(added), str(patch), 'exec')


BLOCK = patch_block()


def optimized(values, count):
    scope = dict(np=np, logprobs_token=values, logprobs=count)
    exec(BLOCK, scope)
    return scope['top_ids']


def reference(values, count):
    return np.lexsort((np.arange(len(values)), values))[::-1][:count]


def echo_dictionary(values, ids, supplied, collide):
    # Simulate special/byte-token rendering collisions and supplied-token update
    # in precisely the wrapper's dictionary insertion order.
    def render(token):
        return str(token % 3) if collide else str(token)
    result = {render(int(i)): values[i] for i in ids}
    result.update({render(supplied): values[supplied]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    rng = np.random.default_rng(42)
    arrays = [np.array([], dtype=np.float32)]
    for n in (1, 32, 262144):
        arrays.extend([rng.normal(size=n).astype(np.float32),
                       rng.integers(-2, 3, size=n).astype(np.float32),
                       np.zeros(n, dtype=np.float32),
                       np.full(n, -np.inf, dtype=np.float32)])
    arrays.extend(np.array(a, dtype=np.float32) for a in (
        [0., -0., 0., -0.], [np.inf, 1., np.inf],
        [np.nan, 1., np.nan, -np.inf], [1., np.nan, 2.],
        [np.nan, np.nan], [-np.inf, np.inf, np.nan, np.inf]))
    comparisons = dictionaries = 0
    for values in arrays:
        for count in (0, 1, 2, 10):
            expected = reference(values, count)
            actual = optimized(values, count)
            assert list(actual) == list(expected), (values, count, actual, expected)
            comparisons += 1
            if not len(values):
                continue
            for supplied in sorted({0, len(values) // 2, len(values) - 1}):
                for collide in (False, True):
                    left = echo_dictionary(values, expected, supplied, collide)
                    right = echo_dictionary(values, actual, supplied, collide)
                    assert list(left) == list(right)
                    assert np.array_equal(list(left.values()), list(right.values()), equal_nan=True)
                    dictionaries += 1
    values = rng.normal(size=262144).astype(np.float32)
    # Warm both paths, then measure repeated complete selection calls.
    reference(values, 1)
    optimized(values, 1)
    def duration(function, repeats):
        start = time.perf_counter()
        for _ in range(repeats):
            function(values, 1)
        return (time.perf_counter() - start) / repeats
    old = duration(reference, 10)
    new = duration(optimized, 100)
    report = dict(scope='CPU ranking and echo dictionary parity only; no inference',
        numpy_version=np.__version__, seed=42, arrays=len(arrays),
        selection_comparisons=comparisons, dictionary_comparisons=dictionaries,
        benchmark_vocab=262144, lexsort_seconds=old, top1_seconds=new,
        speedup=old/new, counts_tested=[0, 1, 2, 10],
        special_values='ties, all-equal, signed zero, negative/positive infinity, NaN',
        note='Timing is this CPU microbenchmark, not end-to-end model throughput')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
