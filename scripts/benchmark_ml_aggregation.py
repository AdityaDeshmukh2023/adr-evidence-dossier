"""Measure equivalent mean GraphSAGE aggregation orders on the frozen graph."""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adr_system.ml.io import read_json, write_json, object_hash, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/ml/dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ml/aggregation_benchmark.json')
    parser.add_argument('--repetitions', type=int, default=30)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError('Use at least three timing repetitions.')
    import torch
    from adr_system.ml.models import MeanSage, mean_neighbors, tensors
    torch.manual_seed(17)
    torch.set_num_threads(4)
    dataset, splits = read_json(args.dataset / 'dataset.json'), read_json(args.dataset / 'splits.json')
    graph_pairs = [dataset['pairs'][position]['ingredient_ids'] for position in splits['pair']['message']]
    _, x, _, _, edges = tensors(dataset, graph_pairs)
    layer = MeanSage(2048, 128).eval()
    original = lambda: layer.root(x) + layer.neighbor(mean_neighbors(x, edges))
    optimized = lambda: layer(x, edges)
    measurements = {'original': [], 'project_before_mean': []}
    with torch.no_grad():
        first, second = original(), optimized()
        if not torch.allclose(first, second, atol=1e-5, rtol=1e-4):
            raise ValueError('Aggregation order equivalence failed.')
        for _ in range(3):
            original()
            optimized()
        # Alternate order to reduce systematic thermal/order effects.
        for repeat in range(args.repetitions):
            order = [('original', original), ('project_before_mean', optimized)]
            if repeat % 2:
                order.reverse()
            for name, operation in order:
                started = time.perf_counter()
                operation()
                measurements[name].append((time.perf_counter() - started) * 1000)
    timings = {name: {'median_ms': statistics.median(values),
                      'p95_ms': sorted(values)[int(.95 * (len(values) - 1))]}
               for name, values in measurements.items()}
    result = {'status': 'executed', 'seed': 17, 'threads': 4, 'repetitions': args.repetitions,
              'dataset_sha256': dataset['content_sha256'], 'split_sha256': object_hash(splits),
              'script_sha256': file_hash(__file__), 'nodes': len(x), 'directed_edges': edges.shape[1],
              'original_aggregated_channels': 2048, 'optimized_aggregated_channels': 128,
              'max_absolute_output_difference': float((first - second).abs().max()),
              'timings': timings, 'median_speedup': timings['original']['median_ms'] / timings['project_before_mean']['median_ms'],
              'scope': 'Same bias-free neighbor linear transform and mean aggregation. '
                       'Synthetic seeded weights on real frozen graph; CPU microbenchmark, not whole-training speedup.'}
    write_json(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
