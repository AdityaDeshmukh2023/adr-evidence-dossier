"""Actual optional SHAP/GNNExplainer with perturbation fidelity; never a mechanism proof."""
from __future__ import annotations

import random
import time
from pathlib import Path

from . import CLASSES
from .calibration import softmax
from .io import read_json, write_json, file_hash, object_hash
from .dataset import audit_splits, audit_predictive_features
from .inference import _load_bundle, bundle_manifest


def _probability(model, x, edges, query, roles, target, temperature, edge_weight=None):
    import torch
    with torch.no_grad():
        logits = model(x, edges, query, roles, edge_weight=edge_weight).tolist()
    return softmax(logits, temperature)[0][target]


def explain_pair(artifacts_dir, ingredient_ids, *, method='auto', background_pairs=None,
                 epochs=30, samples=64, seed=17):
    import numpy as np
    import torch
    from torch import nn
    status = bundle_manifest(artifacts_dir)
    if status['status'] != 'available':
        raise ValueError('A valid trained bundle is required.')
    bundle = _load_bundle(str(Path(artifacts_dir).resolve()), status['manifest_sha256'])
    manifest, features, model, index, x, roles, edges, _, calibration, scope = bundle
    if any(ident not in index for ident in ingredient_ids):
        raise ValueError('Missing molecule for explanation.')
    torch.manual_seed(seed)
    np.random.seed(seed)
    ids = sorted(ingredient_ids)
    query = torch.tensor([[index[ident] for ident in ids]], dtype=torch.long)
    temperature = calibration['temperature']
    with torch.no_grad():
        target = int(model(x, edges, query, roles).argmax(-1)[0])
    original = _probability(model, x, edges, query, roles, target, temperature)
    started = time.perf_counter()
    method = ('shap' if manifest['model_name'] == 'fingerprint' else 'gnn') if method == 'auto' else method
    top_features, top_edges, eligible_feature_count, eligible_edge_count = [], [], 0, 0
    eligible_random_edges = []
    if method == 'shap':
        if manifest['model_name'] != 'fingerprint':
            raise ValueError('SHAP baseline wrapper is defined for the fingerprint-only model.')
        import shap
        class MolecularPair(nn.Module):
            def __init__(self, classifier):
                super().__init__()
                self.classifier = classifier
            def forward(self, pair_x):
                first = self.classifier.molecular(pair_x[:, :2048])
                second = self.classifier.molecular(pair_x[:, 2048:])
                return self.classifier.decoder(torch.cat((first + second, torch.abs(first - second), first * second), 1))
        candidates = background_pairs or scope['message_pair_keys']
        mapped = [key.split('|') if isinstance(key, str) else key for key in candidates]
        mapped = [pair for pair in mapped if len(pair) == 2 and all(i in index for i in pair)]
        if not mapped:
            raise ValueError('SHAP requires training-only background pairs.')
        selected = random.Random(seed).sample(mapped, min(32, len(mapped)))
        background = torch.stack([torch.cat((x[index[a]], x[index[b]])) for a, b in selected])
        sample = torch.cat((x[index[ids[0]]], x[index[ids[1]]])).unsqueeze(0)
        explainer = shap.GradientExplainer(MolecularPair(model).eval(), background)
        values = np.asarray(explainer.shap_values(sample, nsamples=samples, rseed=seed))
        # Current SHAP: (batch,features,outputs); older SHAP: a list per output.
        attribution = values[0, :, target] if values.shape == (1, 4096, 3) else values[target, 0, :]
        active = [position for position in range(4096) if sample[0, position] > 0]
        eligible_feature_count = len(active)
        ranking = sorted(active, key=lambda position: abs(float(attribution[position])), reverse=True)[:20]
        top_features = [{'ingredient_id': ids[position // 2048], 'bit': position % 2048,
                         'node_index': index[ids[position // 2048]], 'attribution': float(attribution[position])}
                        for position in ranking]
        method_name, dependency_version = 'SHAP GradientExplainer', shap.__version__
    elif method == 'gnn':
        if manifest['model_name'] == 'fingerprint':
            raise ValueError('Fingerprint-only model has no message-passing graph to explain.')
        import torch_geometric
        from torch_geometric.nn import MessagePassing
        from torch_geometric.explain import Explainer, GNNExplainer
        class ExplainableMean(MessagePassing):
            def __init__(self, layer):
                super().__init__(aggr='mean')
                self.root, self.neighbor = layer.root, layer.neighbor
            def forward(self, inputs, edge_index):
                # Project before aggregation; linear no-bias neighbor transform commutes with mean.
                return self.root(inputs) + self.propagate(edge_index, x=self.neighbor(inputs), size=None)
            def message(self, x_j):
                return x_j
        class Adapter(nn.Module):
            def __init__(self, classifier):
                super().__init__()
                self.classifier = classifier
                self.first, self.second = ExplainableMean(classifier.sage1), ExplainableMean(classifier.sage2)
            def forward(self, inputs, edge_index, query, biological_roles):
                parts = []
                if self.classifier.model_name != 'graphsage':
                    molecular_x = torch.cat((inputs, biological_roles), 1) if self.classifier.model_name == 'role_fusion' else inputs
                    parts.append(self.classifier.molecular(molecular_x))
                h = self.classifier.dropout(torch.relu(self.first(inputs, edge_index)))
                parts.append(torch.relu(self.second(h, edge_index)))
                embeddings = torch.cat(parts, 1) if len(parts) > 1 else parts[0]
                return self.classifier.decode(embeddings, query)
        adapter = Adapter(model).eval()
        with torch.no_grad():
            if not torch.allclose(adapter(x, edges, query, roles), model(x, edges, query, roles), atol=1e-5, rtol=1e-4):
                raise ValueError('Explainability adapter differs from production classifier.')
        explainer = Explainer(model=adapter, algorithm=GNNExplainer(epochs=epochs, lr=.01),
                              explanation_type='model', node_mask_type='attributes', edge_mask_type='object',
                              model_config={'mode': 'multiclass_classification', 'task_level': 'edge', 'return_type': 'raw'})
        explanation = explainer(x, edges, query=query, biological_roles=roles, index=0)
        feature_scores = (explanation.node_mask.detach() * x).flatten()
        eligible_feature_count = int((feature_scores > 0).sum())
        values, positions = torch.topk(feature_scores, min(20, int((feature_scores > 0).sum())))
        reverse = {value: ident for ident, value in index.items()}
        top_features = [{'ingredient_id': reverse[int(position) // 2048], 'bit': int(position) % 2048,
                         'node_index': int(position) // 2048, 'attribution': float(value)}
                        for value, position in zip(values, positions)]
        edge_scores = explanation.edge_mask.detach().tolist()
        undirected = {}
        for position, (a, b) in enumerate(edges.t().tolist()):
            key = tuple(sorted((a, b)))
            record = undirected.setdefault(key, {'indices': [], 'scores': []})
            record['indices'].append(position)
            record['scores'].append(edge_scores[position])
        eligible_edges = {key: row for key, row in undirected.items() if any(score > 0 for score in row['scores'])}
        eligible_edge_count = len(eligible_edges)
        eligible_random_edges = [row['indices'] for row in eligible_edges.values()]
        ranking = sorted(eligible_edges.items(), key=lambda pair: sum(pair[1]['scores']) / len(pair[1]['scores']), reverse=True)[:10]
        top_edges = [{'ingredient_ids': [reverse[a], reverse[b]], 'edge_indices': record['indices'],
                      'attribution': sum(record['scores']) / len(record['scores'])} for (a, b), record in ranking]
        method_name, dependency_version = 'PyG GNNExplainer', torch_geometric.__version__
    else:
        raise ValueError('Choose shap, gnn, or auto.')
    altered = x.clone()
    for feature in top_features:
        altered[feature['node_index'], feature['bit']] = 0
    feature_removed = _probability(model, altered, edges, query, roles, target, temperature)
    rng = random.Random(seed)
    # Match feature deletion count within the same explained nodes, using present fingerprint bits.
    affected_nodes = sorted({feature['node_index'] for feature in top_features})
    present = [(node, bit) for node in affected_nodes for bit in (x[node] > 0).nonzero().flatten().tolist()]
    random_altered = x.clone()
    for node, bit in rng.sample(present, min(len(top_features), len(present))):
        random_altered[node, bit] = 0
    random_removed = _probability(model, random_altered, edges, query, roles, target, temperature)
    fidelity = {'feature_probability_drop': original - feature_removed,
                'random_feature_probability_drop': original - random_removed}
    if top_edges:
        gate = torch.ones(edges.shape[1])
        for edge in top_edges:
            gate[edge['edge_indices']] = 0
        edge_removed = _probability(model, x, edges, query, roles, target, temperature, gate)
        random_gate = torch.ones(edges.shape[1])
        # Use the same mask-reachable candidate edges, rather than unrelated global edges.
        for positions in rng.sample(eligible_random_edges, min(len(top_edges), len(eligible_random_edges))):
            random_gate[positions] = 0
        random_edge_removed = _probability(model, x, edges, query, roles, target, temperature, random_gate)
        fidelity.update(edge_probability_drop=original - edge_removed,
                        random_edge_probability_drop=original - random_edge_removed)
    return {'pair_key': '|'.join(ids), 'ingredient_ids': ids, 'model_version': manifest['model_version'],
            'status': 'executed', 'method': method_name, 'dependency_version': dependency_version,
            'predicted_source_severity': CLASSES[target], 'calibrated_probability': original,
            'top_features': [{k: v for k, v in value.items() if k != 'node_index'} for value in top_features],
            'top_edges': [{k: v for k, v in value.items() if k != 'edge_indices'} for value in top_edges],
            'sparsity': {'selected_present_feature_count': len(top_features),
                         'eligible_present_feature_count': eligible_feature_count,
                         'selected_present_feature_fraction': len(top_features) / max(1, eligible_feature_count),
                         'feature_scope': 'pair present bits' if method == 'shap' else 'present bits with nonzero GNN explanation mask',
                         'selected_undirected_edge_count': len(top_edges),
                         'eligible_undirected_edge_count': eligible_edge_count,
                         'selected_undirected_edge_fraction': len(top_edges) / max(1, eligible_edge_count)},
            'fidelity': fidelity, 'seconds': time.perf_counter() - started,
            'random_deletion_scope': {'features': 'same explained nodes, present bits, matched count',
                                      'edges': 'nonzero explanation-mask edges, matched undirected count'},
            'seed': seed, 'epochs': epochs if method == 'gnn' else None,
            'samples': samples if method == 'shap' else None,
            'biology_attributes_held_fixed': manifest['model_name'] == 'role_fusion',
            'explanation_scope': 'chemical fingerprint attributes and training graph edges; biological role inputs held fixed',
            'interpretation': 'Features/neighbors affecting model output, not proven biological causality.'}


def run_explanations(artifacts_dir, dataset_dir, *, count=30, method='auto', epochs=30, samples=64,
                     stability_cases=3, output=None):
    dataset_dir, artifacts_dir = Path(dataset_dir), Path(artifacts_dir)
    dataset, splits = read_json(dataset_dir / 'dataset.json'), read_json(dataset_dir / 'splits.json')
    audit_splits(dataset['pairs'], splits)
    audit_predictive_features(dataset)
    manifest = read_json(artifacts_dir / 'manifest.json')
    if manifest['dataset_sha256'] != dataset['content_sha256']:
        raise ValueError('Explanation dataset does not match trained model.')
    scope = read_json(artifacts_dir / 'training_scope.json')
    if scope['split_sha256'] != object_hash(splits):
        raise ValueError('Explanation split does not match the trained artifact.')
    protocol = manifest['protocol']
    test_indices = splits[protocol].get('test', splits[protocol].get('test_one_unseen', []))
    train_nodes = {ident for position in splits[protocol]['message']
                   for ident in dataset['pairs'][position]['ingredient_ids']}
    require_graph = method == 'gnn' or (method == 'auto' and manifest['model_name'] != 'fingerprint')
    eligible = [position for position in test_indices if not require_graph or
                all(ident in train_nodes for ident in dataset['pairs'][position]['ingredient_ids'])]
    excluded = {'no_training_graph_neighbor': len(test_indices) - len(eligible)}
    rng = random.Random(17)
    chosen = []
    # Balanced source-label sampling is declared, not silently used as deployment prevalence.
    for severity in CLASSES:
        candidates = [position for position in eligible if dataset['pairs'][position]['severity'] == severity]
        chosen.extend(rng.sample(candidates, min(max(1, count // 3), len(candidates))))
    chosen = chosen[:count]
    cases = []
    for position in chosen:
        pair = dataset['pairs'][position]
        try:
            case = explain_pair(artifacts_dir, pair['ingredient_ids'], method=method, epochs=epochs, samples=samples)
            case['source_severity'] = pair['severity']
            if len(cases) < stability_cases:
                repeated = explain_pair(artifacts_dir, pair['ingredient_ids'], method=method, epochs=epochs,
                                        samples=samples, seed=29)
                first = {(row['ingredient_id'], row['bit']) for row in case['top_features']}
                second = {(row['ingredient_id'], row['bit']) for row in repeated['top_features']}
                case['feature_jaccard_second_seed'] = len(first & second) / len(first | second) if first | second else None
                first_edges = {tuple(row['ingredient_ids']) for row in case['top_edges']}
                second_edges = {tuple(row['ingredient_ids']) for row in repeated['top_edges']}
                case['edge_jaccard_second_seed'] = len(first_edges & second_edges) / len(first_edges | second_edges) if first_edges | second_edges else None
            cases.append(case)
        except Exception as exc:
            cases.append({'pair_key': pair['source_record_id'], 'status': 'failed',
                          'reason': type(exc).__name__, 'detail': str(exc)[:300]})
        if output:
            write_json(output, {'status': 'in_progress', 'cases': cases})
    summary = {'status': 'executed' if all(case['status'] == 'executed' for case in cases) and cases else 'partial_or_failed',
               'model_version': manifest['model_version'], 'requested_cases': count, 'selected_cases': len(chosen),
               'executed_cases': sum(case['status'] == 'executed' for case in cases), 'cases': cases,
               'eligible_cases': len(eligible), 'exclusions': excluded, 'sampling_seed': 17,
               'sample_policy': 'seed17 balanced sampling per known source severity from held-out test',
               'scope': 'Model explanation fidelity and stability; no pharmacological mechanism validation.'}
    executed = [case for case in cases if case['status'] == 'executed']
    aggregate = {'executed_count': len(executed), 'stability_case_count': sum('feature_jaccard_second_seed' in case for case in executed)}
    for metric in ('feature_probability_drop', 'random_feature_probability_drop',
                   'edge_probability_drop', 'random_edge_probability_drop'):
        values = [case['fidelity'][metric] for case in executed if metric in case['fidelity']]
        aggregate[metric] = {'mean': sum(values) / len(values), 'count': len(values)} if values else None
    for metric in ('feature_jaccard_second_seed', 'edge_jaccard_second_seed'):
        values = [case[metric] for case in executed if case.get(metric) is not None]
        aggregate[metric] = {'mean': sum(values) / len(values), 'count': len(values)} if values else None
    runtimes = sorted(case['seconds'] for case in executed)
    aggregate['runtime_seconds'] = {'mean': sum(runtimes) / len(runtimes),
                                     'p95': runtimes[int(.95 * (len(runtimes) - 1))]} if runtimes else None
    summary['aggregate'] = aggregate
    destination = Path(output) if output else artifacts_dir / 'explanations.json'
    write_json(destination, summary)
    if destination.resolve() == (artifacts_dir / 'explanations.json').resolve():
        manifest['files']['explanations.json'] = file_hash(destination)
        write_json(artifacts_dir / 'manifest.json', manifest)
    return summary
