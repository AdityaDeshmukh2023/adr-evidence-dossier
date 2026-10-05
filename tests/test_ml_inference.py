from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import uuid
import sys
from contextlib import nullcontext
import pytest
from adr_system.ml.inference import bundle_manifest, predict_unknown


@pytest.fixture
def model_tempdir():
    # Ordinary workspace directory avoids Windows restricted-user mode-0700 temp ACLs.
    path = Path(__file__).resolve().parents[1] / 'artifacts/ml/tests' / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def test_missing_artifacts_are_unavailable_without_importing_torch(model_tempdir):
    assert bundle_manifest(model_tempdir)['reason'] == 'model_artifacts_absent'


def test_absent_pair_never_receives_prediction(model_tempdir):
    result = SimpleNamespace(pair_assessments=[{'ingredient_ids': ['DDInter1', 'DDInter10'], 'state': 'no_record'}])
    assert predict_unknown(result, model_tempdir) == []


def test_declared_unknown_pair_is_rechecked_against_authoritative_source(model_tempdir):
    result = SimpleNamespace(pair_assessments=[{'ingredient_ids': ['nonexistent', 'another'], 'state': 'unknown'}])
    assert predict_unknown(result, model_tempdir) == []


def test_real_source_unknown_pair_abstains_when_model_unavailable(model_tempdir):
    from adr_system.data import load_ddinter_lookup
    lookup, _ = load_ddinter_lookup()
    pair = next(key.split('|') for key, record in lookup['interactions'].items() if record['severity'] == 'unknown')
    result = SimpleNamespace(pair_assessments=[{'ingredient_ids': pair, 'state': 'unknown'}])
    rows = predict_unknown(result, model_tempdir)
    assert len(rows) == 1
    assert rows[0]['accepted'] is False
    assert rows[0]['abstention_reason'] == 'model_artifacts_absent'
    assert rows[0]['probabilities'] == {}


@pytest.mark.parametrize('failure_type', [RuntimeError, ImportError])
def test_late_optional_model_failure_resets_already_scored_rows(monkeypatch, failure_type):
    from adr_system.ml import inference
    fake_torch = SimpleNamespace(tensor=lambda values, dtype=None: values,
                                  long='long', no_grad=nullcontext)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    status = {'status': 'available', 'runtime_available': True, 'source_snapshot_compatible': True,
              'biology_snapshot_compatible': True, 'manifest_sha256': 'test', 'hashes': {}, 'protocol': 'pair'}
    monkeypatch.setattr(inference, 'bundle_manifest', lambda directory: status)
    class Model:
        calls = 0
        def decode(self, embeddings, query):
            self.calls += 1
            if self.calls == 2:
                raise failure_type('optional checkpoint failure')
            return SimpleNamespace(tolist=lambda: [[10, 0, 0]])
    bundle = ({'protocol': 'pair'}, {'nodes': [{'roles': [1]}, {'roles': [0]}]}, Model(),
              {'a': 0, 'b': 1}, None, None, None, None,
              {'temperature': 1, 'policy': {'threshold': .5}},
              {'purpose': 'source_label_benchmark', 'train_drug_ids': ['a', 'b']})
    monkeypatch.setattr(inference, '_load_bundle', lambda directory, digest: bundle)
    rows = inference.predict_pairs([['a', 'b'], ['a', 'b']], '.')
    assert all(row['status'] == 'abstained' and not row['accepted'] for row in rows)
    assert all(row['probabilities'] == {} and row['explanation_status'] == 'failed' for row in rows)
