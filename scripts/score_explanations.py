"""Aggregate completed human ratings; blank sheets are rejected."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ratings', type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.ratings)
    fields = ['supported_claims', 'total_claims', 'correct_citations', 'total_citations', 'omissions', 'reviewer_id']
    if frame.empty or frame[fields].isna().any().any():
        raise ValueError('Complete every required rating and reviewer identifier before scoring.')
    for field in fields[:-1]:
        frame[field] = pd.to_numeric(frame[field], errors='raise')
        if (frame[field] < 0).any() or (frame[field] % 1 != 0).any():
            raise ValueError('Rating counts must be nonnegative integers.')
    if (frame.supported_claims > frame.total_claims).any() or (frame.correct_citations > frame.total_citations).any():
        raise ValueError('Supported/correct counts cannot exceed totals.')
    outputs = pd.DataFrame(json.loads((args.ratings.parent / 'outputs.json').read_text(encoding='utf-8')))
    merged = frame.merge(outputs[['output_id', 'method']], on='output_id', validate='many_to_one')
    if len(merged) != len(frame):
        raise ValueError('Unknown output ID in ratings.')
    results = []
    for method, group in merged.groupby('method'):
        total, citations = group.total_claims.sum(), group.total_citations.sum()
        results.append({'method': method, 'unsupported_claim_rate': float(1 - group.supported_claims.sum() / total) if total else None,
                        'citation_correctness': float(group.correct_citations.sum() / citations) if citations else None,
                        'omissions': int(group.omissions.sum()), 'ratings': len(group)})
    (args.ratings.parent / 'rated_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
