"""Create an unlabelled private annotation workbook. Never invent expert labels."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=150)
    parser.add_argument('--output', type=Path, default=Path('data/private/annotations.json'))
    args = parser.parse_args()
    if not 1 <= args.count <= 1000:
        raise ValueError('Use 1–1000 cases.')
    if args.output.exists():
        raise FileExistsError('Annotation file already exists; it will not be overwritten.')
    data = {'version': 'expert-pilot-UNFROZEN', 'kind': 'expert_reviewed', 'permission_confirmed': False,
            'label_provenance': 'PENDING expert annotation', 'scope': 'English medication entries from permitted prescriptions',
            'cases': [{'id': f'CASE-{i + 1:03d}', 'group_id': f'ORIGIN-{i + 1:03d}',
                       'split': 'development' if i % 5 == 0 else 'test', 'input_kind': 'real_printed_text',
                       'origin': 'permitted_example', 'annotation_status': 'pending', 'reviewer_ids': [],
                       'entries': [{'text': '', 'gold_names': [], 'is_medication': True}], 'gold_findings': []}
                      for i in range(args.count)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f'Created {args.count} unlabelled cases at {args.output}. Evaluation will reject them until completed and reviewed.')


if __name__ == '__main__':
    main()
