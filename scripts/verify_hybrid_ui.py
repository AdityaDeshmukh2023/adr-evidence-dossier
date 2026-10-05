"""Execute the model-enabled Streamlit flow on synthetic local demo medicines."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify() -> dict:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(ROOT / 'llm.py', default_timeout=40).run()
    checks = {'initial_page_runs': not bool(app.exception)}
    app.text_area(key='med_input').set_value('warfarin\naspirin\nacetaminophen')
    app.checkbox(key='enable_local_model').set_value(True)
    next(button for button in app.button if button.label == 'Build review dossier').click().run()
    result = app.session_state['result']
    checks['model_enabled_page_runs'] = not bool(app.exception)
    checks['documented_high_and_unknown_preserved'] = {'high', 'unknown'} <= {a.severity for a in result.alerts}
    checks['unknown_source_model_outputs_present'] = bool(result.model_predictions)
    checks['local_model_status_available'] = result.hybrid_status['model']['status'] == 'available'
    model_status = result.hybrid_status['model']
    predictions = result.model_predictions
    app.radio[0].set_value('Methods').run()
    app.radio[0].set_value('Review desk').run()
    checks['model_setting_survives_navigation'] = app.checkbox(key='enable_local_model').value is True
    checks['navigation_has_no_stale_warning'] = not any('setting has changed' in item.value for item in app.warning)
    app.checkbox(key='enable_local_model').set_value(False).run()
    checks['toggle_marks_previous_result_stale'] = any('setting has changed' in item.value for item in app.warning)
    checks['stale_result_has_no_exports'] = not app.get('download_button')
    next(button for button in app.button if button.label == 'Build review dossier').click().run()
    checks['rebuild_disables_model'] = (app.session_state['result'].hybrid_status['model']['status'] == 'disabled'
                                      and not app.session_state['result'].model_predictions)
    app.radio[0].set_value('Interaction map').run()
    checks['typed_graph_page_runs'] = not bool(app.exception)
    app.radio[0].set_value('Research workspace').run()
    checks['research_page_runs'] = not bool(app.exception)
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
            'checks_passed': sum(bool(v) for v in checks.values()), 'checks_total': len(checks),
            'executed_at_utc': datetime.now(timezone.utc).isoformat(),
            'model': model_status, 'predictions': predictions,
            'manifest': result.manifest,
            'scope': 'Actual Streamlit AppTest navigation and state/export behavior using synthetic medicines; no live LLM requests.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/hybrid-release/model_ui.json')
    args = parser.parse_args()
    report = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'checks_passed', 'checks_total', 'checks')}, indent=2))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
