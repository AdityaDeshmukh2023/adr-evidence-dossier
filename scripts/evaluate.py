"""Run the frozen synthetic evaluation snapshot without external APIs."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adr_system.engine import analyze_medications


def main() -> None:
    fixture = json.loads((ROOT / "data" / "evaluation_cases.json").read_text())
    rows = []
    for case in fixture["cases"]:
        result = analyze_medications(case["medications"], case["foods"])
        actual = max((a.severity for a in result.alerts), key=lambda x: {"high":3,"moderate":2,"low":1,"unknown":0}[x], default=None)
        rows.append({"id":case["id"],"passed":len(result.alerts)==case["expected_alerts"] and actual==case["expected_severity"],"alerts":len(result.alerts),"severity":actual})
    print(json.dumps({"snapshot":fixture["version"],"passed":sum(r["passed"] for r in rows),"total":len(rows),"cases":rows}, indent=2))


if __name__ == "__main__": main()
