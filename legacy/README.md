# Archived components

`context.py`, `context_FoodDataCentral.py`, and `preprocess.py` are historical experiments and are not imported by the application. Their optional dependencies are not part of the supported installation.

`frozen_v1/adr_system/` preserves the original committed screening engine for research comparisons. `origin.json` records its commit. The experiment runner redirects its data directory to the same frozen local source snapshots used by the current engine, then scores the findings it actually produces. Its known defects are intentionally preserved. Never use this engine for application findings.
