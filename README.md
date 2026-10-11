# Curvature-Audited Active Telemetry

This project evaluates cost-aware telemetry acquisition for incident-origin
localization on the LANL Comprehensive Multi-Source Cyber-Security Events
dataset. It is intentionally separate from `ba2`.

The red-team source field is severely imbalanced, so the primary estimand is
the first compromised destination host in each independently sessionized
red-team burst. The first labelled event is hidden. Candidate construction,
feature extraction, and acquisition use only ordinary telemetry available
before or after that hidden event according to the declared observation
budget. Episode-gap sensitivity is reported at 15, 30, 60, and 120 minutes.

## Reproducible entry points

```bash
conda run -n cdg python -m ba3.data.episodes --redteam data/raw/lanl/redteam.txt.gz
conda run -n cdg python -m ba3.experiments.run_lanl --config configs/lanl_active_telemetry.json
conda run -n cdg pytest -q
```

No manuscript claim is accepted unless it is supported by held-out incidents,
paired uncertainty intervals, and a machine-readable result table.
