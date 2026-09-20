# Stella v1.12 strict protocol alignment

This implementation follows `versions/stella_v112_handoff/Experiment_Contract.md` and keeps the old strict baseline read-only.

| Contract item | Implementation | Evidence |
|---|---|---|
| Frozen seed42 county folds and 163 columns | `src.data_access.load_data` calls the frozen strict loader through an explicit file module | `preflight/data_validation.json` |
| Scope-internal probe, floor mean rounds, scope refit | `src.models.fit_task` / `selected_rows` | model receipts and `preflight/scope_sentinel_tests.json` |
| XGB/Cat outer models for direct plus P/N/D/R at 24h/48h | `src.planning.outer_tasks` | `dependency_plan.json` |
| Three-fold LightGBM inner anchor models, deduplicated by actual scope | `src.planning.inner_tasks` | `dependency_plan.json` threshold sources |
| Reuse 50 strict outer LightGBM models | `src.preflight.replay_reference_lgb` | `preflight/reference_lgb_replay.json` |
| Pinned F2 1h/6h predictions | `src.preflight.validate_f2_control` and assembly | `preflight/f2_control_validation.json` |
| Six-source processed mean and q95 strict gate | `src.arithmetic` and assembly | protocol examples, thresholds, candidate rows |
| County-trajectory paired bootstrap | evaluation stage | run metrics and bootstrap artifact |
| Independent fresh-process reload | `verify` command | run `verification.json` and `CV_COMPLETE` |

## Recorded portability mapping

The handoff directory was moved from the inventory's historical nested path to `versions/stella_v112_handoff`. The verifier maps only that exact prefix.

The checkout has `core.autocrlf=true`. Source verification therefore records both raw SHA-256 and LF-canonical SHA-256. A file is accepted only when either raw bytes match the inventory or changing CRLF to LF produces the exact frozen digest. The old files and identities are not edited. Referenced LightGBM text models are materialized as LF-canonical copies under the new experiment's `preflight/reference_models/`, verified against their original receipts, and replayed from those copies.

## Deliberate exclusions

- No test-set inference, final all-data model, or submission generation.
- No feature, weight, quantile, county, loss, fold, or early-stopping search.
- Confirmation splits 20260917 and 20260918 are not part of the seed42 run.
- The old Stella global OOF is not used to construct q95.

