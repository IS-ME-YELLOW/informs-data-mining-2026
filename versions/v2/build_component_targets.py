"""Build and validate the shared v2 component targets from DM_Train.csv."""

from component_experiment import TARGETS_FILE, prepare_component_targets


if __name__ == "__main__":
    targets, validation = prepare_component_targets(force=True)
    print(f"Saved {targets.shape} component-target table to {TARGETS_FILE}")
    for horizon, result in validation["horizons"].items():
        print(
            f"{horizon}: rows={result['valid_rows']}, "
            f"max formula error={result['formula_max_abs_error_vs_official']:.8f}"
        )
