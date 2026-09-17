import numpy as np
import pandas as pd

from causal import assert_future_outage_invariant


def _builder(frame, is_train=False):
    # Toy analogue of the approved contract: only the observed window enters X.
    observed = frame[frame["hour_idx"] < 72].groupby("fipsCode")["outage"].mean()
    meta = frame[frame["hour_idx"] >= 72][["fipsCode", "hour_idx"]].reset_index(drop=True)
    X = pd.DataFrame({"observed_mean": meta["fipsCode"].map(observed).to_numpy()})
    y = None if not is_train else pd.DataFrame({"target": 0.0}, index=X.index)
    return X, y, meta


def test_future_outage_perturbation_is_invariant():
    raw = pd.DataFrame({
        "fipsCode": ["00001"] * 4,
        "hour_idx": [70, 71, 72, 73],
        "outage": [1.0, 2.0, 3.0, 4.0],
    })

    def perturb(frame):
        frame.loc[frame["hour_idx"] >= 72, "outage"] = np.nan
        return frame

    assert assert_future_outage_invariant(raw, _builder, perturb)
