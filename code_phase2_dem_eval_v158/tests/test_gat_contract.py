import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gat_model import fit_gat, predict_gat


def test_fit_and_predict_use_edge_attributes_identically():
    features = np.zeros((2, 3, 205), dtype=np.float32)
    features[:, :, 0] = np.asarray([[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]])
    residual = np.zeros((2, 3), dtype=np.float32)
    residual[:, 0] = 0.1
    mask = np.zeros((2, 3), dtype=bool)
    mask[:, 0] = True
    edge_index = np.asarray([[0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 2, 0]], dtype=np.int64)
    edge_attr = np.ones((6, 4), dtype=np.float32)
    model, returned, _ = fit_gat(
        features, residual, mask, edge_index, epochs=3, patience=2,
        seed=7, device="cpu", edge_attr=edge_attr,
    )
    repeated = predict_gat(model, features, edge_index, device="cpu", edge_attr=edge_attr)
    assert np.array_equal(returned, repeated)
