import json
import os

import numpy as np
from pymatgen.core import Structure

from jax_xtal.config import load_config
from jax_xtal.predictor import predict_from_structures


def test_predict_from_structures():
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    config = load_config(os.path.join(root_dir, "configs", "default.json"))
    ckpt_path = os.path.join(root_dir, "checkpoints", "checkpoint.default.pkl")

    structures = []
    for basename in ["MgO_0", "MgO_1"]:
        path = os.path.join(root_dir, "data", "structures_dummy", f"{basename}.json")
        with open(path, "r") as f:
            structures.append(Structure.from_dict(json.load(f)))

    predictions = predict_from_structures(config=config, ckpt_path=ckpt_path, structures=structures)

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
