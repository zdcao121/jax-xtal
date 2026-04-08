from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from pymatgen.core import Structure

from jax_xtal.config import Config
from jax_xtal.data import (
    AtomFeaturizer,
    Batch,
    BondFeaturizer,
    collate_pool,
    create_dataset,
    create_dataset_from_structures,
)
from jax_xtal.model import get_model_fn_t
from jax_xtal.train_utils import restore_checkpoint


def _predict_dataset(
    config: Config,
    ckpt_path: str,
    dataset,
    num_initial_atom_features: int,
) -> np.ndarray:
    if len(dataset) == 0:
        return np.array([], dtype=np.float32)

    model_fn_t = get_model_fn_t(
        num_initial_atom_features=num_initial_atom_features,
        num_atom_features=config.num_atom_features,
        num_bond_features=config.num_bond_features,
        num_convs=config.num_convs,
        num_hidden_layers=config.num_hidden_layers,
        num_hidden_features=config.num_hidden_features,
        max_num_neighbors=config.max_num_neighbors,
        batch_size=config.batch_size,
    )
    model = hk.without_apply_rng(model_fn_t)

    params, state, normalizer = restore_checkpoint(ckpt_path)

    @jax.jit
    def predict_one_step(batch: Batch) -> jnp.ndarray:
        predictions, _ = model.apply(params, state, batch, is_training=False)
        return predictions

    batch_size = config.batch_size
    steps_per_epoch = (len(dataset) + batch_size - 1) // batch_size
    predictions = []
    for i in range(steps_per_epoch):
        batch = collate_pool(dataset[i * batch_size: min(len(dataset), (i + 1) * batch_size)], False)
        preds = predict_one_step(batch)
        predictions.append(preds)

    predictions = jnp.concatenate(predictions)
    denormed_preds = normalizer.denormalize(predictions)
    return np.asarray(denormed_preds[:, 0], dtype=np.float32)


def predict_from_structures(config: Config, ckpt_path: str, structures: List[Structure]) -> np.ndarray:
    atom_featurizer = AtomFeaturizer(atom_features_json=config.atom_init_features_path)
    bond_featurizer = BondFeaturizer(
        dmin=config.dmin, dmax=config.cutoff, num_filters=config.num_bond_features
    )
    dataset = create_dataset_from_structures(
        structures=structures,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        max_num_neighbors=config.max_num_neighbors,
        cutoff=config.cutoff,
    )
    return _predict_dataset(
        config=config,
        ckpt_path=ckpt_path,
        dataset=dataset,
        num_initial_atom_features=atom_featurizer.num_initial_atom_features,
    )


def predict_from_structures_dir(
    config: Config, ckpt_path: str, structures_dir: str
) -> Tuple[List[str], np.ndarray]:
    atom_featurizer = AtomFeaturizer(atom_features_json=config.atom_init_features_path)
    bond_featurizer = BondFeaturizer(
        dmin=config.dmin, dmax=config.cutoff, num_filters=config.num_bond_features
    )
    dataset, list_ids = create_dataset(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        structures_dir=structures_dir,
        targets_csv_path="",
        max_num_neighbors=config.max_num_neighbors,
        cutoff=config.cutoff,
        is_training=False,
        seed=config.seed,
        n_jobs=config.n_jobs,
    )
    predictions = _predict_dataset(
        config=config,
        ckpt_path=ckpt_path,
        dataset=dataset,
        num_initial_atom_features=atom_featurizer.num_initial_atom_features,
    )
    return list_ids, predictions
