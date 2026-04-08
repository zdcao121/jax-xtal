import argparse

from jax_xtal.config import load_config, Config
from jax_xtal.predictor import predict_from_structures_dir


def main(config: Config, ckpt_path: str, structures_dir: str, output: str):
    list_ids, predictions = predict_from_structures_dir(
        config=config, ckpt_path=ckpt_path, structures_dir=structures_dir
    )
    with open(output, "w") as f:
        for idx, prediction in zip(list_ids, predictions):
            f.write(f"{idx},{prediction}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="pre-trained parameters")
    parser.add_argument("--config", required=True, type=str, help="json config used for training")
    parser.add_argument(
        "--structures_dir", required=True, type=str, help="directory of json files to be predicted"
    )
    parser.add_argument("--output", required=True, type=str, help="path to output predictions")
    args = parser.parse_args()

    config = load_config(args.config)

    main(config, args.checkpoint, args.structures_dir, args.output)
