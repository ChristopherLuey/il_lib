"""Compute Mahalanobis calibration statistics from a trained checkpoint.

Runs all training observations through the policy's feature extractor,
computes mean and inverse covariance (precision) of the embeddings,
and saves to a .npz file for use with the MahalanobisReleaseDetector.

Usage:
    python scripts/calibrate_mahalanobis.py arch=dp_a1 robot=a1_iiil task=pnp \
        data_dir=<path> ckpt_path=<path> output_path=mahalanobis_stats.npz
"""

import numpy as np
import os
import sys
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from il_lib.utils.config_utils import register_omegaconf_resolvers
from il_lib.utils.training_utils import load_state_dict, load_torch
from il_lib.utils.convert_utils import any_to_torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    config_dir = os.path.join(os.path.dirname(__file__), "../il_lib/configs")
    config_dir = os.path.abspath(config_dir)

    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        overrides = [a for a in sys.argv[1:] if not a.startswith("output_path=")]
        output_path = "mahalanobis_stats.npz"
        for a in sys.argv[1:]:
            if a.startswith("output_path="):
                output_path = a.split("=", 1)[1]

        overrides += [
            "hydra.output_subdir=null",
            "hydra.run.dir=.",
            "hydra/job_logging=none",
            "hydra/hydra_logging=none",
        ]
        cfg = compose(config_name="base_config", overrides=overrides)

        register_omegaconf_resolvers()
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)

        print("[calibrate] instantiating module", flush=True)
        policy = instantiate(cfg.module, _recursive_=False)

        print(f"[calibrate] loading checkpoint {cfg.ckpt_path}", flush=True)
        ckpt = load_torch(cfg.ckpt_path, map_location="cpu")
        load_state_dict(policy, ckpt["state_dict"], strict=True)
        policy = policy.to("cuda")
        policy.eval()

        print("[calibrate] instantiating data module", flush=True)
        data_module = instantiate(cfg.data, _recursive_=False)
        data_module.setup("fit")
        train_dataset = data_module.train_dataset

        loader = DataLoader(
            train_dataset,
            batch_size=cfg.bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        all_embeddings = []
        print("[calibrate] extracting embeddings...", flush=True)
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = any_to_torch(batch, device="cuda")
                obs = policy.process_data(batch, extract_action=False)
                obs_feature = policy.encode_obs(obs)  # (B, T_O, D)
                embedding = obs_feature.mean(dim=1)  # (B, D)
                all_embeddings.append(embedding.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)  # (N, D)
        N, D = embeddings.shape
        print(f"[calibrate] collected {N} embeddings of dimension {D}", flush=True)

        if N < 2 * D:
            print(
                f"[WARNING] N={N} < 2*D={2*D}. Covariance may be poorly conditioned. "
                "Consider collecting more data or using regularization.",
                flush=True,
            )

        mean = embeddings.mean(axis=0)  # (D,)
        cov = np.cov(embeddings.T)  # (D, D)

        eps = 1e-6
        precision = np.linalg.inv(cov + eps * np.eye(D))  # (D, D)

        cond_number = np.linalg.cond(cov)
        print(f"[calibrate] covariance condition number: {cond_number:.2e}", flush=True)

        deltas = embeddings - mean  # (N, D)
        distances = np.sqrt(np.sum(deltas @ precision * deltas, axis=1))  # (N,)

        percentiles = [50, 75, 90, 95, 99]
        print("[calibrate] training set Mahalanobis distance percentiles:")
        for p in percentiles:
            val = np.percentile(distances, p)
            print(f"  {p}th: {val:.4f}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez(
            output_path,
            mean=mean,
            precision=precision,
            cov=cov,
            embedding_dim=D,
            n_samples=N,
            distance_percentiles=np.array(
                [np.percentile(distances, p) for p in percentiles]
            ),
            percentile_labels=np.array(percentiles),
        )
        print(f"[calibrate] saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
