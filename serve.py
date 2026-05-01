import hydra
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from il_lib.policies import ResidualPolicy
from il_lib.utils.config_utils import register_omegaconf_resolvers
from il_lib.utils.training_utils import load_state_dict, load_torch
from omegaconf import OmegaConf
from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
import os
import sys


def main():
    # Initialize Hydra with logging disabled for serve mode
    config_dir = os.path.join(os.path.dirname(__file__), "il_lib/configs")
    config_dir = os.path.abspath(config_dir)
    
    # Also add iiil and OmniGibson config search paths (mirroring the SearchPathPlugin
    # which is not loaded when using initialize_config_dir)
    import omnigibson as og
    extra_search_paths = [f"file://{og.__path__[0]}/learning/configs"]
    try:
        import iiil as _iiil
        extra_search_paths.append(f"file://{_iiil.__path__[0]}/configs")
    except ImportError:
        pass
    search_path_overrides = [
        f"hydra.searchpath=[{','.join(extra_search_paths)}]"
    ]

    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        # Compose config with Hydra logging disabled
        overrides = sys.argv[1:] + search_path_overrides + [
            "hydra.output_subdir=null",
            "hydra.run.dir=.",
            "hydra/job_logging=none",
            "hydra/hydra_logging=none"
        ]
        cfg = compose(config_name="base_config", overrides=overrides)
        
        register_omegaconf_resolvers()
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        policy = instantiate(cfg.module, _recursive_=False)
        ckpt = load_torch(
            cfg.ckpt_path,
            map_location="cpu",
        )
        load_state_dict(
            policy,
            ckpt["state_dict"],
            strict=True
        )
        policy = policy.to("cuda")
        policy.eval()
        # instantiate wrapper for policy
        policy_wrapper = instantiate(cfg.policy_wrapper)
        policy_wrapper.policy = policy
        server = WebsocketPolicyServer(
            policy=policy_wrapper,
            host="0.0.0.0",
            port=8000,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
