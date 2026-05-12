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
import time
import traceback
from copy import deepcopy

import torch

from omnigibson.learning.utils import network_utils as net_utils


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _policy_aux_payload(policy_wrapper):
    policy = getattr(policy_wrapper, "policy", None)
    embedding = getattr(policy_wrapper, "_last_obs_embedding", None)
    if embedding is None and policy is not None:
        embedding = getattr(policy, "_last_obs_embedding", None)
    if embedding is None and hasattr(policy_wrapper, "_base_obs_embedding"):
        embedding = getattr(policy_wrapper, "_base_obs_embedding", None)

    payload = {}
    embedding_np = _to_numpy(embedding)
    if embedding_np is not None:
        payload["obs_embedding"] = embedding_np
    return payload


class AuxWebsocketPolicyServer(WebsocketPolicyServer):
    """Websocket server that returns action plus DP auxiliary signals.

    OmniGibson's default server only sends {"action": ...}. I3L's
    MahalanobisDetector needs the diffusion policy observation embedding, so
    this subclass preserves the same protocol and adds "obs_embedding" when the
    wrapped policy exposes it.
    """

    async def _handler(self, websocket):
        net_utils.logger.info(f"Connection from {websocket.remote_address} opened")
        packer = net_utils.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = net_utils.unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in result:
                    self._policy.reset()
                    continue

                obs = deepcopy(result)

                infer_time = time.monotonic()
                action_tensor = self._policy.act(obs)
                infer_time = time.monotonic() - infer_time

                action = {
                    "action": action_tensor.cpu().numpy(),
                }
                action.update(_policy_aux_payload(self._policy))
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except net_utils.websockets.ConnectionClosed:
                net_utils.logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                net_utils.logger.error(
                    f"Error in connection from {websocket.remote_address}:\n{traceback.format_exc()}"
                )
                if net_utils.gm.DEBUG:
                    await websocket.send(traceback.format_exc())
                try:
                    await websocket.close(
                        code=net_utils.websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                except AttributeError:
                    await websocket.close(code=1011, reason="Internal server error")
                raise


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
        server = AuxWebsocketPolicyServer(
            policy=policy_wrapper,
            host="0.0.0.0",
            port=8000,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
