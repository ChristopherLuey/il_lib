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
import websockets
from copy import deepcopy
from msgpack import Packer, unpackb
from omnigibson.macros import gm

import logging
logger = logging.getLogger(__name__)


class EmbeddingPolicyServer(WebsocketPolicyServer):
    """Extends WebsocketPolicyServer to also return observation embeddings."""

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in result:
                    self._policy.reset()
                    continue

                obs = deepcopy(result)

                infer_time = time.monotonic()
                action = self._policy.act(obs)
                infer_time = time.monotonic() - infer_time

                response = {
                    "action": action.cpu().numpy(),
                }
                embedding = getattr(self._policy, 'last_obs_embedding', None)
                if embedding is not None:
                    response["obs_embedding"] = embedding
                response["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logger.error(f"Error in connection from {websocket.remote_address}:\n{traceback.format_exc()}")
                if gm.DEBUG:
                    await websocket.send(traceback.format_exc())
                try:
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                except AttributeError:
                    await websocket.close(code=1011, reason="Internal server error")
                raise


def main():
    # Initialize Hydra with logging disabled for serve mode
    config_dir = os.path.join(os.path.dirname(__file__), "il_lib/configs")
    config_dir = os.path.abspath(config_dir)
    port = int(os.environ.get("IL_LIB_WEBSOCKET_PORT", "8000"))
    print(f"[serve.py] target websocket port={port}", flush=True)

    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        # Compose config with Hydra logging disabled
        print("[serve.py] composing config", flush=True)
        overrides = sys.argv[1:] + [
            "hydra.output_subdir=null",
            "hydra.run.dir=.",
            "hydra/job_logging=none",
            "hydra/hydra_logging=none"
        ]
        cfg = compose(config_name="base_config", overrides=overrides)
        print("[serve.py] config composed", flush=True)

        register_omegaconf_resolvers()
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        print("[serve.py] instantiating module", flush=True)
        policy = instantiate(cfg.module, _recursive_=False)
        print("[serve.py] module instantiated", flush=True)
        print(f"[serve.py] loading checkpoint {cfg.ckpt_path}", flush=True)
        ckpt = load_torch(
            cfg.ckpt_path,
            map_location="cpu",
        )
        print("[serve.py] checkpoint loaded", flush=True)
        load_state_dict(
            policy,
            ckpt["state_dict"],
            strict=True
        )
        print("[serve.py] state dict loaded", flush=True)
        policy = policy.to("cuda")
        print("[serve.py] module moved to cuda", flush=True)
        policy.eval()
        # instantiate wrapper for policy
        print("[serve.py] instantiating policy wrapper", flush=True)
        policy_wrapper = instantiate(cfg.policy_wrapper)
        policy_wrapper.policy = policy
        print("[serve.py] policy wrapper ready", flush=True)
        server = EmbeddingPolicyServer(
            policy=policy_wrapper,
            host="0.0.0.0",
            port=port,
        )
        print("[serve.py] entering serve_forever", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
