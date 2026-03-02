from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from .infer import D_GLOBAL, D_REQ, N_REQ
from .models import (
    AlphaZeroModel,
    V_LINEAR_MIN,
    V_LINEAR_NORM_MIN,
    V_MAX,
    V_MIN,
    V_NORM_MAX,
    V_NORM_MIN,
    V_TAIL_COMPRESS_POWER,
)


@dataclass(frozen=True)
class TorchScriptExportArtifacts:
    controller_path: Path
    adversary_path: Path
    meta_path: Path
    model_version: int


def _extract_state_dict(ckpt_obj: Dict[str, torch.Tensor] | Dict[str, object]) -> Dict[str, torch.Tensor]:
    if "model_state" in ckpt_obj and isinstance(ckpt_obj["model_state"], dict):
        return {str(k): v for k, v in ckpt_obj["model_state"].items()}  # type: ignore[return-value]
    return {str(k): v for k, v in ckpt_obj.items() if torch.is_tensor(v)}  # type: ignore[return-value]


class _ControllerTSWrapper(nn.Module):
    def __init__(self, model: AlphaZeroModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        req_features: torch.Tensor,
        global_features: torch.Tensor,
        req_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        return self.model(
            req_features=req_features,
            global_features=global_features,
            player="controller",
            req_mask=req_mask,
            action_mask=action_mask,
        )


class _AdversaryTSWrapper(nn.Module):
    def __init__(self, model: AlphaZeroModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        req_features: torch.Tensor,
        global_features: torch.Tensor,
        req_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        return self.model(
            req_features=req_features,
            global_features=global_features,
            player="adversary",
            req_mask=req_mask,
            action_mask=action_mask,
        )


def export_torchscript_artifacts(
    *,
    checkpoint_path: Path | str,
    out_dir: Path | str,
    model_version: int,
    device: str = "cpu",
    num_actions_controller: int,
    num_actions_adversary: int,
) -> TorchScriptExportArtifacts:
    ckpt_path = Path(checkpoint_path)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)

    model = AlphaZeroModel(
        num_actions_controller=int(num_actions_controller),
        num_actions_adversary=int(num_actions_adversary),
    ).to(torch.device(device))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    controller_wrapper = _ControllerTSWrapper(model).to(torch.device(device)).eval()
    adversary_wrapper = _AdversaryTSWrapper(model).to(torch.device(device)).eval()

    req_feat = torch.zeros((1, N_REQ, D_REQ), dtype=torch.float32, device=torch.device(device))
    global_feat = torch.zeros((1, D_GLOBAL), dtype=torch.float32, device=torch.device(device))
    req_mask = torch.ones((1, N_REQ), dtype=torch.bool, device=torch.device(device))
    ctrl_action_mask = torch.ones((1, int(num_actions_controller)), dtype=torch.bool, device=torch.device(device))
    adv_action_mask = torch.ones((1, int(num_actions_adversary)), dtype=torch.bool, device=torch.device(device))

    with torch.inference_mode():
        controller_ts = torch.jit.trace(
            controller_wrapper,
            (req_feat, global_feat, req_mask, ctrl_action_mask),
            strict=False,
        )
        adversary_ts = torch.jit.trace(
            adversary_wrapper,
            (req_feat, global_feat, req_mask, adv_action_mask),
            strict=False,
        )

    model_version_i = int(model_version)
    controller_path = dst / f"selfplay_weights_gen_{model_version_i:06d}_controller.ts"
    adversary_path = dst / f"selfplay_weights_gen_{model_version_i:06d}_adversary.ts"
    meta_path = dst / f"selfplay_weights_gen_{model_version_i:06d}_meta.json"

    controller_ts.save(str(controller_path))
    adversary_ts.save(str(adversary_path))

    meta = {
        "model_version": model_version_i,
        "checkpoint_path": str(ckpt_path.resolve()),
        "N_REQ": int(N_REQ),
        "D_REQ": int(D_REQ),
        "D_GLOBAL": int(D_GLOBAL),
        "num_actions_controller": int(num_actions_controller),
        "num_actions_adversary": int(num_actions_adversary),
        "value": {
            "type": "scalar",
            "real_min": float(V_MIN),
            "real_max": float(V_MAX),
            "linear_real_min": float(V_LINEAR_MIN),
            "normalized_min": float(V_NORM_MIN),
            "normalized_max": float(V_NORM_MAX),
            "linear_normalized_min": float(V_LINEAR_NORM_MIN),
            "tail_power": float(V_TAIL_COMPRESS_POWER),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    return TorchScriptExportArtifacts(
        controller_path=controller_path,
        adversary_path=adversary_path,
        meta_path=meta_path,
        model_version=model_version_i,
    )
