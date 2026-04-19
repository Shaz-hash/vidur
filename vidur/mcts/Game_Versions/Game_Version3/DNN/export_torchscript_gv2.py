from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from .value_models import AlphaZeroModel
from .dnn_spec import DNNGameSpec

@dataclass(frozen=True)
class TorchScriptArtifactsGV2:
    controller_path: Path
    adversary_path: Path
    model_version: int

class _CtrlWrap(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, p_feat, d_feat, g_feat, p_mask, d_mask, a_mask):
        return self.model(
            player="controller",
            prefill_req_features=p_feat, decode_req_features=d_feat, global_features=g_feat,
            prefill_req_mask=p_mask, decode_req_mask=d_mask, action_mask=a_mask,
        )

class _AdvWrap(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, p_feat, d_feat, g_feat, p_mask, d_mask, a_mask):
        return self.model(
            player="adversary",
            prefill_req_features=p_feat, decode_req_features=d_feat, global_features=g_feat,
            prefill_req_mask=p_mask, decode_req_mask=d_mask, action_mask=a_mask,
        )

def export_torchscript_artifacts_gv2(*, checkpoint_path: Path, out_dir: Path, model_version: int, spec: DNNGameSpec, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    model = AlphaZeroModel(spec=spec).to(torch.device(device))
    model.load_state_dict(state, strict=True)
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)

    p_feat = torch.zeros((1, spec.n_prefill_req, spec.d_prefill_req), dtype=torch.float32)
    d_feat = torch.zeros((1, spec.n_decode_req, spec.d_decode_req), dtype=torch.float32)
    g_feat = torch.zeros((1, spec.d_global), dtype=torch.float32)
    p_mask = torch.ones((1, spec.n_prefill_req), dtype=torch.bool)
    d_mask = torch.ones((1, spec.n_decode_req), dtype=torch.bool)
    a_ctrl = torch.ones((1, spec.num_actions_controller), dtype=torch.bool)
    a_adv = torch.ones((1, spec.num_actions_adversary), dtype=torch.bool)

    with torch.inference_mode():
        ctrl_ts = torch.jit.trace(_CtrlWrap(model), (p_feat, d_feat, g_feat, p_mask, d_mask, a_ctrl), strict=False)
        adv_ts = torch.jit.trace(_AdvWrap(model), (p_feat, d_feat, g_feat, p_mask, d_mask, a_adv), strict=False)

    ctrl_path = out_dir / f"gv2_{int(model_version):06d}_controller.ts"
    adv_path = out_dir / f"gv2_{int(model_version):06d}_adversary.ts"
    ctrl_ts.save(str(ctrl_path))
    adv_ts.save(str(adv_path))

    return TorchScriptArtifactsGV2(controller_path=ctrl_path, adversary_path=adv_path, model_version=int(model_version))
