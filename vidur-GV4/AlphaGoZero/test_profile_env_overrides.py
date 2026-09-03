from __future__ import annotations

from vidur.AlphaGoZero.deploy import _agz_env_prefix
from vidur.Game_Version3.config import LegacyMCTSBridgeConfig, _default_sim_cli_args


def test_predictor_profile_environment_overrides(monkeypatch) -> None:
    values = {
        "VIDUR_GV3_COMPUTE_INPUT_FILE": "/profiles/mlp.csv",
        "VIDUR_GV3_ATTENTION_INPUT_FILE": "/profiles/attention.csv",
        "VIDUR_GV3_PREDICTOR_CACHE_DIR": "/profiles/cache",
        "VIDUR_GV3_PREFILL_PROFILE_PATH": "/profiles/prefill_profile.csv",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    args = _default_sim_cli_args()
    assert args[args.index(
        "--random_forest_execution_time_predictor_config_compute_input_file"
    ) + 1] == values["VIDUR_GV3_COMPUTE_INPUT_FILE"]
    assert args[args.index(
        "--random_forest_execution_time_predictor_config_attention_input_file"
    ) + 1] == values["VIDUR_GV3_ATTENTION_INPUT_FILE"]
    assert args[args.index(
        "--random_forest_execution_time_predictor_config_cache_dir"
    ) + 1] == values["VIDUR_GV3_PREDICTOR_CACHE_DIR"]
    assert (
        LegacyMCTSBridgeConfig().prefill_profile_path
        == values["VIDUR_GV3_PREFILL_PROFILE_PATH"]
    )

    prefix = _agz_env_prefix()
    for key, value in values.items():
        assert f"{key}={value}" in prefix
