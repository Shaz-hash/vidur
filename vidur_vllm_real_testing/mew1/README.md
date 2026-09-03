# Mew1 User-Space Deployment

This deployment uses no Docker and makes no system changes. Everything lives
under `/home/shaz/vidur`. Commands refuse to run on hosts other than `mew1`, as
users other than `shaz`, or with physical GPUs other than 2 and 3.

## Local Deployment

From the repository root on the local machine:

```bash
bash vidur_vllm_real_testing/mew1/deploy_from_local.sh
```

This transfers source, frozen simulator profiles, and the XL3/XL4 promoted
model bundles. It excludes Git metadata, environments, bulk simulator output,
native build products, caches, and SSH files.

## Remote Bootstrap

```bash
ssh mew1
tmux new -s vidur-bootstrap
bash /home/shaz/vidur/source/vidur-classical-search/vidur_vllm_real_testing/mew1/bootstrap_user_env.sh
```

The bootstrap pins uv 0.12.5, Python 3.12, CUDA-12.9 vLLM 0.13.0, and the
model/native dependencies. It installs the guarded scheduler patch, builds the
native GV3 module with at most eight jobs, runs unit and scheduler integration
tests, and writes environment/source manifests.

## GPU Validation

```bash
VIDUR_PHYSICAL_GPU=2 \
bash /home/shaz/vidur/source/vidur-classical-search/vidur_vllm_real_testing/mew1/validate_install.sh
```

The process sees physical GPU 2 as logical `cuda:0`. Validation refuses to run
if that GPU already has a compute process.

Run the end-to-end vLLM acceptance test after validation:

```bash
VIDUR_PHYSICAL_GPU=2 \
bash /home/shaz/vidur/source/vidur-classical-search/vidur_vllm_real_testing/mew1/gpu_vllm_smoke.sh
```

It downloads the small public `facebook/opt-125m` model into the user-owned
cache, starts pinned vLLM with `GV3Scheduler`, submits one real completion,
records the result, and stops the server. It never uses a GPU other than the
explicitly selected physical GPU.

## Server Modes

```bash
VIDUR_PHYSICAL_GPU=2 \
bash /home/shaz/vidur/source/vidur-classical-search/vidur_vllm_real_testing/mew1/launch_server.sh \
  sjf256 /home/shaz/vidur/models/Meta-Llama-3-8B
```

Available modes are `stock`, `shadow`, `active-validation`, `controller`,
`sjf256`, and `sjf512`. Shadow, active-validation, and controller also require
`VIDUR_VLLM_GV3_PLANNER=module:attribute`. The generic fail-closed planner
boundary is installed and tested, but the model-specific live GV3 planner
adapter must be supplied before controller mode can schedule real requests.
The exact Llama-3-8B weights are not stored in Git and must be placed under
`/home/shaz/vidur/models` separately.
