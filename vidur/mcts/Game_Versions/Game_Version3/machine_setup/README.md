# GV3 AWS Worker Setup

This directory contains the setup script for a new Game Version 3 network worker.

The script is intended to run on the AWS worker machine. It:

- installs basic Ubuntu packages when `apt-get` and `sudo` are available,
- clones or updates the Vidur repo,
- creates `/home/ubuntu/vidur/.venv`,
- installs runtime Python dependencies plus `torch` and `pybind11`,
- validates Python `>=3.10` and imports `torch`, `numpy`, `sklearn`, `ray`, `yaml`, and `fasteners`,
- skips native compilation by default,
- runs the prefill calibrator,
- copies `simulator_output/new_prefill_profile.csv` to `simulator_output/prefill_profile.csv`,
- validates the GV3 network client import.

Default remote run from the local machine:

```bash
ssh aws-gpu-Beta 'bash -s' < /home/shazer/Desktop/Research/Vidur/vidur/vidur/mcts/Game_Versions/Game_Version3/machine_setup/setup_aws_worker.sh
```

If the repo URL, branch, or target path differ:

```bash
ssh aws-gpu-Beta 'REPO_DIR=/home/ubuntu/vidur GIT_BRANCH=gv3-ssh-network GIT_REPO_URL=git@github.com:Shaz-hash/vidur.git bash -s' \
  < /home/shazer/Desktop/Research/Vidur/vidur/vidur/mcts/Game_Versions/Game_Version3/machine_setup/setup_aws_worker.sh
```

Useful overrides:

```bash
BUILD_NATIVE=1                 # build the optional GV3 native module
RUN_PREFILL_CALIBRATION=0      # skip prefill calibration
FORCE_PREFILL=1                # regenerate prefill profile even if it exists
INSTALL_SYSTEM_PACKAGES=0      # skip apt-get install
INSTALL_PROJECT_EDITABLE=1     # try pip install -e; default uses PYTHONPATH instead
TORCH_INSTALL_COMMAND="..."    # custom torch install command
```

After setup, verify SSH and the worker repo path in:

```text
vidur/mcts/Game_Versions/Game_Version3/Network/server/machines.json
```
