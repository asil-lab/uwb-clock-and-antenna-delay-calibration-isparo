# UWB Clock Synchronization and Antenna Delay Calibration

The work investigates joint clock synchronization and antenna-delay calibration for Ultra-Wideband (UWB) ranging in multi-agent lunar rover networks. The proposed method uses Least Squares (LS) estimation over a connected network and removes the inherent rank deficiency by anchoring one node to a known antenna delay. The experimental evaluation compares pairwise LS and centralized network-wide LS, followed by independent ranging verification.

The experimental results reported in the paper demonstrate calibrated ranging errors bounded between approximately −2.56 cm and +4.26 cm, with standard deviations between 0.31 cm and 3.44 cm across the reported ranging experiments.

## Prerequisites

Create a virtual Python environment and install the dependencies:

```sh
cd path/to/this/repository
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -f requirements.txt
```

Git clone the dataset:

```sh
cd path/to/this/repository
git clone https://github.com/asil-lab/Uwb-clock-and-antenna-delay-dataset data
```

## Citation

If you use this repository for scientific publication, we would appreciate citation of the following paper:

```latex
@inproceedings{Becoy2026,
  author    = {Becoy, Alexander James and Peternel, Luka and Rajan, Raj Thilak},
  title     = {Clock Synchronization and Antenna Delay Estimation for UWB-based Lunar Rover Networks},
  booktitle = {2026 International Conference on Space Robotics (iSpaRo)},
  year      = {2026}
}
```

## Issue

If you come across bugs, unintended functions, or have some points of improvement, please refer to the issues, and fill in your remarks.