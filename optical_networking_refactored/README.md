# Optical networking: refactored teaching baseline

This project reorganizes `step25_traffic_vs_distance.py` into independent physical,
network, and experiment modules. It preserves the uploaded script's equations,
parameters, traffic requests, resource states, and optimization objective.
Identifiers, output, plot labels, and docstrings are English; code comments are Chinese.

## Quick start in Windows / VS Code

Extract the entire ZIP, then open the `optical_networking_refactored` folder in VS Code.
Open a terminal in that folder and run:

```powershell
python -m pip install -r requirements.txt
python run_experiments.py
```

The default command runs **only the distance experiment**, reproducing step 25.
It saves `results/traffic_vs_distance.png` and `results/distance_results.json`.
Closing the figure window allows the command to finish. If packages are already
installed in the selected Python interpreter, the installation command is optional.

Use Python 3.10 or later. The user's Python 3.13 can use the standard dependencies
listed in `requirements.txt`; Windows Python 3.13 was not directly available for validation.
The tested environment is recorded in `VALIDATION.json` and `requirements-tested.txt`.

## Files

| File | Responsibility |
|---|---|
| `physical_model.py` | Configuration, ASE, SPM/XPM, SNR, Shannon capacity, modulation and net rate |
| `network_model.py` | Toy cases, K shortest simple paths, wavelength options, sequential allocation, exhaustive search, ILP and margin search |
| `run_experiments.py` | Explicit selection of experiments, console output, figures and numerical JSON results |
| `test_regression.py` | Nine regression checks against the established baseline |
| `example_results/` | Figures and numerical outputs generated during validation |
| `QUICKSTART_ZH.txt` | Short Chinese operating instructions |

Importing any of the three modules does not execute an experiment, open a figure,
or print the earlier teaching steps. Run `run_experiments.py` as the entry point.

## Select an experiment

```powershell
python run_experiments.py --experiment baseline
python run_experiments.py --experiment physical
python run_experiments.py --experiment allocation
python run_experiments.py --experiment power
python run_experiments.py --experiment margin
python run_experiments.py --experiment max-margin
python run_experiments.py --experiment distance
```

| Choice | What it reproduces |
|---|---|
| `baseline` | Physical constants, route SNR/rate and wavelength continuity |
| `physical` | Central-channel SNR-P, ideal capacity-P and ideal capacity-L |
| `allocation` | Candidate-route count, min/max wavelength rules, exhaustive and ILP optima |
| `power` | Step 22: optimal admitted traffic versus common power |
| `margin` | Step 23: power sweeps at 0, 2, 4 and 6 dB required SNR margin |
| `max-margin` | Step 24: maximum margin at fixed power for a 3000 Gbit/s target |
| `distance` | Step 25: distance scaling at 0 and 4 dB required margin |

Run all experiments without opening windows:

```powershell
python run_experiments.py --experiment all --no-show
```

Use `--output-dir my_results` to choose a different output folder. Each experiment
overwrites its own named output files on repeated runs. `example_results/` provides
the initial validated outputs; the default runtime destination is `results/`.

## Model and units

- Optical powers are total powers over both polarizations. Common launch power is
  per WDM channel. Wavelength resource identifiers are one-based; the physical
  central-channel index is zero-based.
- The baseline has 25 channels at 400 GHz spacing, 200 GBd symbol rate, 80 km spans,
  0.22 dB/km loss, 4.5 dB noise figure, and `epsilon = 0`.
- ASE uses the original high-gain approximation `F * h * nu * Rs * G` per span.
- Nonlinear noise uses `(Ns**(1 + epsilon) * X_spm + Ns * X_xpm_sum) * P**3`.
- Every route uses the conservative full-load central-channel estimate. Physical
  XPM is not recalculated from the number of currently free wavelength resources.
- The modulation thresholds and 20% FEC overhead are inherited from the uploaded
  script. These are model assumptions, not universal hardware thresholds.
- Shannon capacity is `2 * Rs * log2(1 + SNR)`. Modulation net rate is
  `Rs * bits_dp / (1 + fec_overhead)`, with both polarizations already in `bits_dp`.
- Links are undirected and share one resource state across both directions, as in
  the teaching example. A connection uses one continuous wavelength along a route.
  There is no conversion, regeneration, grooming, request splitting or departure.
- Each admitted request consumes a whole wavelength on each route link. The
  objective sums admitted demand, not the available net rates of the lightpaths.
  Background occupied resources are excluded from reported admitted traffic.
- Four candidate routes per request are retained. Offline optima apply to these
  candidates and the fixed physical model, and know the complete request list.

Network A has links AB=400, BD=400, AC=800, CD=800 and BC=1200 km. Its free
wavelength sets are AB={1,2}, BD={3,4}, AC={5,6}, CD={6,7} and BC={2,6}.
Network B changes only AC to {5,7}. Requests, in order, are CD, AD, BD, AD,
each demanding 1000 Gbit/s.

The original script's silent `int(length / 80)` truncation is replaced by a
validation check: each link must contain an integer number of complete spans.
All existing experiments already meet this condition, so their results are unchanged.
Model extensions involving partial spans require an explicit physical treatment.

## Editing and reuse

Change physical defaults in `PhysicalConfig`, topology/resource/request defaults
in `toy_case`, and experiment ranges in the selected `experiment_*` function.
Alternatively, pass configurations explicitly:

```python
from physical_model import PhysicalConfig, PhysicalModel
from network_model import OpticalNetwork, toy_case

# 显式创建参数和网络，避免依赖之前实验留下的状态
model = PhysicalModel(PhysicalConfig(epsilon=0.0))
links, free_wavelengths, requests = toy_case("A")
network = OpticalNetwork(model, links, k_candidates=4)
power_w = model.optimal_power_w(spans=10)

result = network.solve_ilp(
    requests, free_wavelengths, power_w, snr_margin_db=4.0
)
print(result.admitted_traffic_gbps)
```

`network.scaled(2)` returns a separate network at twice the original link lengths.
Allocation methods copy resource states; they do not mutate the caller's sets.
Solvers also return a plan so that admitted requests, routes, wavelengths and
selected modulation can be inspected. Different equally optimal plans are possible.

`maximum_snr_margin` searches margin at a **supplied fixed common power**.
At epsilon=0 with identical spans, the reference power of 6.166 dBm maximizes
SNR on all routes. At epsilon>0 the optimum depends on route span count; this
function does not jointly optimize power and margin. The distance experiment
always holds power fixed at the reference route's optimal power.

## Validation

```powershell
python -m unittest -v test_regression
```

Validation included all nine regression tests and execution of every CLI experiment.
The uploaded original script was also executed separately: the refactored model
matched its admitted traffic at all **1044 power/margin samples** (261 powers,
four margins) and all **12 distance/margin samples**. Physical coefficients matched
within a relative tolerance of 1e-12. See `VALIDATION.json` for the original SHA-256
and actual dependency versions. This establishes numerical equivalence for the
tested baseline, not experimental validation of the physical approximations.

Reference results:

- Optimal power: 6.165990525 dBm; 800 km maximum SNR: 18.215407990 dB.
- Central-channel Shannon capacity at that point: 2.429050 Tbit/s.
- Scenario A/B offline optimum: 3000 Gbit/s; 7 binary variables, 10 constraints.
- Maximum required margin for 3000 Gbit/s: approximately 4.395 dB.
- 3000 Gbit/s power sample ranges at margins 0/2/4 dB: 0.1-10.5,
  2.2-9.3 and 4.8-7.4 dBm. At 6 dB, maximum admitted traffic is 2000 Gbit/s.

| Distance scale | Traffic at 0 dB margin (Gbit/s) | Traffic at 4 dB margin (Gbit/s) |
|---:|---:|---:|
| 1 | 3000 | 3000 |
| 2 | 3000 | 2000 |
| 3 | 2000 | 1000 |
| 4 | 2000 | 1000 |
| 5 | 2000 | 0 |
| 6 | 1000 | 0 |

Distance-plot connecting lines are visual guides between sampled networks. Power
plateau endpoints are sampled at 0.1 dB resolution. These results concern the
specified requests and initial resources, not total network capacity.
