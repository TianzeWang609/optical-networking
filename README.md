# Optical Networking

Python experiments for optical-link modeling and routing and wavelength
allocation, from ASE noise and nonlinear interference to traffic optimization.
Code identifiers, output and plot labels use English; code comments use Chinese.

## Project layout

- `step1_ase.py` through `step25_traffic_vs_distance.py`: standalone teaching
  scripts that progressively introduce physical-link and network models.
- Root-level PNG files: saved plots from the teaching scripts.
- `optical_networking_refactored/`: reusable physical and network modules,
  an experiment runner, regression tests and reference results. See its
  [README](optical_networking_refactored/README.md) for model assumptions,
  experiment options and validation details.
- `step26_reference_physics.py` through `step30_reference_curves.py`: reference
  parameters, numerical GN integration and accumulation checks.
- `reference_physical_model.py`: adopted 32 GBd reference physical layer.
- `reference_network.py`: reference-physics adapter, modulation/FEC and plan audits.
- `step31_reference_network.py`: four-node reference-parameter allocation experiments.
- `network_scenarios.py` and `step32_network_traffic.py`: synthetic 12-node mesh,
  paired uniform/hotspot and fixed/mixed traffic experiments.
- [EXPERIMENTS_WEEK2.md](EXPERIMENTS_WEEK2.md): Chinese experiment guide, assumptions
  and interpretation of the new results.
- `python step33_online_routing.py` replays the Step 32 fixed-rate batches with
  a congestion-aware tie-break. See the [paired results and interpretation](results_step33/README.md).
- `python step34_hotspot_strength.py` sweeps nine hotspot probabilities and compares
  both online rules on 40 new seeds. See the [hotspot strength results](results_step34/README.md).
- `python step35_wavelength_conversion.py` compares no, A-only and network-wide
  wavelength conversion on the archived fixed-rate batches, including certified
  offline optima. See the [conversion results and model assumptions](results_step35/README.md).
- `python review_mixed_rates.py` repeats the Step 33–35 comparisons with paired
  50/100/150 Gbit/s demands. See the [mixed-rate results](results_mixed_rate_review/README.md).
- `python step36_endpoint_grooming.py` compares dedicated lightpaths and online
  same-endpoint grooming on those mixed batches. See the [admissions, wavelengths and rate classes](results_step36/README.md).

## Reference network and larger traffic experiments

Run from the project root:

```shell
python step31_reference_network.py --no-show
python step32_network_traffic.py --no-show
python -m unittest -v test_reference_network
```

Step 31 saves outputs in `results_step31/`. Step 32 saves inputs, individual plans,
solver bounds, aggregate statistics and plots in `results_step32/`. Its default
run uses 10 seeds, 4 static request counts, 4 traffic scenarios and 3 methods
(480 allocation runs). Each ILP solve has a 2-second solver limit; time-limited
feasible results are explicitly distinguished from proven optima. The limit
does not include model construction and is not a hard process deadline.

The physical model retains 80 full-load channels. The initial larger-network
pilot makes 8 wavelengths per link available to the generated requests; the
remaining channels are outside the admitted-demand metric. To change that
resource budget, pass `--wavelengths` without changing the physical channel count.
All requests consume a whole wavelength, including lower-rate requests.

For a quick separate run:

```shell
python step32_network_traffic.py --no-show --seeds 2 --loads 24 48 --output-dir work/mesh_smoke
```

The original nine regression tests remain available using the command below.
They also passed on Windows Python 3.13 during the reference-network integration.

To extend only the unproven ILPs in an existing completed run:

```shell
python step32_network_traffic.py --no-show --certify-existing --ilp-seconds 10
```

This reads the original saved configuration and requests, preserves the original
outputs, and saves the derived dataset in `results_step32/certified/`. Its timing
includes initial + retry time for retried ILPs. The supplied pilot has all 160
ILP instances certified after this pass; use the certified dataset for optimal
benchmark comparisons and the original dataset for the fixed 2-second budget.

## Requirements

Use Python 3.10 or later. From the project root, install the dependencies:

```shell
python -m pip install -r requirements.txt
```

## Run the teaching scripts

```shell
python step1_ase.py
python step25_traffic_vs_distance.py
```

Most scripts display plot windows and save PNG files in the current directory.
Close the plot windows to let a script finish. Later scripts build on and repeat
earlier calculations, so they may produce several plots and console sections.

## Run the refactored experiments

```shell
cd optical_networking_refactored
python run_experiments.py --experiment distance --no-show
```

Replace `distance` with `all` to run every experiment. Generated results go into
`optical_networking_refactored/results/`, which is excluded from Git. Validated
reference figures and numerical outputs are included in `example_results/`.

## Run the regression tests

From `optical_networking_refactored/`:

```shell
python -m unittest -v test_regression
```
