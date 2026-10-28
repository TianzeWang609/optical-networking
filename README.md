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
