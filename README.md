# Monolith Decomposition via Multi-Objective Evolutionary Optimization

Research code for the undergraduate thesis *"Evaluación y Mejora Experimental de un Algoritmo Evolutivo Multiobjetivo para la Descomposición de Monolitos en Microservicios"* (UNICEN, 2026).

This repository accompanies a short paper accepted at **GECCO 2026** (RLA track): *Database-Aware Monolith Decomposition via Multi-Objective Optimization*.

---

## What this is

The project applies a multi-objective genetic algorithm (NSGA-III) to the problem of decomposing a monolithic Java application into microservice candidates. Each candidate decomposition is evaluated against four architectural objectives: structural modularity (SM), inter-call percentage (ICP), interface number (IN), and microservice size balance (NED).

The main contribution is a **canonical, partition-based representation** that eliminates label-permutation redundancy from the original label-based encoding, along with redesigned variation operators and a systematic experimental evaluation of the impact of these changes.

---

## Repository structure

```
src/
  algorithm/                    # Core evolutionary algorithm, NSGA-III loop, DEAP configuration
    representations/            # Pluggable solution representations (original/canonical)
    fitness.py                  # Objective functions: SM, ICP, IN, NED
  experiments/                  # Experiment runners and analysis modules
    run_montecarlo_jpetstore.py    # Monte Carlo space exploration
    run_sensitivity_jpetstore.py   # Sobol/Saltelli sensitivity analysis  
    run_redundancy_jpetstore.py    # Redundancy quantification
    montecarlo/                 # Monte Carlo implementation
    redundancy/                 # Redundancy analysis code and data
    sensitivity/                # Sensitivity analysis code and data
  monoliths/                    # Instance metadata and dependency graphs
    jpetstore/                  # JPetStore instance data
    cargo/                      # CargoTracker instance data  
    acmeair/                    # AcmeAir instance data
```

Large precomputed files are not stored in this repository. See the **Data** section below.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.\.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Set the project root in your Python path before running any script:

```bash
export PYTHONPATH=$(pwd)/src     # Linux/macOS
set PYTHONPATH=%cd%\src          # Windows
```

### Dependencies

- `deap==1.4`
- `pymoo==0.6.1.5`
- `SALib`
- `numpy`, `pandas`, `matplotlib`, `scipy`, `seaborn`
- `networkx`, `tabulate`

---

## Running experiments

Each experiment is self-contained. Run from the project root.

**Monte Carlo space exploration** (used to estimate the HV reference point):
```bash
python src/experiments/run_montecarlo_jpetstore.py
```

**Sensitivity analysis** (Sobol/Saltelli over μ, λ/μ, p_mut):
```bash
python src/experiments/run_sensitivity_jpetstore.py
```

**Redundancy analysis** (quantifies label-permutation redundancy in Pareto fronts):
```bash
python src/experiments/run_redundancy_jpetstore.py
```

---

## Switching representations

The representation used by the algorithm is controlled via the `new_representation` configuration parameter. Available options:

- `false` — label-based vector encoding (baseline)
- `true` — partition-based frozenset encoding (proposed)

This is loaded dynamically at runtime via `algorithm.representations`, which acts as a common facade. The evolutionary loop (`ea.py`) operates against stable aliases registered in the DEAP Toolbox and has no direct dependency on the active representation.

---

## Data

Precomputed experiment outputs are not included in this repository due to file size. Expected paths:

- `src/experiments/sensitivity/data/` — Saltelli samples, HV outputs
- `src/experiments/redundancy/data/` — redundancy CSVs, HV-by-generation files

If you need access to the raw data, contact the authors.

---

## Monolith instances

| Instance     | Classes (N) | DB Tables | B(N)        |
|--------------|-------------|-----------|-------------|
| JPetStore    | 24          | 13        | 4.46 × 10¹⁷ |
| AcmeAir      | 32          | 5         | 1.28 × 10³⁵ |
| CargoTracker | 54          | —         | 2.90 × 10⁷⁶ |

Dependency graphs and metadata are stored under `src/monoliths/`.

---

## Citation

If you use this code, please cite:

```
Anonymous Author(s). Database-Aware Monolith Decomposition via Multi-Objective Optimization.
In Proceedings of GECCO 2026. ACM, 2026.
```

*(Author list to be updated upon camera-ready submission.)*

---

## Authors

- Ana Martínez Saucedo — UADE / CONICET
- Delfina Ferreri — UNICEN
- Guillermo Rodríguez — CONICET
- Virginia Yannibelli — CONICET