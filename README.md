# Smart Power Grid Agent

## 1) How to use this project (library + notebook + requirements)

This repository has two main entry points:

- **`smart_grid_gatpo_lib.py`**: a reusable Python library for training/inference with a GAT + PPO controller for smart grid optimal power flow.
- **`Smart_Grid_Agent.ipynb`**: a research-style notebook that explains the full method, trains/evaluates the agent, and generates analysis/visualizations.

### For a normal audience

- Think of this as an **AI controller for electricity grids**.
- The notebook shows how the AI learns to keep the grid stable and fair under difficult conditions (like supply shocks).
- The library is the “engine” you can reuse in your own scripts/projects.

### For experienced AI / power-system engineers

- The project implements **GAT-PPO** for constrained OPF-style dispatch on **`l2rpn_wcci_2022`**.
- It uses a **primal-dual + augmented Lagrangian** reward design
  (automatic constraint handling during RL training), with curriculum training
  and spatial clustering.
- The library exposes modular components for training and deployment:
  `YBusBuilder`, `BusClusterer`, `Grid2OpAdapter`, `PandapowerAdapter`, and
  `GATPO` (the repository API class name for the GAT-PPO controller).

### Environment and backend used

- **Python version**: Python 3.10+ is recommended (Google Colab default runtime is suitable).
- **Primary development/runtime environment**: Google Colab (CPU/GPU runtime), as documented in the notebook.
- **Grid simulation environment**: **Grid2Op** (`l2rpn_wcci_2022`).
- **Power-flow backend**: **LightSim2Grid**.

### Pip requirements

Core dependencies used directly in notebook/library:

```bash
pip install lightsim2grid grid2op networkx seaborn numpy pandas torch gymnasium matplotlib plotly scikit-learn
```

For strict reproducibility, pin versions in your own environment (for example via a `requirements.txt` file with exact versions).

Optional (for model graph visualization in notebook):

```bash
pip install torchviz
```

### Basic usage

1. Install requirements.
2. Open `Smart_Grid_Agent.ipynb` in Colab/Jupyter to reproduce training and analysis.
3. Import `smart_grid_gatpo_lib.py` in scripts to run inference or training loops with Grid2Op or pandapower adapters.

---

## 2) What the notebook covers

The notebook walks through:

1. Environment setup and package installation,
2. Grid environment/wrapper design,
3. GAT policy/value architecture,
4. PPO + (augmented) Lagrangian training loop,
5. Curriculum over different shock regimes,
6. Evaluation, baselines, ablations, and visual analytics.

---

## 3) Project overview

The project targets **real-time grid control** under operational constraints:

- Maintain voltage/thermal safety,
- Reduce blackout risk,
- Balance generation dispatch and renewable curtailment,
- Preserve fairness (load equity) under stress conditions.

It blends physics-informed graph learning with reinforcement learning to handle large, structured power-grid state/action spaces.

---

## 4) Repository structure

- `smart_grid_gatpo_lib.py` — reusable library implementation.
- `Smart_Grid_Agent.ipynb` — full experiment notebook (method + training + evaluation).
- `env_needed_csv/egypt_power_plants_processed.csv` — plant capacity/source data used by the environment logic.
- `checkpoints/` — model checkpoints.
- `output/` — generated plots and result files.

---

## 5) Notes

- The notebook is research-oriented and includes extensive diagnostics/plots.
- The library file is better for integration into production-like Python workflows.
