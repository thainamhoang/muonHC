# muonHC – Climate Downscaling with Hyperloop‑mHC and Muon

muonHC is a research project that tackles the **statistical downscaling of climate data** (ERA5 2‑m temperature, 5.625° → 1.40625°). Our goal is to surpass the current state‑of‑the‑art **GeoFAR** model (1.076 K RMSE) by combining three novel ingredients:

1. **Temporal conditioning** – a stack of adjacent time steps (±6 h) to exploit atmospheric dynamics.  
2. **Manifold‑constrained Hyper‑connections (mHC)** – doubly stochastic residual connections that guarantee stable gradient flow and allow parameter‑efficient Hyperloop recurrence.  
3. **Muon optimizer** – matrix‑orthogonalised gradient updates that improve generalisation on small datasets.

The project is conducted in two phases: first, isolated pilot experiments to validate each component; then, a combined model trained end‑to‑end.

---

## Repository Structure

```
muonHC/
├── README.md
├── requirements.txt
├── data/                         # Symlink or mount to ERA5 shards
├── constants/                    # Orography, land‑sea mask, etc.
│   ├── constants.npz
│   └── orography_hr.npy          # HR elevation for Geo‑INR (used later)
├── model/
│   ├── __init__.py
│   ├── fck.py                    # Fixed DCT basis (from GeoFAR)
│   ├── vit.py                    # Standard ViT (patch_size=1)
│   ├── geo_inr.py                # Geo‑INR: spherical harmonics + elevation → FiLM
│   ├── decoder.py                # Pixel‑shuffle decoder (from SSL pipeline)
│   ├── spectral_loss.py          # Frequency‑weighted loss
│   └── downscaling_model.py      # Assembles FCK → ViT → (GeoINR) → Decoder
├── dataset/
│   ├── downscaling_dataset.py    # Loads ERA5 with optional static 
│   └── temporal_dataset.py       # Loads ERA5 with optional temporal stacking
├── configs/
│   ├── cfgs_static.yaml          # Config for static baseline (Phase 1 Exp 1)
│   └──  cfgs_temporal.yaml       # Config for temporal model (Phase 1 Exp 1)
├── train_utils/
│   ├── metrics.py                # RMSE in Kelvin, bias, correlation
│   └── trainer.py                # Generic training loop with early stopping
├── outputs/                      # Experiment results (created automatically)
└──  training.py                  # Main training path
```

## Data Preparation

The code expects ERA5 2‑m temperature data stored as **sharded `.npy` files**, each containing a dictionary with keys `'lr'` (low‑resolution, shape `(32, 64)`) and `'hr'` (high‑resolution, shape `(128, 256)`). The values are in Kelvin. The dataset already used in our SSL pipeline is compatible.

The data can be downloaded [here on HuggingFace](https://huggingface.co/datasets/thainamhoang/era5-climate-learn).

---

# Phase 1 – Pilot Experiments

**Objective:** Validate that each of the three core ideas provides a **measurable, statistically significant improvement** over a shared GeoFAR‑like baseline.

All experiments use:
- **Backbone:** ViT with `embed_dim=128`, `depth=8`, `heads=4`, `patch_size=1` (token per pixel).  
- **Fixed frequency decomposition:** FCK with `n_coeff=64` (8×8 DCT basis).  
- **Decoder:** pixel‑shuffle (×4) with hidden dimension 256.  
- **Loss:** MSE (z‑score space). Spectral loss is available but disabled by default to isolate component gains.  
- **Optimizer:** AdamW (`lr=2e-4`, `weight_decay=1e-4`, cosine schedule).  
- **Training:** 50 epochs, early stopping with patience 10, batch size 32 (static) / 16 (temporal).  
- **Evaluation:** test‑set RMSE in Kelvin, averaged over 3–5 random seeds.

---

### Experiment 1 – Temporal Conditioning Gain

| Configuration | `cfgs_static.yaml` | `cfgs_temporal.yaml` |
|---------------|----------------------|------------------------|
| Input channels | 1 (current time) | 3 (t‑6 h, t, t+6 h) |
| FCK output | 64 frequency bands | 192 frequency bands |
| Dataset | `temporal=False` | `temporal=True` |

**Success criterion:** The temporal model achieves a test RMSE that is **at least 5 % lower** (p < 0.05) than the static baseline.

Run:
```bash
python training.py --config configs/cfgs_static.yaml
python training.py --config configs/cfgs_temporal.yaml
```

Results are saved to `outputs/static_baseline` and `outputs/temporal`.

---

### Experiment 2 – Muon Optimizer Efficiency

*(Config and script will be added after Phase 1.1; Muon is applied only to 2‑D weight matrices.)*

**Success criterion:** Muon reaches the same validation RMSE as AdamW in **fewer epochs**, or achieves a lower final RMSE, without overfitting.

---

### Experiment 3 – Hyperloop‑mHC Parameter Efficiency

*(Requires implementation of `hyperloop_mhc.py`.)*

**Architecture:**  
- Begin block: 2 standard layers  
- Middle block: 4 mHC layers (2 parallel streams, doubly stochastic mixing) looped `K=3` times → effective depth 16, unique depth 8.  
- End block: 2 standard layers  
Total unique parameters ~40 % less than a depth‑16 standard ViT.

**Baseline:** a standard ViT with depth 16, matched hidden dimensions.

**Success criterion:** Hyperloop‑mHC achieves **non‑inferior** test RMSE to the depth‑matched standard ViT, with stable gradient norms throughout training.

---

# Phase 2 – Full Combined Model

Once all three components are validated, we integrate them:

- **Backbone:** Hyperloop‑mHC ViT (from Phase 1.3)  
- **Conditioning:** FCK + Geo‑INR (FiLM from spherical harmonics and elevation)  
- **Input:** Temporal stack of 3 time steps  
- **Optimizer:** Muon on 2‑D parameters, AdamW on the rest  
- **Loss:** MSE + spectral loss (λ = 0.1)  
- **Training:** on full ERA5 training set, with thorough hyper‑parameter tuning

The model will be compared against the reproduced **GeoFAR** baseline (scratch ViT + FCK + Geo‑INR, static only) to quantify the cumulative improvement.

---

## Running the Code

1. Edit the config files to set `data.root_dir` to the correct path.  
2. Launch training:  
   ```bash
   python scripts/train.py --config scripts/config_static.yaml
   ```  
3. Monitor `outputs/<experiment>/best_model.pt` and console logs for validation RMSE.  
4. For multiple runs, change the `seed` in the config or pass it via command‑line override (OmegaConf supports merging).

---

## Logging & Reproducibility

The `Trainer` class stores:
- `train_losses`, `val_losses`, `val_rmses_k` per epoch.  
- The best model checkpoint.  

These are saved in the experiment’s output folder. We recommend using Weights & Biases (`wandb`) for later phases; the trainer can be easily extended to log to W&B by adding a few lines.

---

## References

- [GeoFAR](https://eceo-epfl.github.io/GeoFAR/): *Geography‑Informed Frequency‑Aware Super‑Resolution for Climate Data* (2025)  
- [mHC](https://arxiv.org/abs/2512.24880): *Manifold‑Constrained Hyper‑Connections* (2024)  
- [Muon](https://kellerjordan.github.io/posts/muon/): *MomentUm Orthogonalized by Newton-Schulz* (Keller Jordan et al., 2024)  
- [Newton‑Muon](https://arxiv.org/abs/2604.01472): *The Newton‑Muon Optimizer* (2026)

---
