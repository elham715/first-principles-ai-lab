# 03. Residual Connections & Normalization Dynamics

Experiments on identity skip connections, gradient variance across depth, and Pre-LN vs Post-LN normalization.

## Papers

1. He et al. (2015) — *Deep Residual Learning for Image Recognition*
2. Veit et al. (2016) — *Residual Networks Behave Like Ensembles of Relatively Shallow Networks*
3. Xiong et al. (2020) — *On Layer Normalization in the Transformer Architecture*
4. Wang et al. (2022) — *DeepNet: Scaling Transformers to 1,000 Layers*

## Core Equations

$$x_{l+1} = x_l + F(x_l) \implies \frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L}\left(I + \sum_{k=l}^{L-1} \frac{\partial F_k}{\partial x_l}\right)$$

## Planned Files

| File | Description | Status |
|---|---|---|
| `pre_vs_post_ln_analysis.py` | Gradient norm tracking across depth comparing Pre-LN and Post-LN | Planned |
| `ensemble_path_analysis.py` | Layer-dropping sensitivity test reproducing Veit et al. (2016) | Planned |
