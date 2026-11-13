# Milestone 03: The Residual Stream Highway & Normalization Dynamics

> **Context Anchor**: This folder houses the from-scratch implementations reproducing the mathematical foundations of this milestone.

## 📄 Primary Papers for this Milestone

- [`05_2015_he_resnet_skip_connections.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/03_Residuals/05_2015_he_resnet_skip_connections.pdf)
- [`07_2016_veit_resnets_behave_like_ensembles.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/03_Residuals/07_2016_veit_resnets_behave_like_ensembles.pdf)
- [`09_2020_xiong_pre_ln_layer_normalization_transformers.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/03_Residuals/09_2020_xiong_pre_ln_layer_normalization_transformers.pdf)
- [`15_2022_wang_deepnet_1000_layers.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/03_Residuals/15_2022_wang_deepnet_1000_layers.pdf)

---

### Mathematical Formulation
1. **Identity Skip Connection**:
   $$x_{l+1} = x_l + F(x_l) \implies \frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L}\left(I + \sum_{k=l}^{L-1} \frac{\partial F_k}{\partial x_l}\right)$$
2. **Pre-LN Gradient Bound**:
   $$\left\|\frac{\partial \mathcal{L}}{\partial W_l}\right\| = \mathcal{O}\left(\frac{1}{\sqrt{L}}\right) \quad \text{vs Post-LN exploding at initialization}$$

---

## 🛠️ Code Artifacts to Build

| Script Filename | Implementation Focus | Status |
|---|---|:---:|
| `pre_vs_post_ln_analysis.py` | Empirical gradient norm tracking at initialization across 50 layers demonstrating Pre-LN stability | ⚪ Pending |
| `ensemble_path_analysis.py` | Veit ensemble pathway deletion test: dropping random layers to measure accuracy drop | ⚪ Pending |

---

## 🚀 Current Active Target

**First Paper to Reproduce**: `Paper 05: Kaiming He et al. (2015) — "Deep Residual Learning for Image Recognition"`

- **Next Action**: Create `pre_vs_post_ln_analysis.py` and verify with analytical gradients.
