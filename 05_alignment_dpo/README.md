# Milestone 05: Preference Optimization & Alignment (DPO)

> **Context Anchor**: This folder houses the from-scratch implementations reproducing the mathematical foundations of this milestone.

## 📄 Primary Papers for this Milestone

- [`35_2023_zhou_lima_less_is_more_for_alignment.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/07_Alignment_and_Reasoning/35_2023_zhou_lima_less_is_more_for_alignment.pdf)
- [`36_2023_rafailov_dpo_direct_preference_optimization.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/07_Alignment_and_Reasoning/36_2023_rafailov_dpo_direct_preference_optimization.pdf)

---

### Mathematical Formulation
1. **DPO Closed-Form Loss**:
   $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

---

## 🛠️ Code Artifacts to Build

| Script Filename | Implementation Focus | Status |
|---|---|:---:|
| `dpo_loss.py` | Closed-form Direct Preference Optimization loss function using reference policy logits | ⚪ Pending |
| `toy_dpo_training.py` | Training loop optimizing preferred vs dispreferred responses on pairwise text | ⚪ Pending |

---

## 🚀 Current Active Target

**First Paper to Reproduce**: `Paper 36: Rafael Rafailov et al. (2023) — "Direct Preference Optimization"`

- **Next Action**: Create `dpo_loss.py` and verify with analytical gradients.
