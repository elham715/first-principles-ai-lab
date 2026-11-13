# Milestone 04: Parameter-Efficient Fine-Tuning (LoRA & Adapters)

> **Context Anchor**: This folder houses the from-scratch implementations reproducing the mathematical foundations of this milestone.

## 📄 Primary Papers for this Milestone

- [`23_2021_hu_lora_low_rank_adaptation.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/05_Fine_Tuning/23_2021_hu_lora_low_rank_adaptation.pdf)
- [`24_2023_zhang_llama_adapter_zero_init_attention_grafting.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/05_Fine_Tuning/24_2023_zhang_llama_adapter_zero_init_attention_grafting.pdf)
- [`28_2024_li_mixlora_mixture_of_lora_experts.pdf`](file:///Users/admin/Documents/Elhams%20Universe/Artificial%20Intelligence/llm%20Intelligence/papers/05_Fine_Tuning/28_2024_li_mixlora_mixture_of_lora_experts.pdf)

---

### Mathematical Formulation
1. **LoRA Decomposition**:
   $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A, \quad B \in \mathbb{R}^{d_{out} \times r}, A \in \mathbb{R}^{r \times d_{in}}$$
   $$\text{Initialization: } A \sim \mathcal{N}(0, \sigma^2), B = 0 \implies \Delta W = 0 \text{ at step 0}$$

---

## 🛠️ Code Artifacts to Build

| Script Filename | Implementation Focus | Status |
|---|---|:---:|
| `lora_linear.py` | Custom PyTorch Linear module with frozen base weights W_0 and rank-r trainable decomposition B*A | ⚪ Pending |
| `zero_init_adapter.py` | Zero-initialized attention gating for disturbance-free step-0 grafting | ⚪ Pending |

---

## 🚀 Current Active Target

**First Paper to Reproduce**: `Paper 23: Edward J. Hu et al. (2021) — "LoRA: Low-Rank Adaptation of Large Language Models"`

- **Next Action**: Create `lora_linear.py` and verify with analytical gradients.
