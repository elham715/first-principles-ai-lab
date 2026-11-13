# 05. Preference Optimization & Alignment

Direct preference optimization and supervised alignment fundamentals.

## Papers

1. Zhou et al. (2023) — *LIMA: Less Is More for Alignment*
2. Rafailov et al. (2023) — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*

## Core Equation

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

## Planned Files

| File | Description | Status |
|---|---|---|
| `dpo_loss.py` | Closed-form DPO loss comparing policy and reference log-probabilities | Planned |
| `toy_dpo_training.py` | Pairwise preference optimization loop on synthetic data | Planned |
