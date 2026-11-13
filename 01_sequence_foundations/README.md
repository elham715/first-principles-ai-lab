# Milestone 01: Sequence Foundations (Recurrence to Attention)

> **Context Anchor**: This folder houses the from-scratch implementations reproducing the mathematical foundations of sequence modeling.  
> **Prerequisites**: New to Python OOP, NumPy, or PyTorch? Start in [`../00_coding_foundations/`](../00_coding_foundations/README.md).

## 📄 Primary Papers for this Milestone

1. **Elman (1990)** — *Finding Structure in Time* (Simple Recurrent Network & BPTT)
2. **Hochreiter & Schmidhuber (1997)** — *Long Short-Term Memory* (Constant Error Carousel & Gating)
3. **Sutskever et al. (2014)** — *Sequence to Sequence Learning with Neural Networks*
4. **Bahdanau et al. (2014)** — *Neural Machine Translation by Jointly Learning to Align and Translate*

---

### Mathematical Formulation
1. **Elman Recurrence**:
   $$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
2. **LSTM Constant Error Carousel**:
   $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
3. **Bahdanau Additive Attention**:
   $$e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j), \quad \alpha_{ij} = \text{softmax}(e_{i \cdot})_j, \quad c_i = \sum_j \alpha_{ij} h_j$$

---

## 🛠️ Paper Reproduction Artifacts

| Script / Notebook Filename | Implementation Focus | Status |
|---|---|:---:|
| [`01_elman_rnn_numpy.py`](./01_elman_rnn_numpy.py) | Pure NumPy implementation of Elman RNN with manual analytical BPTT backward pass and vanishing gradient proof | 🟢 **Completed & Verified** (GradCheck Error: $6.9 \times 10^{-11}$) |
| [`01_elman_rnn_walkthrough.ipynb`](./01_elman_rnn_walkthrough.ipynb) | Interactive step-by-step Jupyter walkthrough with analytical gradcheck and Matplotlib vanishing gradient curves | 🟢 **Completed & Verified** |
| `02_lstm_cell_from_scratch.py` | PyTorch `nn.Module` LSTM cell featuring fused projection weights and Constant Error Carousel (CEC) tracking | ⚪ Next Target |
| `03_seq2seq_encoder_decoder.py` | Encoder-Decoder LSTM with reversed source sequences and teacher forcing | ⚪ Pending |
| `04_bahdanau_attention.py` | Additive soft attention mechanism with dynamic alignment score heatmap generation | ⚪ Pending |

---

## 🚀 Current Active Target

**First Paper Reproduced**: `Paper 00: Jeffrey L. Elman (1990) — "Finding Structure in Time"` (Done ✅)  
**Next Paper to Reproduce**: `Paper 01: Hochreiter & Schmidhuber (1997) — "Long Short-Term Memory"` (LSTM)
