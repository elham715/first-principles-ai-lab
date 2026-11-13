# 01. Sequence Foundations

From-scratch implementations of recurrent architectures, BPTT, gating, and additive attention.  
If you need a refresher on NumPy or PyTorch basics first, see [`../00_coding_foundations/`](../00_coding_foundations/README.md).

## Papers

1. Elman (1990) — *Finding Structure in Time*
2. Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*
3. Sutskever et al. (2014) — *Sequence to Sequence Learning with Neural Networks*
4. Bahdanau et al. (2014) — *Neural Machine Translation by Jointly Learning to Align and Translate*

## Core Equations

* **Elman recurrence:**
  $$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
* **LSTM cell update:**
  $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
* **Bahdanau additive attention:**
  $$e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j), \quad \alpha_{ij} = \text{softmax}(e_{i\cdot})_j, \quad c_i = \sum_j \alpha_{ij} h_j$$

## Implementations

| File | Description | Status |
|---|---|---|
| [`01_elman_rnn_numpy.py`](./01_elman_rnn_numpy.py) | NumPy Elman RNN with analytical BPTT, numerical gradcheck (`rel_err < 1e-10`), and gradient decay test | Done |
| [`01_elman_rnn_walkthrough.ipynb`](./01_elman_rnn_walkthrough.ipynb) | Notebook walkthrough of Elman RNN equations, BPTT, and vanishing gradient plot | Done |
| [`02_lstm_cell_from_scratch.py`](./02_lstm_cell_from_scratch.py) | PyTorch LSTM cell from raw tensor ops, CEC gradient highway test, and 1997 Adding Problem | Done |
| `03_seq2seq_encoder_decoder.py` | Multi-layer LSTM encoder-decoder with teacher forcing | Planned |
| `04_bahdanau_attention.py` | Additive attention decoder with alignment weights | Planned |
