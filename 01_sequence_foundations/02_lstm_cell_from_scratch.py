"""From-scratch PyTorch implementation of LSTM (Hochreiter & Schmidhuber, 1997; Gers et al., 2000)."""

import torch
import torch.nn as nn


class LSTMCellFromScratch(nn.Module):
    """Single-step LSTM cell implemented from raw tensor ops."""

    def __init__(self, input_dim: int, hidden_dim: int, forget_bias_init: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Fused projections for all 4 gates in PyTorch order: (i, f, g, o)
        self.W_ih = nn.Parameter(torch.empty(input_dim, 4 * hidden_dim))
        self.W_hh = nn.Parameter(torch.empty(hidden_dim, 4 * hidden_dim))
        self.b = nn.Parameter(torch.zeros(4 * hidden_dim))

        self.reset_parameters(forget_bias_init)

    def reset_parameters(self, forget_bias_init: float):
        bound = 1.0 / (self.hidden_dim ** 0.5)
        nn.init.uniform_(self.W_ih, -bound, bound)
        nn.init.uniform_(self.W_hh, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)
        # Keep forget gate open near initialization so the Constant Error Carousel flows across long horizons.
        with torch.no_grad():
            self.b[self.hidden_dim : 2 * self.hidden_dim].fill_(forget_bias_init)

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None):
        B = x_t.shape[0]
        if state is None:
            h_prev = torch.zeros(B, self.hidden_dim, device=x_t.device, dtype=x_t.dtype)
            c_prev = torch.zeros(B, self.hidden_dim, device=x_t.device, dtype=x_t.dtype)
        else:
            h_prev, c_prev = state

        gates = x_t @ self.W_ih + h_prev @ self.W_hh + self.b  # [B, 4 * H]
        i_raw, f_raw, g_raw, o_raw = gates.chunk(4, dim=-1)

        i_t = torch.sigmoid(i_raw)
        f_t = torch.sigmoid(f_raw)
        g_t = torch.tanh(g_raw)
        o_t = torch.sigmoid(o_raw)

        # Constant Error Carousel: additive highway avoids repeated matrix multiplication by W_hh.
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class LSTMSequenceModel(nn.Module):
    """Unrolls LSTMCellFromScratch over (B, T, input_dim) sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, forget_bias_init: float = 1.0):
        super().__init__()
        self.cell = LSTMCellFromScratch(input_dim, hidden_dim, forget_bias_init=forget_bias_init)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, return_states: bool = False):
        B, T, _ = x.shape
        h_t = torch.zeros(B, self.cell.hidden_dim, device=x.device, dtype=x.dtype)
        c_t = torch.zeros(B, self.cell.hidden_dim, device=x.device, dtype=x.dtype)

        h_list, c_list = [], []
        for t in range(T):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            if return_states:
                h_t.retain_grad()
                c_t.retain_grad()
                h_list.append(h_t)
                c_list.append(c_t)

        logits = self.head(h_t)
        if return_states:
            return logits, h_list, c_list
        return logits


class VanillaRNNBaseline(nn.Module):
    """Minimal tanh RNN baseline for long-range gradient and memory comparisons."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_ih = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, return_states: bool = False):
        B, T, _ = x.shape
        h_t = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
        h_list = []

        for t in range(T):
            h_t = torch.tanh(self.W_ih(x[:, t, :]) + self.W_hh(h_t))
            if return_states:
                h_t.retain_grad()
                h_list.append(h_t)

        logits = self.head(h_t)
        if return_states:
            return logits, h_list
        return logits


def verify_against_pytorch_lstmcell():
    """Check forward outputs and parameter gradients against torch.nn.LSTMCell."""
    torch.manual_seed(0)
    B, T, D, H = 4, 6, 5, 8
    x = torch.randn(B, T, D)

    custom = LSTMCellFromScratch(D, H)
    ref = nn.LSTMCell(D, H)

    with torch.no_grad():
        ref.weight_ih.copy_(custom.W_ih.T)
        ref.weight_hh.copy_(custom.W_hh.T)
        ref.bias_ih.copy_(custom.b)
        ref.bias_hh.zero_()

    h_c, c_c = torch.zeros(B, H), torch.zeros(B, H)
    h_r, c_r = torch.zeros(B, H), torch.zeros(B, H)

    for t in range(T):
        h_c, c_c = custom(x[:, t, :], (h_c, c_c))
        h_r, c_r = ref(x[:, t, :], (h_r, c_r))

    h_diff = (h_c - h_r).abs().max().item()
    c_diff = (c_c - c_r).abs().max().item()

    (h_c.pow(2).sum()).backward()
    (h_r.pow(2).sum()).backward()
    grad_diff = (custom.W_hh.grad - ref.weight_hh.grad.T).abs().max().item()

    print(f"PyTorch LSTMCell equivalence | h_diff={h_diff:.2e} | c_diff={c_diff:.2e} | grad_diff={grad_diff:.2e}")
    assert h_diff < 1e-6 and c_diff < 1e-6 and grad_diff < 1e-5


def compare_gradient_highway():
    """Compare final-step gradient preservation at t=0 across sequence lengths."""
    torch.manual_seed(42)
    B, D, H, C = 4, 8, 16, 2
    print("\nFinal-step gradient magnitude reaching t=0 (loss only at t=T-1):")

    for T in [10, 25, 50, 100]:
        x = torch.randn(B, T, D)

        rnn = VanillaRNNBaseline(D, H, C)
        rnn_logits, rnn_h = rnn(x, return_states=True)
        rnn_logits.sum().backward()
        rnn_ratio = rnn_h[0].grad.norm().item() / (rnn_h[-1].grad.norm().item() + 1e-12)

        lstm = LSTMSequenceModel(D, H, C, forget_bias_init=3.0)
        lstm_logits, lstm_h, lstm_c = lstm(x, return_states=True)
        lstm_logits.sum().backward()
        cec_ratio = lstm_c[0].grad.norm().item() / (lstm_c[-1].grad.norm().item() + 1e-12)

        print(
            f"T={T:3d} | VanillaRNN ||dh_0||/||dh_T|| = {rnn_ratio:.3e} | "
            f"LSTM CEC ||dc_0||/||dc_T|| = {cec_ratio:.3e}"
        )


def train_adding_problem(seq_len: int = 50, steps: int = 700):
    """Hochreiter & Schmidhuber (1997) Adding Problem: sum two marked Uniform(0,1) values."""
    torch.manual_seed(7)
    batch_size = 64

    def sample_batch(n: int = batch_size):
        vals = torch.rand(n, seq_len, 1)
        mask = torch.zeros(n, seq_len, 1)
        p1 = torch.randint(0, seq_len // 2, (n,))
        p2 = torch.randint(seq_len // 2, seq_len, (n,))
        idx = torch.arange(n)
        mask[idx, p1, 0] = 1.0
        mask[idx, p2, 0] = 1.0
        x = torch.cat([vals, mask], dim=-1)          # [n, T, 2]
        targets = (vals * mask).sum(dim=1)           # [n, 1]
        return x, targets

    def eval_mse(model):
        model.eval()
        with torch.no_grad():
            x_val, y_val = sample_batch(n=512)
            return nn.functional.mse_loss(model(x_val), y_val).item()

    criterion = nn.MSELoss()
    # Predicting constant mean (1.0) yields Var(U1 + U2) = 1/12 + 1/12 = 0.1667.
    print(f"\n1997 Adding Problem (T={seq_len}, constant-mean baseline MSE=0.1667):")

    for name, model in [
        ("VanillaRNN", VanillaRNNBaseline(2, 32, 1)),
        ("LSTM(from_scratch)", LSTMSequenceModel(2, 32, 1, forget_bias_init=3.0)),
    ]:
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        for _ in range(steps):
            x_b, y_b = sample_batch()
            opt.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(f"  {name:<20} final MSE = {eval_mse(model):.5f}")


if __name__ == "__main__":
    verify_against_pytorch_lstmcell()
    compare_gradient_highway()
    train_adding_problem()
