"""From-scratch PyTorch implementation of Sequence-to-Sequence Learning (Sutskever et al., 2014)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


class LSTMCellFromScratch(nn.Module):
    """Single-step LSTM cell implemented from raw tensor ops."""

    def __init__(self, input_dim: int, hidden_dim: int, forget_bias_init: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Fused projections for all 4 gates in PyTorch order: (i, f, g, o).
        self.W_ih = nn.Parameter(torch.empty(input_dim, 4 * hidden_dim))
        self.W_hh = nn.Parameter(torch.empty(hidden_dim, 4 * hidden_dim))
        self.b = nn.Parameter(torch.zeros(4 * hidden_dim))

        self.reset_parameters(forget_bias_init)

    def reset_parameters(self, forget_bias_init: float):
        bound = 1.0 / (self.hidden_dim ** 0.5)
        nn.init.uniform_(self.W_ih, -bound, bound)
        nn.init.uniform_(self.W_hh, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)
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

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class StackedLSTM(nn.Module):
    """Multi-layer LSTM stacked from independent LSTMCellFromScratch layers."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, forget_bias_init: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                LSTMCellFromScratch(
                    input_dim if layer == 0 else hidden_dim,
                    hidden_dim,
                    forget_bias_init=forget_bias_init,
                )
                for layer in range(num_layers)
            ]
        )

    def init_states(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        return [
            (
                torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
                torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            )
            for _ in range(self.num_layers)
        ]

    def step(
        self,
        x_t: torch.Tensor,
        states: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        if states is None:
            states = self.init_states(x_t.shape[0], x_t.device, x_t.dtype)

        h_in = x_t
        next_states = []
        for layer, cell in enumerate(self.layers):
            h_l, c_l = cell(h_in, states[layer])
            next_states.append((h_l, c_l))
            h_in = h_l
        return h_in, next_states

    def forward(
        self,
        x: torch.Tensor,
        states: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        B, T, _ = x.shape
        if states is None:
            states = self.init_states(B, x.device, x.dtype)

        outputs = []
        for t in range(T):
            top_h, states = self.step(x[:, t, :], states)
            outputs.append(top_h)
        return torch.stack(outputs, dim=1), states


class Encoder(nn.Module):
    """Encodes a source token sequence into a fixed-size multi-layer LSTM state."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        pad_idx: int = PAD_IDX,
        forget_bias_init: float = 1.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = StackedLSTM(
            embed_dim, hidden_dim, num_layers=num_layers, forget_bias_init=forget_bias_init
        )

    def forward(self, src_tokens: torch.Tensor):
        embedded = self.embedding(src_tokens)  # [B, T_src, E]
        _, final_states = self.lstm(embedded)
        return final_states


class Decoder(nn.Module):
    """2-layer LSTM decoder supporting teacher forcing and greedy autoregressive decoding."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        pad_idx: int = PAD_IDX,
        forget_bias_init: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = StackedLSTM(
            embed_dim, hidden_dim, num_layers=num_layers, forget_bias_init=forget_bias_init
        )
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def step(self, token_t: torch.Tensor, states: list[tuple[torch.Tensor, torch.Tensor]]):
        emb_t = self.embedding(token_t)  # [B, E]
        top_h, next_states = self.lstm.step(emb_t, states)
        logits = self.proj(top_h)  # [B, V]
        return logits, next_states

    def forward_teacher_forcing(
        self,
        tgt_in: torch.Tensor,
        encoder_states: list[tuple[torch.Tensor, torch.Tensor]],
    ):
        """Unroll decoder with ground-truth shifted target inputs (<SOS>, y_1, ..., y_{T-1})."""
        states = encoder_states
        logits_seq = []
        for t in range(tgt_in.shape[1]):
            logits_t, states = self.step(tgt_in[:, t], states)
            logits_seq.append(logits_t)
        return torch.stack(logits_seq, dim=1)  # [B, T_tgt, V]

    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_states: list[tuple[torch.Tensor, torch.Tensor]],
        max_len: int,
        sos_idx: int = SOS_IDX,
        eos_idx: int = EOS_IDX,
        pad_idx: int = PAD_IDX,
    ):
        """Autoregressively decode by feeding argmax predictions back as the next input."""
        B = encoder_states[0][0].shape[0]
        device = encoder_states[0][0].device
        states = encoder_states
        cur_token = torch.full((B,), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        preds = []
        logits_seq = []
        for _ in range(max_len):
            logits_t, states = self.step(cur_token, states)
            next_token = logits_t.argmax(dim=-1)
            # Keep finished sequences padded after their first <EOS>.
            out_token = torch.where(finished, torch.full_like(next_token, pad_idx), next_token)
            preds.append(out_token)
            logits_seq.append(logits_t)

            finished = finished | (next_token == eos_idx)
            cur_token = next_token
            if finished.all():
                break

        return torch.stack(preds, dim=1), torch.stack(logits_seq, dim=1)


class Seq2Seq(nn.Module):
    """Sutskever et al. (2014) encoder-decoder with fixed-size state transfer."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        sos_idx: int = SOS_IDX,
        eos_idx: int = EOS_IDX,
        pad_idx: int = PAD_IDX,
        forget_bias_init: float = 1.0,
    ):
        super().__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.encoder = Encoder(
            vocab_size, embed_dim, hidden_dim, num_layers=num_layers, pad_idx=pad_idx, forget_bias_init=forget_bias_init
        )
        self.decoder = Decoder(
            vocab_size, embed_dim, hidden_dim, num_layers=num_layers, pad_idx=pad_idx, forget_bias_init=forget_bias_init
        )

    def forward(self, src_tokens: torch.Tensor, tgt_in: torch.Tensor):
        encoder_states = self.encoder(src_tokens)
        return self.decoder.forward_teacher_forcing(tgt_in, encoder_states)

    @torch.no_grad()
    def greedy_decode(self, src_tokens: torch.Tensor, max_len: int):
        encoder_states = self.encoder(src_tokens)
        return self.decoder.greedy_decode(
            encoder_states,
            max_len=max_len,
            sos_idx=self.sos_idx,
            eos_idx=self.eos_idx,
            pad_idx=self.pad_idx,
        )


def make_translation_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int = 11,
    shift: int = 3,
    reverse_src: bool = False,
):
    """Generate a synthetic monotonic token-translation batch with <SOS> and <EOS>.

    Each content token x_t in [3, vocab_size - 1] maps 1-to-1 to y_t = shift_cipher(x_t).
    Reversing the source sequence places x_1 directly adjacent to y_1 across the bottleneck,
    matching the Sutskever et al. (2014) source-reversal trick.
    """
    num_content = vocab_size - 3
    x = torch.randint(3, vocab_size, (batch_size, seq_len))
    y = 3 + ((x - 3 + shift) % num_content)

    src = torch.flip(x, dims=[1]) if reverse_src else x
    sos_col = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long)
    eos_col = torch.full((batch_size, 1), EOS_IDX, dtype=torch.long)
    tgt_in = torch.cat([sos_col, y], dim=1)
    tgt_out = torch.cat([y, eos_col], dim=1)
    return src, tgt_in, tgt_out


def verify_shapes_and_decoding():
    """Verify state shapes, teacher-forcing step equivalence, and greedy decoding."""
    torch.manual_seed(0)
    B, T, V, E, H, L = 4, 6, 11, 16, 32, 2
    model = Seq2Seq(vocab_size=V, embed_dim=E, hidden_dim=H, num_layers=L)
    src, tgt_in, tgt_out = make_translation_batch(B, T, vocab_size=V, reverse_src=True)

    enc_states = model.encoder(src)
    assert len(enc_states) == L
    for h_l, c_l in enc_states:
        assert h_l.shape == (B, H) and c_l.shape == (B, H)

    tf_logits = model(src, tgt_in)
    assert tf_logits.shape == (B, T + 1, V)

    # Stepping the decoder manually with ground-truth tokens must match forward_teacher_forcing.
    states = enc_states
    step_logits = []
    for t in range(tgt_in.shape[1]):
        logits_t, states = model.decoder.step(tgt_in[:, t], states)
        step_logits.append(logits_t)
    step_logits = torch.stack(step_logits, dim=1)
    tf_diff = (tf_logits - step_logits).abs().max().item()

    model.eval()
    preds, greedy_logits = model.greedy_decode(src, max_len=T + 1)
    assert preds.shape[0] == B and 1 <= preds.shape[1] <= T + 1
    first_step_diff = (greedy_logits[:, 0, :] - tf_logits[:, 0, :]).abs().max().item()

    # Check backward pass flows through both encoder and decoder parameters.
    loss = F.cross_entropy(tf_logits.reshape(-1, V), tgt_out.reshape(-1))
    loss.backward()
    enc_grad = model.encoder.lstm.layers[0].W_hh.grad.norm().item()
    dec_grad = model.decoder.lstm.layers[1].W_hh.grad.norm().item()
    assert enc_grad > 0.0 and dec_grad > 0.0
    assert tf_diff < 1e-6 and first_step_diff < 1e-6

    print(
        f"Sanity check passed | tf_step_diff={tf_diff:.2e} | "
        f"greedy_t0_diff={first_step_diff:.2e} | enc_grad={enc_grad:.3e} | dec_grad={dec_grad:.3e}"
    )


def evaluate_model(model: Seq2Seq, seq_len: int, reverse_src: bool, vocab_size: int = 11, num_samples: int = 256):
    """Evaluate teacher-forced cross-entropy and greedy autoregressive accuracy."""
    model.eval()
    with torch.no_grad():
        src, tgt_in, tgt_out = make_translation_batch(
            num_samples, seq_len, vocab_size=vocab_size, reverse_src=reverse_src
        )
        tf_logits = model(src, tgt_in)
        tf_loss = F.cross_entropy(tf_logits.reshape(-1, vocab_size), tgt_out.reshape(-1)).item()

        max_len = tgt_out.shape[1]
        preds, _ = model.greedy_decode(src, max_len=max_len)
        if preds.shape[1] < max_len:
            pad_cols = torch.full(
                (num_samples, max_len - preds.shape[1]), PAD_IDX, dtype=torch.long, device=preds.device
            )
            preds = torch.cat([preds, pad_cols], dim=1)

        tok_acc = (preds[:, :seq_len] == tgt_out[:, :seq_len]).float().mean().item()
        seq_acc = (preds == tgt_out).all(dim=1).float().mean().item()
    return tf_loss, tok_acc, seq_acc


def train_seq2seq(
    seq_len: int,
    reverse_src: bool,
    vocab_size: int = 11,
    embed_dim: int = 24,
    hidden_dim: int = 64,
    steps: int = 450,
    batch_size: int = 64,
    lr: float = 5e-3,
    seed: int = 42,
):
    """Train a 2-layer Seq2Seq model with teacher forcing and evaluate greedy decoding."""
    torch.manual_seed(seed)
    model = Seq2Seq(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for _ in range(steps):
        model.train()
        src, tgt_in, tgt_out = make_translation_batch(
            batch_size, seq_len, vocab_size=vocab_size, reverse_src=reverse_src
        )
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_history.append(loss.item())

    tf_loss, tok_acc, seq_acc = evaluate_model(model, seq_len, reverse_src, vocab_size=vocab_size)
    return model, loss_history, (tf_loss, tok_acc, seq_acc)


def compare_reversal_and_bottleneck():
    """Demonstrate Sutskever et al. (2014) source reversal and fixed-context bottleneck."""
    print("\nSutskever et al. (2014) Reversal & Fixed-Context Bottleneck (2-layer LSTM, H=64):")
    results = {}
    for T in [8, 16, 20]:
        for reverse_src in [False, True]:
            _, _, (tf_loss, tok_acc, seq_acc) = train_seq2seq(
                seq_len=T, reverse_src=reverse_src, hidden_dim=64, steps=450
            )
            results[(T, reverse_src)] = (tf_loss, tok_acc, seq_acc)
            order = "reversed" if reverse_src else "normal  "
            print(
                f"  T={T:2d} | src={order} | tf_loss={tf_loss:.4f} | "
                f"greedy_tok_acc={tok_acc:.3f} | greedy_seq_acc={seq_acc:.3f}"
            )

    assert results[(8, True)][1] > 0.90, "T=8 reversed should solve the translation task."
    assert results[(20, True)][1] < results[(8, True)][1] - 0.30, (
        "Fixed-size context vector should degrade sharply between T=8 and T=20."
    )
    assert results[(16, True)][1] > results[(16, False)][1], (
        "Reversed source should outperform normal order at T=16."
    )
    assert results[(20, True)][1] > results[(20, False)][1], (
        "Reversed source should outperform normal order at T=20."
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    verify_shapes_and_decoding()
    compare_reversal_and_bottleneck()
