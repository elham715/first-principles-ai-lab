"""From-scratch PyTorch implementation of Bahdanau Attention (Bahdanau et al., 2014)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
CONTENT_START = 3


class GRUCellFromScratch(nn.Module):
    """Single-step GRU cell from raw tensor ops (Cho et al., 2014 / Bahdanau et al., 2014)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Fused projections for update (z) and reset (r) gates
        self.W_xzr = nn.Parameter(torch.empty(input_dim, 2 * hidden_dim))
        self.U_hzr = nn.Parameter(torch.empty(hidden_dim, 2 * hidden_dim))
        self.b_zr = nn.Parameter(torch.zeros(2 * hidden_dim))

        # Candidate state projection (reset applied before recurrent matmul as in Appendix A.2)
        self.W_xh = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.U_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / (self.hidden_dim ** 0.5)
        for p in (self.W_xzr, self.U_hzr, self.W_xh, self.U_hh):
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros(
                x_t.shape[0], self.hidden_dim, device=x_t.device, dtype=x_t.dtype
            )

        zr = torch.sigmoid(x_t @ self.W_xzr + h_prev @ self.U_hzr + self.b_zr)
        z_t, r_t = zr.chunk(2, dim=-1)
        h_tilde = torch.tanh(x_t @ self.W_xh + (r_t * h_prev) @ self.U_hh + self.b_h)
        return (1.0 - z_t) * h_prev + z_t * h_tilde


class BidirectionalGRUEncoder(nn.Module):
    """Bidirectional GRU encoder producing concatenated annotations [h_fwd; h_bwd]."""

    def __init__(self, vocab_size: int, emb_dim: int, enc_dim: int, dec_dim: int):
        super().__init__()
        self.enc_dim = enc_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.fwd_cell = GRUCellFromScratch(emb_dim, enc_dim)
        self.bwd_cell = GRUCellFromScratch(emb_dim, enc_dim)
        # Initialize decoder state s_0 from first backward state as in Eq. (A.2)
        self.init_decoder = nn.Linear(enc_dim, dec_dim)

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T_x = src.shape
        x = self.embedding(src)  # [B, T_x, E]

        if src_mask is None:
            src_mask = torch.ones(B, T_x, dtype=torch.bool, device=src.device)

        h_fwd = torch.zeros(B, self.enc_dim, device=src.device, dtype=x.dtype)
        fwd_states = []
        for j in range(T_x):
            m_j = src_mask[:, j : j + 1].to(x.dtype)
            h_next = self.fwd_cell(x[:, j], h_fwd)
            h_fwd = m_j * h_next + (1.0 - m_j) * h_fwd
            fwd_states.append(h_fwd)

        h_bwd = torch.zeros(B, self.enc_dim, device=src.device, dtype=x.dtype)
        bwd_states: list[torch.Tensor] = [h_bwd] * T_x
        for j in range(T_x - 1, -1, -1):
            # Hold backward state unchanged on right-padded tokens so h_bwd[0] stays clean
            m_j = src_mask[:, j : j + 1].to(x.dtype)
            h_next = self.bwd_cell(x[:, j], h_bwd)
            h_bwd = m_j * h_next + (1.0 - m_j) * h_bwd
            bwd_states[j] = h_bwd

        annotations = torch.cat(
            [torch.stack(fwd_states, dim=1), torch.stack(bwd_states, dim=1)], dim=-1
        )  # [B, T_x, 2 * enc_dim]
        s0 = torch.tanh(self.init_decoder(bwd_states[0]))  # [B, dec_dim]
        return annotations, s0


class BahdanauAttention(nn.Module):
    """Additive alignment model: e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)."""

    def __init__(self, dec_dim: int, ann_dim: int, attn_dim: int):
        super().__init__()
        self.W_a = nn.Linear(dec_dim, attn_dim, bias=False)
        self.U_a = nn.Linear(ann_dim, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def precompute_keys(self, annotations: torch.Tensor) -> torch.Tensor:
        return self.U_a(annotations)  # [B, T_x, attn_dim]

    def forward(
        self,
        s_prev: torch.Tensor,
        annotations: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        keys: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keys is None:
            keys = self.precompute_keys(annotations)

        query = self.W_a(s_prev).unsqueeze(1)  # [B, 1, attn_dim]
        energies = self.v_a(torch.tanh(query + keys)).squeeze(-1)  # [B, T_x]

        if src_mask is not None:
            energies = energies.masked_fill(~src_mask, -1e9)

        alpha = F.softmax(energies, dim=-1)  # [B, T_x]
        context = torch.bmm(alpha.unsqueeze(1), annotations).squeeze(1)  # [B, ann_dim]
        return alpha, context


class BahdanauDecoder(nn.Module):
    """Attentional GRU decoder conditioning each step on [y_{i-1}; c_i]."""

    def __init__(self, vocab_size: int, emb_dim: int, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        ann_dim = 2 * enc_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.attention = BahdanauAttention(dec_dim, ann_dim, attn_dim)
        self.cell = GRUCellFromScratch(emb_dim + ann_dim, dec_dim)
        self.readout = nn.Linear(dec_dim + emb_dim + ann_dim, vocab_size)

    def step(
        self,
        y_prev: torch.Tensor,
        s_prev: torch.Tensor,
        annotations: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        keys: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_emb = self.embedding(y_prev)  # [B, E]
        alpha, context = self.attention(s_prev, annotations, src_mask, keys)
        gru_input = torch.cat([y_emb, context], dim=-1)  # [B, E + 2 * enc_dim]
        s_i = self.cell(gru_input, s_prev)
        logits = self.readout(torch.cat([s_i, y_emb, context], dim=-1))
        return logits, s_i, alpha


class BahdanauSeq2Seq(nn.Module):
    """Full encoder-decoder model with Bahdanau additive attention."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 32,
        enc_dim: int = 48,
        dec_dim: int = 48,
        attn_dim: int = 48,
    ):
        super().__init__()
        self.encoder = BidirectionalGRUEncoder(vocab_size, emb_dim, enc_dim, dec_dim)
        self.decoder = BahdanauDecoder(vocab_size, emb_dim, enc_dim, dec_dim, attn_dim)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        retain_annotations: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        annotations, s = self.encoder(src, src_mask)
        if retain_annotations and annotations.requires_grad:
            annotations.retain_grad()
        keys = self.decoder.attention.precompute_keys(annotations)

        logits_list, alpha_list = [], []
        for i in range(tgt_in.shape[1]):
            logits, s, alpha = self.decoder.step(tgt_in[:, i], s, annotations, src_mask, keys)
            logits_list.append(logits)
            alpha_list.append(alpha)

        return torch.stack(logits_list, dim=1), torch.stack(alpha_list, dim=1), annotations

    @torch.no_grad()
    def decode_greedy(
        self, src: torch.Tensor, src_mask: torch.Tensor | None, max_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        annotations, s = self.encoder(src, src_mask)
        keys = self.decoder.attention.precompute_keys(annotations)

        y_prev = torch.full((src.shape[0],), SOS_IDX, dtype=torch.long, device=src.device)
        preds, alphas = [], []
        for _ in range(max_len):
            logits, s, alpha = self.decoder.step(y_prev, s, annotations, src_mask, keys)
            y_prev = logits.argmax(dim=-1)
            preds.append(y_prev)
            alphas.append(alpha)
        return torch.stack(preds, dim=1), torch.stack(alphas, dim=1)


class FixedBottleneckSeq2Seq(nn.Module):
    """Fixed-vector GRU encoder-decoder baseline without attention."""

    def __init__(self, vocab_size: int, emb_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.src_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.enc_cell = GRUCellFromScratch(emb_dim, hidden_dim)
        self.dec_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.dec_cell = GRUCellFromScratch(emb_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim + emb_dim, vocab_size)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        B, T_x = src.shape
        x = self.src_embedding(src)
        if src_mask is None:
            src_mask = torch.ones(B, T_x, dtype=torch.bool, device=src.device)
        h = torch.zeros(B, self.hidden_dim, device=src.device, dtype=x.dtype)
        for j in range(T_x):
            m_j = src_mask[:, j : j + 1].to(x.dtype)
            h = m_j * self.enc_cell(x[:, j], h) + (1.0 - m_j) * h
        return h

    def forward(
        self, src: torch.Tensor, tgt_in: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        s = self.encode(src, src_mask)
        logits_list = []
        for i in range(tgt_in.shape[1]):
            y_emb = self.dec_embedding(tgt_in[:, i])
            s = self.dec_cell(y_emb, s)
            logits_list.append(self.readout(torch.cat([s, y_emb], dim=-1)))
        return torch.stack(logits_list, dim=1)

    @torch.no_grad()
    def decode_greedy(
        self, src: torch.Tensor, src_mask: torch.Tensor | None, max_len: int
    ) -> torch.Tensor:
        s = self.encode(src, src_mask)
        y_prev = torch.full((src.shape[0],), SOS_IDX, dtype=torch.long, device=src.device)
        preds = []
        for _ in range(max_len):
            y_emb = self.dec_embedding(y_prev)
            s = self.dec_cell(y_emb, s)
            y_prev = self.readout(torch.cat([s, y_emb], dim=-1)).argmax(dim=-1)
            preds.append(y_prev)
        return torch.stack(preds, dim=1)


def sample_reversal_batch(
    batch_size: int, min_len: int, max_len: int, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate variable-length token reversal batches with <SOS>, <EOS>, and <PAD>."""
    lengths = torch.randint(min_len, max_len + 1, (batch_size,))
    max_t = int(lengths.max().item()) + 1  # +1 for <EOS> / <SOS>
    src = torch.full((batch_size, max_t), PAD_IDX, dtype=torch.long)
    tgt_in = torch.full((batch_size, max_t), PAD_IDX, dtype=torch.long)
    tgt_out = torch.full((batch_size, max_t), PAD_IDX, dtype=torch.long)

    for b in range(batch_size):
        L = int(lengths[b].item())
        seq = torch.randint(CONTENT_START, vocab_size, (L,))
        rev = seq.flip(0)
        src[b, :L] = seq
        src[b, L] = EOS_IDX
        tgt_in[b, 0] = SOS_IDX
        tgt_in[b, 1 : L + 1] = rev
        tgt_out[b, :L] = rev
        tgt_out[b, L] = EOS_IDX

    src_mask = src != PAD_IDX
    return src, tgt_in, tgt_out, src_mask


def verify_shapes_and_conservation():
    """Verify tensor shapes, simplex conservation, padding masking, and annotation gradients."""
    torch.manual_seed(0)
    B, T_x, T_y, V = 4, 12, 10, 15
    model = BahdanauSeq2Seq(vocab_size=V, emb_dim=16, enc_dim=24, dec_dim=24, attn_dim=20)

    src = torch.randint(CONTENT_START, V, (B, T_x))
    src[:, -3:] = PAD_IDX
    src_mask = src != PAD_IDX
    tgt_in = torch.randint(CONTENT_START, V, (B, T_y))
    tgt_in[:, 0] = SOS_IDX

    logits, attn_weights, annotations = model(src, tgt_in, src_mask, retain_annotations=True)

    assert logits.shape == (B, T_y, V), f"Unexpected logits shape: {logits.shape}"
    assert attn_weights.shape == (B, T_y, T_x), f"Unexpected attn shape: {attn_weights.shape}"
    assert annotations.shape == (B, T_x, 48), f"Unexpected annotations shape: {annotations.shape}"

    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(B, T_y), atol=1e-6), "Weights must sum to 1"

    pad_mass = attn_weights[:, :, -3:].abs().max().item()
    assert pad_mass < 1e-6, f"Padding positions received non-zero attention: {pad_mass}"

    # Verify direct gradient flow from a single decoder step i=0 back to encoder annotations h_j
    step_loss = logits[:, 0, :].sum()
    step_loss.backward()
    assert annotations.grad is not None
    grad_norms = annotations.grad.norm(dim=-1).mean(dim=0)  # [T_x]
    valid_grad_norm = grad_norms[:-3].mean().item()
    assert valid_grad_norm > 0.0, "Expected non-zero gradients at encoder annotations"

    print(
        f"Shape & conservation checks passed | logits={tuple(logits.shape)} | "
        f"attn={tuple(attn_weights.shape)} | max_sum_err={(row_sums - 1.0).abs().max().item():.2e} | "
        f"pad_attn_max={pad_mass:.2e} | mean ||dL_0/dh_j||={valid_grad_norm:.4e}"
    )


def evaluate_greedy(
    model: nn.Module, vocab_size: int, seq_len: int, n_samples: int = 256
) -> tuple[float, float]:
    """Evaluate autoregressive greedy decoding token and exact-sequence accuracy."""
    model.eval()
    src, _, tgt_out, src_mask = sample_reversal_batch(n_samples, seq_len, seq_len, vocab_size)
    out = model.decode_greedy(src, src_mask, max_len=tgt_out.shape[1])
    preds = out[0] if isinstance(out, tuple) else out
    valid = tgt_out != PAD_IDX
    tok_acc = ((preds == tgt_out) & valid).sum().item() / valid.sum().item()
    seq_acc = (((preds == tgt_out) | ~valid).all(dim=1)).float().mean().item()
    model.train()
    return tok_acc, seq_acc


def run_bottleneck_benchmark(vocab_size: int = 15, steps: int = 320, batch_size: int = 64):
    """Compare Fixed-Bottleneck Seq2Seq vs Bahdanau Attention on long sequences (T=20+)."""
    torch.manual_seed(42)
    bahdanau = BahdanauSeq2Seq(vocab_size=vocab_size, emb_dim=32, enc_dim=48, dec_dim=48, attn_dim=48)
    fixed = FixedBottleneckSeq2Seq(vocab_size=vocab_size, emb_dim=32, hidden_dim=64)

    opt_b = torch.optim.Adam(bahdanau.parameters(), lr=5e-3)
    opt_f = torch.optim.Adam(fixed.parameters(), lr=5e-3)

    print(f"\nTraining Fixed-Bottleneck (H=64) vs Bahdanau Attention (H_enc=48, H_dec=48) for {steps} steps:")
    for step in range(1, steps + 1):
        src, tgt_in, tgt_out, src_mask = sample_reversal_batch(batch_size, 6, 24, vocab_size)

        opt_b.zero_grad()
        logits_b, _, _ = bahdanau(src, tgt_in, src_mask)
        loss_b = F.cross_entropy(logits_b.view(-1, vocab_size), tgt_out.view(-1), ignore_index=PAD_IDX)
        loss_b.backward()
        nn.utils.clip_grad_norm_(bahdanau.parameters(), 1.0)
        opt_b.step()

        opt_f.zero_grad()
        logits_f = fixed(src, tgt_in, src_mask)
        loss_f = F.cross_entropy(logits_f.view(-1, vocab_size), tgt_out.view(-1), ignore_index=PAD_IDX)
        loss_f.backward()
        nn.utils.clip_grad_norm_(fixed.parameters(), 1.0)
        opt_f.step()

        if step % 80 == 0:
            print(f"  step {step:3d} | loss_fixed={loss_f.item():.4f} | loss_bahdanau={loss_b.item():.4f}")

    print("\nGreedy decoding accuracy across sequence lengths (trained on T in [6, 24]):")
    for T in [10, 16, 20, 24, 26]:
        tok_f, seq_f = evaluate_greedy(fixed, vocab_size, T)
        tok_b, seq_b = evaluate_greedy(bahdanau, vocab_size, T)
        print(
            f"  T={T:2d} | Fixed tok={tok_f:.3f}, seq={seq_f:.3f} | "
            f"Bahdanau tok={tok_b:.3f}, seq={seq_b:.3f}"
        )

    tok_f22, _ = evaluate_greedy(fixed, vocab_size, 22)
    tok_b22, seq_b22 = evaluate_greedy(bahdanau, vocab_size, 22)
    assert tok_b22 > 0.90 and tok_b22 > tok_f22 + 0.30
    assert seq_b22 > 0.70


if __name__ == "__main__":
    verify_shapes_and_conservation()
    run_bottleneck_benchmark()

