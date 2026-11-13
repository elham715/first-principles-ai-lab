"""Pure NumPy implementation of the Elman (1990) Simple Recurrent Network."""

import numpy as np


class ElmanRNNFromScratch:
    """Single-layer Elman RNN with analytical BPTT."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        np.random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W_xh = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.b_h = np.zeros((1, hidden_dim))

        self.W_hy = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros((1, output_dim))

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        shifted = z - np.max(z, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray, h_init: np.ndarray = None):
        """Run forward pass over X of shape (T, B, input_dim)."""
        T, B, _ = X.shape
        if h_init is None:
            h_init = np.zeros((B, self.hidden_dim))

        h_states = [h_init]
        y_probs = []

        for t in range(T):
            a_t = X[t] @ self.W_xh + h_states[-1] @ self.W_hh + self.b_h
            h_t = np.tanh(a_t)
            z_t = h_t @ self.W_hy + self.b_y
            p_t = self.softmax(z_t)

            h_states.append(h_t)
            y_probs.append(p_t)

        return y_probs, h_states

    def backward(self, X: np.ndarray, Y: np.ndarray, y_probs: list, h_states: list):
        """Compute parameter gradients via Backpropagation Through Time."""
        T, B, _ = X.shape

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((B, self.hidden_dim))
        grad_norms_over_time = []

        for t in reversed(range(T)):
            h_t = h_states[t + 1]
            h_prev = h_states[t]

            dz_t = y_probs[t] - Y[t]
            dW_hy += h_t.T @ dz_t
            db_y += np.sum(dz_t, axis=0, keepdims=True)

            dh_t = dz_t @ self.W_hy.T + dh_next
            grad_norms_over_time.append(np.linalg.norm(dh_t))

            da_t = dh_t * (1.0 - h_t ** 2)
            dW_xh += X[t].T @ da_t
            dW_hh += h_prev.T @ da_t
            db_h += np.sum(da_t, axis=0, keepdims=True)

            dh_next = da_t @ self.W_hh.T

        return {
            "dW_xh": dW_xh / B,
            "dW_hh": dW_hh / B,
            "db_h": db_h / B,
            "dW_hy": dW_hy / B,
            "db_y": db_y / B,
            "grad_norms_over_time": list(reversed(grad_norms_over_time)),
        }

    def compute_loss(self, y_probs: list, Y: np.ndarray, eps: float = 1e-12) -> float:
        T, B, _ = Y.shape
        total_loss = 0.0
        for t in range(T):
            p_t = np.clip(y_probs[t], eps, 1.0 - eps)
            total_loss -= np.sum(Y[t] * np.log(p_t))
        return total_loss / (T * B)


def check_gradients_numerical():
    """Verify analytical BPTT gradients against central finite differences."""
    T, B, Din, Dhidden, Dout = 4, 2, 3, 5, 4
    np.random.seed(0)
    X = np.random.randn(T, B, Din)

    targets_idx = np.random.randint(0, Dout, size=(T, B))
    Y = np.zeros((T, B, Dout))
    for t in range(T):
        for b in range(B):
            Y[t, b, targets_idx[t, b]] = 1.0

    model = ElmanRNNFromScratch(Din, Dhidden, Dout, seed=123)
    probs, states = model.forward(X)
    analytical_grads = model.backward(X, Y, probs, states)

    eps = 1e-5
    for param_name in ["W_hh", "W_xh", "W_hy"]:
        param = getattr(model, param_name)
        grad_analytical = analytical_grads[f"d{param_name}"]
        grad_numerical = np.zeros_like(param)

        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            ix = it.multi_index
            old_val = param[ix]

            param[ix] = old_val + eps
            probs_pos, _ = model.forward(X)
            loss_pos = model.compute_loss(probs_pos, Y)

            param[ix] = old_val - eps
            probs_neg, _ = model.forward(X)
            loss_neg = model.compute_loss(probs_neg, Y)

            param[ix] = old_val
            # Scale by T because compute_loss averages over T*B whereas backward averages over B.
            grad_numerical[ix] = (loss_pos - loss_neg) / (2.0 * eps) * T
            it.iternext()

        diff = np.linalg.norm(grad_analytical - grad_numerical)
        norm_sum = np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical)
        rel_error = diff / (norm_sum + 1e-12)

        print(f"{param_name:<6} rel_error = {rel_error:.2e} ({'PASS' if rel_error < 1e-5 else 'FAIL'})")
        assert rel_error < 1e-5, f"Gradient check failed for {param_name}"


def demonstrate_vanishing_gradients():
    """Measure ||dh_0|| vs ||dh_T|| as sequence length T increases."""
    hidden_dim, input_dim, output_dim = 16, 8, 8
    print("\nVanishing gradient check across sequence lengths:")

    for T in [5, 10, 15, 20, 30, 40]:
        model = ElmanRNNFromScratch(input_dim, hidden_dim, output_dim, seed=42)
        X = np.random.randn(T, 1, input_dim)
        Y = np.zeros((T, 1, output_dim))
        Y[:, 0, 0] = 1.0

        probs, states = model.forward(X)
        grads = model.backward(X, Y, probs, states)
        norms = grads["grad_norms_over_time"]

        ratio = norms[0] / (norms[-1] + 1e-12)
        print(f"T={T:2d} | ||dh_0||={norms[0]:.3e} | ||dh_T||={norms[-1]:.3e} | ratio={ratio:.4f}")


if __name__ == "__main__":
    check_gradients_numerical()
    demonstrate_vanishing_gradients()
