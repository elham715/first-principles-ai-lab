# 00. Core Coding Foundations (80/20 Mastery)

> **Purpose:** The essential 20% of Python, OOP, NumPy, Pandas, PyTorch, and TensorFlow 2 mechanics required to read, derive, and reproduce deep learning papers from scratch without confusion.

---

## 📚 Notebook Progression

| Notebook | Core Topics Covered | Verification |
|---|---|---|
| [`00a_python_and_oop_80_20.ipynb`](./00a_python_and_oop_80_20.ipynb) | Variables, lists/slicing, dicts/sets, loops (`enumerate`, `zip`), comprehensions, functions (`*args`/`**kwargs`), `class`/`__init__`/`self`, dunders (`__call__`, `__len__`, `__getitem__`), inheritance + `super()`, composition, `@dataclass`/`@property`/`@classmethod`/`@staticmethod`/`ABC` | ✅ Executed cleanly |
| [`00b_numpy_80_20.ipynb`](./00b_numpy_80_20.ipynb) | Arrays, `.shape`/`.dtype`, indexing & boolean masks, `*` vs `@`, `axis=0` vs `axis=1`, broadcasting traps, `.reshape`/`None`/`.transpose()`, Sigmoid/Softmax/Cross-Entropy, `concatenate` vs `stack` | ✅ Executed cleanly |
| [`00c_pandas_80_20.ipynb`](./00c_pandas_80_20.ipynb) | `DataFrame` vs `Series`, `.loc`/`.iloc`, boolean filtering, `isna`/`fillna`, `.map()`, `value_counts()`, `groupby().agg()`, `pd.merge()`, patient-safe splits & `.to_numpy()` | ✅ Executed cleanly |
| [`00d_pytorch_core_80_20.ipynb`](./00d_pytorch_core_80_20.ipynb) | Tensors (`float32` vs `long`), Autograd, `nn.Module`, `Linear`/`Conv2d`/`Embedding`/`LSTM`, `Dataset`/`DataLoader`, `CrossEntropyLoss`, 5-step training loop, `train()` vs `eval()`, `state_dict()` | ✅ Executed cleanly |
| [`00e_tensorflow_core_80_20.ipynb`](./00e_tensorflow_core_80_20.ipynb) | `tf.constant` vs `tf.Variable`, `tf.GradientTape`, Sequential / Functional / Subclassed `tf.keras.Model`, `tf.data.Dataset`, `compile`/`fit` vs custom loops, PyTorch ↔ TF cheat sheet | ✅ Syntax-checked |

---

## ➡️ Next Step
Once comfortable with these 5 notebooks, jump directly into [`../01_sequence_foundations/`](../01_sequence_foundations/README.md) starting with **Paper 00: Elman (1990)**.
