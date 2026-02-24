# Double Descent in Overparameterized MLPs
A teacher–student regression study of interpolation and modern generalization behavior.
1. Overview

Short paragraph:

We simulate data from a fixed nonlinear teacher MLP.

We train student networks of increasing width.

We investigate the double descent phenomenon.

We analyze the effects of noise, regularization, and optimizer choice.

2. Main Result

Leave placeholder for figure:

## Double Descent Curve

![double_descent](reports/figures/main_double_descent.png)

This image will be inserted later.

3. Experimental Setup

Bullet points:

Input dimension: 20

Teacher network: 2-layer MLP (width 32)

Student networks: width sweep 4 → 1024

Training samples: 1000

Test samples: 10000

Loss: MSE

Optimizer: Adam

Deterministic seed control

4. Key Findings

Leave placeholder — we will fill after results.

5. Reproducibility
pip install -r requirements.txt

Run notebooks in order:

01_teacher_student_double_descent.ipynb

02_regularization_shift.ipynb

03_optimizer_effects.ipynb


---

# 🧪 6️⃣ Deterministic Reproducibility Setup

Inside every notebook, first cell should be:

```python
import torch
import numpy as np
import random

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

This ensures reproducibility.
