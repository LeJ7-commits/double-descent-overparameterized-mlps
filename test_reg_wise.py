import numpy as np
import matplotlib.pyplot as plt

SEED = 42 
rng = np.random.default_rng(SEED)

n_train = 256
n_test  = 4096
d_in    = 50
noise_std = 0.5

Xtr = rng.standard_normal((n_train, d_in))
Xte = rng.standard_normal((n_test,  d_in))

w_star = rng.standard_normal(d_in) / np.sqrt(d_in)
ytr_clean = Xtr @ w_star
yte_clean = Xte @ w_star

ytr = ytr_clean + noise_std * rng.standard_normal(n_train)
yte = yte_clean

y_mean = ytr.mean()
y_std  = ytr.std() + 1e-8
ytr = (ytr - y_mean) / y_std
yte = (yte - y_mean) / y_std

def relu(z): return np.maximum(z, 0.0)

# Fix m at interpolation threshold
m = n_train

# Sweep over Regularization (Weight Decay) finely
ridges = np.logspace(-8, 1, 50)

train_mse = []
test_mse  = []

W = rng.standard_normal((m, d_in)) / np.sqrt(d_in)

Phi_tr = relu(Xtr @ W.T)
Phi_te = relu(Xte @ W.T)

Phi_tr_T_Phi_tr = Phi_tr.T @ Phi_tr
Phi_tr_T_ytr = Phi_tr.T @ ytr

for ridge in ridges:
    A = Phi_tr_T_Phi_tr + ridge * np.eye(m)
    b = Phi_tr_T_ytr
    
    beta = np.linalg.solve(A, b)

    yhat_tr = Phi_tr @ beta
    yhat_te = Phi_te @ beta

    train_mse.append(np.mean((yhat_tr - ytr) ** 2))
    test_mse.append(np.mean((yhat_te - yte) ** 2))

plt.figure(figsize=(8, 5))
plt.plot(ridges, test_mse, marker="o", markersize=4, color="purple", label=f"Test MSE")
plt.plot(ridges, train_mse, marker="^", markersize=4, linestyle="--", alpha=0.5, label=f"Train MSE")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Weight Decay (Ridge Penalty)')
plt.ylabel('MSE (log scale)')
plt.title(f'Regularization-wise Double Descent (Width={m})')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig("reg_wise.png")
print("Done")
