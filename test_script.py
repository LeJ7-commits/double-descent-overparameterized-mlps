import numpy as np
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)

# --- DGP ---
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

ms = list(range(10, 401, 10)) + [500, 700, 1000, 1500, 2000, 3000]

ridges = [1e-8, 1e-4, 1e-2, 1e-1, 1.0]

train_mse_dict = {r: [] for r in ridges}
test_mse_dict  = {r: [] for r in ridges}

for m in ms:
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

        train_mse_dict[ridge].append(np.mean((yhat_tr - ytr) ** 2))
        test_mse_dict[ridge].append(np.mean((yhat_te - yte) ** 2))

print("Interpolation point found: ", ms[np.where(np.array(train_mse_dict[1e-8]) < 1e-8)[0][0]])
print("Test complete.")
