import numpy as np
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)

n_train = 256
n_test  = 4096
d_in    = 50

noise_levels = [0.0, 0.5, 2.0]
m_peak = n_train
ridges_noise_exp = np.logspace(-8, 2, 40)

Xte = rng.standard_normal((n_test,  d_in))
Xtr = rng.standard_normal((n_train, d_in))

W = rng.standard_normal((m_peak, d_in)) / np.sqrt(d_in)

def relu(z): return np.maximum(z, 0.0)

Phi_tr = relu(Xtr @ W.T)
Phi_te = relu(Xte @ W.T)

Phi_tr_T_Phi_tr = Phi_tr.T @ Phi_tr

w_star = rng.standard_normal(d_in) / np.sqrt(d_in)
ytr_clean = Xtr @ w_star
yte_clean = Xte @ w_star

plt.figure(figsize=(10, 6))
colors = plt.cm.plasma(np.linspace(0, 0.8, len(noise_levels)))

for i, n_std in enumerate(noise_levels):
    ytr = ytr_clean + n_std * rng.standard_normal(n_train)
    yte = yte_clean
    
    y_mean = ytr.mean()
    y_std  = ytr.std() + 1e-8
    ytr = (ytr - y_mean) / y_std
    yte = (yte - y_mean) / y_std
    
    Phi_tr_T_ytr = Phi_tr.T @ ytr
    
    test_mse_noise = []
    
    for ridge in ridges_noise_exp:
        A = Phi_tr_T_Phi_tr + ridge * np.eye(m_peak)
        b = Phi_tr_T_ytr
        
        beta = np.linalg.solve(A, b)

        yhat_te = Phi_te @ beta
        test_mse_noise.append(np.mean((yhat_te - yte) ** 2))
    
    min_idx = np.argmin(test_mse_noise)
    opt_ridge = ridges_noise_exp[min_idx]
    
    plt.plot(ridges_noise_exp, test_mse_noise, marker="o", markersize=4, 
             color=colors[i], label=f"Noise Std = {n_std} (Opt. WD ≈ {opt_ridge:.1e})")
    
    plt.plot(opt_ridge, test_mse_noise[min_idx], marker="*", markersize=15, 
             color=colors[i], markeredgecolor='black')


plt.xscale('log')
plt.yscale('log')
plt.xlabel('Weight Decay / Ridge Penalty (log scale)')
plt.ylabel('Test MSE (log scale)')
plt.title(f'Optimal Regularization Shifts with Noise Level (Width Fixed at m={m_peak})')
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("exp3_noise.png")
print("Done")
