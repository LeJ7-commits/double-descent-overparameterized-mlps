import json

notebook_path = "notebooks/02_regularization_shift.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

insert_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "Attempt 4A" in src and "WITH WEIGHT DECAY SWEEP" in src:
            insert_idx = i
            break

if insert_idx != -1:
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Regularization-Wise Double Descent (Random Features)\n",
            "Fix the number of features $m$ near the interpolation threshold ($m=n$) and carefully sweep over the regularization penalty (weight decay / ridge). This demonstrates how explicit regularization itself can induce a double descent peak."
        ]
    }
    
    code_content = """# --- Regularization-Wise Double Descent ---

# We fix the capacity at the interpolation threshold
m_fixed = n_train

# Sweep over Regularization (Weight Decay/Ridge) finely
ridges_reg = np.logspace(-8, 1, 50)

train_mse_reg = []
test_mse_reg  = []

# Generate same type of fixed random features for m=n_train
W_fixed = rng.standard_normal((m_fixed, d_in)) / np.sqrt(d_in)

Phi_tr_fixed = relu(Xtr @ W_fixed.T)
Phi_te_fixed = relu(Xte @ W_fixed.T)

# Precompute to avoid re-multiplying for every ridge
Phi_tr_T_Phi_tr_fixed = Phi_tr_fixed.T @ Phi_tr_fixed
Phi_tr_T_ytr_fixed = Phi_tr_fixed.T @ ytr

for ridge in ridges_reg:
    # Solve ridge regression in feature space
    A = Phi_tr_T_Phi_tr_fixed + ridge * np.eye(m_fixed)
    b = Phi_tr_T_ytr_fixed
    
    beta = np.linalg.solve(A, b)

    yhat_tr = Phi_tr_fixed @ beta
    yhat_te = Phi_te_fixed @ beta

    train_mse_reg.append(np.mean((yhat_tr - ytr) ** 2))
    test_mse_reg.append(np.mean((yhat_te - yte) ** 2))

# --- Plot Regularization-wise double descent ---
plt.figure(figsize=(8, 5))
plt.plot(ridges_reg, test_mse_reg, marker="o", markersize=4, color="purple", label=f"Test MSE")
plt.plot(ridges_reg, train_mse_reg, marker="^", markersize=4, linestyle="--", alpha=0.5, label=f"Train MSE")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Weight Decay / Ridge Penalty (log scale)')
plt.ylabel('MSE (log scale)')
plt.title(f'Regularization-wise Double Descent (Capacity Fixed at m={m_fixed})')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.show()"""

    code_lines = [line + "\n" for line in code_content.split("\n")]
    code_lines[-1] = code_lines[-1][:-1] # remove last newline

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code_lines
    }

    nb["cells"].insert(insert_idx + 1, md_cell)
    nb["cells"].insert(insert_idx + 2, code_cell)
    
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=1)
    
    print("Cells inserted successfully.")
else:
    print("Target cell not found.")
