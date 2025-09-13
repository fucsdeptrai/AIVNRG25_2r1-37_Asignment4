import matplotlib.pyplot as plt
from src.train import train_model

def main():
    ns = [50, 100, 500, 1000, 2000]  # số lượng cặp key-value
    results = {"omega": {"mse": [], "cos": []}, "delta": {"mse": [], "cos": []}}

    for n in ns:
        # Omega
        mse_omega, cos_omega = train_model(updater_type="omega", mem_size=64, dim=64, steps=n, seed=0)
        results["omega"]["mse"].append(mse_omega)
        results["omega"]["cos"].append(cos_omega)

        # Delta
        mse_delta, cos_delta = train_model(updater_type="delta", mem_size=64, dim=64, steps=n, seed=0)
        results["delta"]["mse"].append(mse_delta)
        results["delta"]["cos"].append(cos_delta)

        print(f"[n={n}] Omega -> MSE: {mse_omega:.6f}, Cos: {cos_omega:.6f} | "
              f"Delta -> MSE: {mse_delta:.6f}, Cos: {cos_delta:.6f}")

    # ---- Plot MSE ----
    plt.figure(figsize=(8, 5))
    plt.plot(ns, results["omega"]["mse"], marker="o", label="Omega (MSE)")
    plt.plot(ns, results["delta"]["mse"], marker="s", label="Delta (MSE)")
    plt.xlabel("n (số lượng cặp key-value)")
    plt.ylabel("MSE")
    plt.title("So sánh MSE giữa Omega và Delta")
    plt.legend()
    plt.grid(True)
    plt.savefig("mse_comparison.png")
    plt.show()

    # ---- Plot 1 - Cosine Similarity ----
    plt.figure(figsize=(8, 5))
    omega_dist = [1 - c for c in results["omega"]["cos"]]
    delta_dist = [1 - c for c in results["delta"]["cos"]]
    plt.plot(ns, omega_dist, marker="o", label="Omega (1 - Cosine)")
    plt.plot(ns, delta_dist, marker="s", label="Delta (1 - Cosine)")
    plt.xlabel("n (số lượng cặp key-value)")
    plt.ylabel("1 - Cosine Similarity")
    plt.title("So sánh Cosine Distance giữa Omega và Delta")
    plt.legend()
    plt.grid(True)
    plt.savefig("cosine_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
