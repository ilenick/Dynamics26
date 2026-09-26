import numpy as np
import matplotlib.pyplot as plt

alpha = 0.5
N = 25

x = np.linspace(-np.pi, np.pi, 50000)

# Compute truncated Weierstrass function
W = np.zeros_like(x)

for n in range(N + 1):
    W += 2**(-n * alpha) * np.cos(2**n * x)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, W)
plt.xlabel("x")
plt.ylabel(r"$W_N(x)$")
plt.title(rf"Approximation of Weierstrass ($\alpha={alpha}$, $N={N}$)")
plt.grid(True)
plt.show()