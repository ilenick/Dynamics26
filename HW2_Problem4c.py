import numpy as np
import matplotlib.pyplot as plt

N = 1001
x = np.linspace(-1, 1, N)
dx = x[1] - x[0]

K = 1 / (1 + (x[:, None] - x[None, :])**2)

w = np.ones(N)
w[0] = 0.5
w[-1] = 0.5

sine = np.sin(2 * np.pi * x)

def T(f, lam):
    integral = K @ (w * f) * dx

    return sine + lam * integral


# fixed point iteration

def fixed_point_iteration(f0, lam, tol=1e-10, max_iter=500):

    f = f0.copy()

    for n in range(max_iter):

        f_new = T(f, lam)

        # Sup norm ||f_new - f||_infinity
        error = np.max(np.abs(f_new - f))

        if error < tol:
            return f_new, n + 1, error, True

        f = f_new

    return f, max_iter, error, False

lambdas = [
    0.25, 
    0.49,  
    0.51,   
    0.75    
]

plt.figure(figsize=(8, 5))

for lam in lambdas:

    f = np.zeros(N)
    errors = []

    for n in range(50):

        f_new = T(f, lam)

        error = np.max(np.abs(f_new - f))
        errors.append(error)

        f = f_new

    plt.semilogy(
        range(1, 51),
        errors,
        label=f"lambda = {lam}"
    )

plt.xlabel("Iteration n")
plt.ylabel(r"$||f_{n+1}-f_n||_\infty$")
plt.title("Fixed-point convergence from f0 = 0")
plt.grid()
plt.legend()
plt.show()