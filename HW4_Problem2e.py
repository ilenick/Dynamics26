
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
alpha = 1.0
beta = 0.3
eps0 = 2 * alpha * beta**1.5

eps_values = [1.2 * eps0, 0.8 * eps0]
labels = [
    r"$\epsilon = 1.2\epsilon_0$ (above threshold)",
    r"$\epsilon = 0.8\epsilon_0$ (below threshold)"
]

# Equilibrium
x_star = np.arccos(beta)
y_star = 0.0
z_star = 1 / np.sqrt(beta)
equilibrium = [x_star, y_star, z_star]

print(f"Critical epsilon = {eps0:.6f}")
print(f"Equilibrium = {equilibrium}")

# Define the system
def governor(t, state, epsilon):
    x, y, z = state

    dx = y
    dy = np.sin(x) * (z**2 * np.cos(x) - 1) - epsilon * y
    dz = alpha * (np.cos(x) - beta)

    return [dx, dy, dz]

# Start near the equilibrium
initial_condition = [
    x_star + 0.03,
    0.0,
    z_star + 0.02
]

solutions = []

for epsilon in eps_values:
    sol = solve_ivp(
        governor,
        [0, 600],
        initial_condition,
        args=(epsilon,),
        rtol=1e-9,
        atol=1e-11,
        max_step=0.05
    )
    solutions.append(sol)

# Plot x(t) to compare convergence and sustained oscillations
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

for ax, sol, label in zip(axes, solutions, labels):
    ax.plot(sol.t, sol.y[0], linewidth=1)
    ax.axhline(x_star, linestyle="--", color="black",
               label=r"$x^*$")
    ax.set_ylabel("x(t)")
    ax.set_title(label)
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel(r"Time $\tau$")
plt.tight_layout()
plt.show()

# Plot the phase portraits after the initial transient
fig = plt.figure(figsize=(12, 5))

for i, (sol, label) in enumerate(zip(solutions, labels), start=1):
    ax = fig.add_subplot(1, 2, i, projection="3d")

    # Show only the later trajectory
    mask = sol.t >= 200

    ax.plot(
        sol.y[0, mask],
        sol.y[1, mask],
        sol.y[2, mask],
        linewidth=0.8,
        label="Trajectory"
    )

    ax.scatter(
        *equilibrium,
        color="red",
        s=45,
        label="Equilibrium"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(label)
    ax.legend()

plt.tight_layout()
plt.show()
