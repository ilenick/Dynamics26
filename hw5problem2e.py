
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Rössler system parameters
alpha = 0.2
b = 0.2
c = 5.7

# Define the Rössler system
def rossler(t, u):
    x, y, z = u
    dx = -y - z
    dy = x + alpha * y
    dz = b + z * (x - c)
    return [dx, dy, dz]

# Poincare section: y = 0, x < 0
def section(t, u):
    return u[1]

section.direction = -1
section.terminal = False

# Numerically solve the system
sol = solve_ivp(
    rossler,
    (0, 5000),
    [1, 1, 1],
    events=section,
    rtol=1e-10,
    atol=1e-12,
    max_step=0.1
)

# Collect intersections after the transient
crossing_times = sol.t_events[0]
crossings = sol.y_events[0]

valid = (crossing_times > 100) & (crossings[:, 0] < 0)
crossings = crossings[valid]

# Use the first N intersections
N = 200

if len(crossings) < N:
    raise ValueError("Not enough intersections. Increase integration time.")

points = crossings[:N]

x_n = points[:, 0]
z_n = points[:, 2]

# Fit degree-5 polynomials
degree = 5

Q = np.poly1d(np.polyfit(x_n, z_n, degree))
P = np.poly1d(np.polyfit(x_n[:-1], x_n[1:], degree))

# Define the k-th iteration of the Poincare map
def iterate_map(x, k):
    for i in range(k):
        x = P(x)
    return x

# Find roots of P^k(x) - x
def find_roots(k):
    xmin = min(x_n)
    xmax = max(x_n)

    grid = np.linspace(xmin, xmax, 20000)
    values = np.array([iterate_map(x, k) - x for x in grid])

    roots = []

    for i in range(len(grid) - 1):
        if values[i] * values[i + 1] < 0:
            root = brentq(
                lambda x: iterate_map(x, k) - x,
                grid[i],
                grid[i + 1]
            )

            # Avoid duplicate roots
            if not roots or abs(root - roots[-1]) > 1e-5:
                roots.append(root)

    return roots

# Find roots for periods 1, 2, and 3
for k in [1, 2, 3]:

    roots = find_roots(k)

    print("\n--------------------------------")
    print("Roots of P^" + str(k) + "(x) - x = 0")
    print("--------------------------------")

    for root in roots:

        print("\nRoot x =", root)

        # Print the orbit under the return map
        x = root

        for j in range(k):
            z = Q(x)

            print(
                "Initial condition on Sigma:",
                "(x, y, z) =",
                (x, 0, z)
            )

            x = P(x)

        print("Final iterate:", x)
