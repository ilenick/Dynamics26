
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

# Integrate the system and collect intersections
sol = solve_ivp(
    rossler,
    (0, 5000),
    [1, 1, 1],
    events=section,
    rtol=1e-10,
    atol=1e-12,
    max_step=0.1
)

crossing_times = sol.t_events[0]
crossings = sol.y_events[0]

valid = (crossing_times > 100) & (crossings[:, 0] < 0)
crossings = crossings[valid]

N = 200
points = crossings[:N]

x_n = points[:, 0]
z_n = points[:, 2]

# Fit degree-5 polynomials
degree = 5

Q = np.poly1d(np.polyfit(x_n, z_n, degree))
P = np.poly1d(np.polyfit(x_n[:-1], x_n[1:], degree))

# Define iterations of the Poincare map
def iterate_map(x, k):
    for i in range(k):
        x = P(x)
    return x

# Find roots of P^k(x) - x
def find_roots(k):
    grid = np.linspace(min(x_n), max(x_n), 20000)
    values = np.array([iterate_map(x, k) - x for x in grid])

    roots = []

    for i in range(len(grid) - 1):
        if np.isfinite(values[i]) and np.isfinite(values[i + 1]):
            if values[i] * values[i + 1] < 0:
                root = brentq(
                    lambda x: iterate_map(x, k) - x,
                    grid[i],
                    grid[i + 1]
                )

                if not roots or abs(root - roots[-1]) > 1e-5:
                    roots.append(root)

    return roots

# Find distinct cycles, avoiding cyclic duplicates
cycles = []

for k in [1, 2, 3]:
    roots = find_roots(k)

    for root in roots:
        orbit = [root]
        x = root

        for j in range(k - 1):
            x = P(x)
            orbit.append(x)

        # Check whether this is actually a lower-period orbit
        lower_period = False
        for j in range(1, k):
            if abs(iterate_map(root, j) - root) < 1e-4:
                lower_period = True
                break

        if lower_period:
            continue

        # Check whether this cycle has already been found
        duplicate = False

        for old_k, old_orbit in cycles:
            if old_k == k:
                for shift in range(k):
                    shifted = old_orbit[shift:] + old_orbit[:shift]
                    if np.allclose(orbit, shifted, atol=1e-3):
                        duplicate = True
                        break

            if duplicate:
                break

        if not duplicate:
            cycles.append((k, orbit))

# Plot each periodic orbit in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for k, orbit in cycles:
    x0 = orbit[0]
    z0 = Q(x0)
    initial_condition = [x0, 0, z0]

    # Integrate the original system from the approximate periodic point
    trajectory = solve_ivp(
        rossler,
        (0, 500),
        initial_condition,
        events=section,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.02
    )

    # Find the time of the kth return, ignoring the initial intersection
    return_times = trajectory.t_events[0]
    return_times = return_times[return_times > 1e-6]

    if len(return_times) >= k:
        end_time = return_times[k - 1]
    else:
        end_time = trajectory.t[-1]
        print("Warning: Did not obtain all returns for period", k)

    # Keep only the trajectory through the kth return
    mask = trajectory.t <= end_time

    ax.plot(
        trajectory.y[0, mask],
        trajectory.y[1, mask],
        trajectory.y[2, mask],
        label="Period " + str(k)
    )

    ax.scatter(x0, 0, z0, s=30)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Approximate Periodic Orbits of the Rossler System")
ax.legend()
plt.show()
