
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
alpha = 0.2
b = 0.2
c = 5.7

# Rossler system
def rossler(t, u):
    x, y, z = u

    dxdt = -y - z
    dydt = x + alpha*y
    dzdt = b + z*(x - c)

    return [dxdt, dydt, dzdt]

# Poincare section: y = 0
def section(t, u):
    return u[1]

section.direction = -1
section.terminal = False

# Initial condition
u0 = [1, 1, 1]

# Integrate
sol = solve_ivp(
    rossler,
    (0, 5000),
    u0,
    events=section,
    method='RK45',
    rtol=1e-10,
    atol=1e-12,
    max_step=0.1
)

# Extract intersection times and coordinates
t_cross = sol.t_events[0]
crossings = sol.y_events[0]

# Remove transient and retain x < 0
transient = 100

idx = (t_cross > transient) & (crossings[:, 0] < 0)

t_cross = t_cross[idx]
crossings = crossings[idx]

# Collect at least 200 consecutive intersections
N = 200

if len(t_cross) < N:
    raise ValueError("Not enough intersections. Increase integration time.")

points = crossings[:N]
x_n = points[:, 0]
z_n = points[:, 2]

# Keep the first N consecutive intersections after the transient
N = 200
points = crossings[:N]

x_n = points[:, 0]
z_n = points[:, 2]

# Figure 1: Poincare section (x_n, z_n)
plt.figure()
plt.scatter(x_n, z_n, s=10)
plt.xlabel(r"$x_n$")
plt.ylabel(r"$z_n$")
plt.title("Poincare Section")
plt.grid()
plt.show()

# Figure 2: Return map (x_n, x_{n+1})
plt.figure()
plt.scatter(x_n[:-1], x_n[1:], s=10)
plt.xlabel(r"$x_n$")
plt.ylabel(r"$x_{n+1}$")
plt.title("Return Map")
plt.grid()
plt.show()

print("Number of intersections:", len(points))
print("First 10 intersection points:")
print(np.column_stack((x_n[:10], z_n[:10])))

# Plot Poincare section
plt.figure(figsize=(8, 6))
plt.scatter(x_n, z_n, s=12)
plt.xlabel('$x_n$')
plt.ylabel('$z_n$')
plt.title('Poincare Section: y = 0, x < 0')
plt.grid()
plt.show()

# Plot intersection coordinates versus iteration number
plt.figure(figsize=(10, 5))
plt.plot(range(1, N+1), x_n, label='$x_n$')
plt.plot(range(1, N+1), z_n, label='$z_n$')
plt.xlabel('Intersection number n')
plt.ylabel('Coordinate value')
plt.title('Consecutive Poincare Intersections')
plt.legend()
plt.grid()
plt.show()
