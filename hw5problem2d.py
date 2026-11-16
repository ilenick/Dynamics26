
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Rössler system parameters
alpha = 0.2
b = 0.2
c = 5.7

# Define the system of equations
def rossler(t, u):
    x, y, z = u
    dx = -y - z
    dy = x + alpha * y
    dz = b + z * (x - c)
    return [dx, dy, dz]

# Poincare section: y = 0, with x < 0
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

# Remove transient intersections and keep x < 0
valid = (crossing_times > 100) & (crossings[:, 0] < 0)
crossings = crossings[valid]

# Use the first N consecutive intersections
N = 200
if len(crossings) < N:
    raise ValueError("Not enough Poincare intersections. Increase the integration time.")

points = crossings[:N]

x_n = points[:, 0]
z_n = points[:, 2]

# Fit degree-5 polynomials
degree = 5

# Q(x): z_n approximately Q(x_n)
Q_coeff = np.polyfit(x_n, z_n, degree)
Q = np.poly1d(Q_coeff)

# P(x): x_(n+1) approximately P(x_n)
P_coeff = np.polyfit(x_n[:-1], x_n[1:], degree)
P = np.poly1d(P_coeff)

# Values for plotting the polynomial approximations
x_plot = np.linspace(min(x_n), max(x_n), 500)

# Plot Q(x)
plt.figure()
plt.scatter(x_n, z_n, s=12, label="Poincare data")
plt.plot(x_plot, Q(x_plot), 'r', label="Q(x) polynomial fit")
plt.xlabel(r"$x_n$")
plt.ylabel(r"$z_n$")
plt.title("Polynomial Approximation of Q(x)")
plt.legend()
plt.grid()
plt.show()

# Plot P(x)
plt.figure()
plt.scatter(x_n[:-1], x_n[1:], s=12, label="Return map data")
plt.plot(x_plot, P(x_plot), 'r', label="P(x) polynomial fit")
plt.xlabel(r"$x_n$")
plt.ylabel(r"$x_{n+1}$")
plt.title("Polynomial Approximation of P(x)")
plt.legend()
plt.grid()
plt.show()

# Print polynomial coefficients
print("Q(x) coefficients (highest power first):")
print(Q_coeff)

print("\nP(x) coefficients (highest power first):")
print(P_coeff)

# Print the polynomial expressions
print("\nQ(x) =")
print(Q)

print("\nP(x) =")
print(P)
