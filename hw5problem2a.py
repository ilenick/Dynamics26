
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
alpha = 0.2
b = 0.2
c = 5.7

# Rössler system
def rossler(t, u):
    x, y, z = u

    dxdt = -y - z
    dydt = x + alpha*y
    dzdt = b + z*(x - c)

    return [dxdt, dydt, dzdt]

# Initial conditions
u0 = [1, 1, 1]

# Time interval
t_span = (0, 300)
t_eval = np.linspace(0, 300, 30001)

# Solve the system
sol = solve_ivp(
    rossler,
    t_span,
    u0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-9,
    atol=1e-11
)

# Extract solution
t = sol.t
x = sol.y[0]
y = sol.y[1]
z = sol.y[2]

# Remove transient (first 100 time units)
transient = 100
idx = t >= transient

x = x[idx]
y = y[idx]
z = z[idx]
t = t[idx]

# Plot 3D Rössler attractor
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, linewidth=0.4)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Rössler Chaotic Attractor')

plt.tight_layout()
plt.show()

# Plot x-y projection
plt.figure(figsize=(8, 6))
plt.plot(x, y, linewidth=0.4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rössler Attractor: x-y Projection')
plt.grid()
plt.show()
