
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
a = 0.2
b = 0.2
c = 5.7

# Driver system
def driver(t, state):
    xD, yD, zD = state

    dxD = -yD - zD
    dyD = xD + a*yD
    dzD = b + zD*(xD - c)

    return [dxD, dyD, dzD]

# Initial conditions
initial_conditions = [1, 1, 1]

# Time interval and evaluation points
t0 = 0
tf = 300
t_eval = np.linspace(t0, tf, 15001)

# Solve the ODE system
sol = solve_ivp(
    driver,
    [t0, tf],
    initial_conditions,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-11
)

# Discard transient: retain t >= 50
transient = 50
idx = sol.t >= transient

xD = sol.y[0, idx]
yD = sol.y[1, idx]
zD = sol.y[2, idx]

# Plot the attractor
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(xD, yD, zD, linewidth=0.5)

ax.set_xlabel('$x_D$')
ax.set_ylabel('$y_D$')
ax.set_zlabel('$z_D$')
ax.set_title("Rössler Attractor (transient discarded)")

plt.tight_layout()
plt.show()
