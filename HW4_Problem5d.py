
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

a, b, c = 0.2, 0.2, 5.7

def system(t, s):
    xD, yD, zD, xR, yR, zR = s

    ex = xR - xD
    ey = yR - yD
    ez = zR - zD

    # Driver
    dxD = -yD - zD
    dyD = xD + a*yD
    dzD = b + zD*(xD-c)

    # Control signals
    u1 = ey + ez - ex
    u2 = -ex - (a+1)*ey
    u3 = -zD*ex - (xD-c)*ez - ex*ez - ez

    # Response
    dxR = -yR - zR + u1
    dyR = xR + a*yR + u2
    dzR = b + zR*(xR-c) + u3

    return [dxD, dyD, dzD, dxR, dyR, dzR]

# Initial conditions: driver and response start differently
s0 = [1, 1, 1, -4, 3, 8]
t_eval = np.linspace(0, 50, 5001)

sol = solve_ivp(
    system, [0, 50], s0, t_eval=t_eval,
    rtol=1e-9, atol=1e-11
)

t = sol.t
xD, yD, zD, xR, yR, zR = sol.y

# Plot errors
plt.figure(figsize=(9, 5))
plt.plot(t, xR-xD, label='e_x')
plt.plot(t, yR-yD, label='e_y')
plt.plot(t, zR-zD, label='e_z')
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Synchronization Errors')
plt.legend()
plt.grid(True)
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Driver trajectory
ax.plot(xD, yD, zD, label='Driver', linewidth=1.2)

# Response trajectory
ax.plot(xR, yR, zR, '--', label='Response', linewidth=1.0)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.set_title('3D Synchronization of Rössler Systems')
ax.legend()

plt.tight_layout()
plt.show()