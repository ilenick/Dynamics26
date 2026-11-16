
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
mu = 0.45
rho = 10

# Transform y = exp(v) to avoid numerical underflow
def system(t, z):
    x, v = z
    y = np.exp(v)

    dxdt = np.sqrt(x) * (x*(1 - mu*x) - y)
    dvdt = rho * (x - 1)

    return [dxdt, dvdt]

# Initial conditions near the unstable focus
x0 = 1.001
y0 = 1 - mu

z0 = [x0, np.log(y0)]

# Integrate
t_span = (0, 250)
t_eval = np.linspace(0, 250, 20000)

sol = solve_ivp(
    system,
    t_span,
    z0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

x = sol.y[0]
y = np.exp(sol.y[1])
t = sol.t

# Plot x(t)
plt.figure(figsize=(11, 5))
plt.plot(t, x)
plt.xlabel('Time')
plt.ylabel('x(t)')
plt.title('Sawtooth Oscillations')
plt.grid()
plt.show()

# Plot y(t)
plt.figure(figsize=(11, 5))
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('Evolution of y(t)')
plt.grid()
plt.show()

# Phase portrait
plt.figure(figsize=(7, 6))
plt.plot(x, y)
plt.plot(1, 1-mu, 'ro', label='Unstable focus')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase Portrait')
plt.legend()
plt.grid()
plt.show()
