import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Lorenz63 system
def lorenz63(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Initial condition
state0 = [1.0, 1.0, 1.0]

# Time span
t_span = (0, 50)
t_eval = np.linspace(0, 50, 10000)

# Integrate
sol = solve_ivp(lorenz63, t_span, state0, t_eval=t_eval)

# Plot trajectory in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(sol.y[0], sol.y[1], sol.y[2])
ax.set_title("Lorenz63 Attractor")
for y in sol.y:
    print(y, "\n\n\n")
plt.show()
