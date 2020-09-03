import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0.0, 5e-3, 200)

# Parallel RC Model
R = 100
C = 10e-6
v_init = [10]
v = integrate.odeint(lambda v, t: -v/(R*C), v_init, t)

# with plt.style.context("dark_background"):
#     fig, ax = plt.subplots()
#     ax.plot(t, v)
#     ax.set_xlabel("time (s)")
#     ax.set_ylabel("voltage")
#     plt.show()

# Parallel RCL Model
R = 100
C = 10e-6
L = 12e-3
v_init = [10.0, 0.0]
v = integrate.odeint(
    lambda v, t: [v[1], -v[1]/(R * C) - v[0]/(C * L)],
    v_init, t)

with plt.style.context("dark_background"):
    fig, ax = plt.subplots()
    ax.plot(t, v[:, 0])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("voltage")
    plt.show()
