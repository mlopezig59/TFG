from scipy.integrate import RK45
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt


# QUIERO SABER CUANTO VALOR U=EXP(-i*H*t) PARA DISTINTOS VALORES DE T

# Parámetros conocidos
H = 1 * np.array([[0, 1], [1, 0]])
t0 = 0
tf = 10

# Definimos U=EXP(-iHt)
U0=expm(-1j * H * t0)

# Definimos la ecuación diferencial: i dU/dt = H*U
def f(t, U):
    dUdt = (-1j * H @ U.reshape((2,2)))
    return dUdt.flatten()

# Creamos el integrador RK45
integrator = RK45(f, t0, U0.flatten(), tf)

# Creamos un array ddonde vamos a guardar la solución
U_values = []
t_values = []
U_exact_values=[]

# Integrar paso a paso
while integrator.status == 'running':
    integrator.step()
    t_values.append(integrator.t)
    U_values.append(integrator.y.reshape((2,2)))
U_values = np.array(U_values)

# Resultado conocido
def U_exact(t):
    return np.cos(t)*np.eye(2) - 1j*np.sin(t)*H
for i in t_values:
    U_exact_values.append(U_exact(i))
U_exact_values = np.array(U_exact_values)

# Graficar la solución
fig, axs = plt.subplots(2, 2)

# Plot U[0,0]
axs[0, 0].plot(t_values, U_values[:, 0, 0].real, label='Re(U[0,0])')
axs[0, 0].plot(t_values, U_values[:, 0, 0].imag, label='Im(U[0,0])')
axs[0, 0].plot(t_values, U_exact_values[:, 0, 0].real, '--', label='Re(U_exact[0,0])')
axs[0, 0].plot(t_values, U_exact_values[:, 0, 0].imag, '--', label='Im(U_exact[0,0])')
axs[0, 0].set_title('U[0,0]')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot U[0,1]
axs[0, 1].plot(t_values, U_values[:, 0, 1].real, label='Re(U[0,1])')
axs[0, 1].plot(t_values, U_values[:, 0, 1].imag, label='Im(U[0,1])')
axs[0, 1].plot(t_values, U_exact_values[:, 0, 1].real, '--', label='Re(U_exact[0,1])')
axs[0, 1].plot(t_values, U_exact_values[:, 0, 1].imag, '--', label='Im(U_exact[0,1])')
axs[0, 1].set_title('U[0,1]')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot U[1,0]
axs[1, 0].plot(t_values, U_values[:, 1, 0].real, label='Re(U[1,0])')
axs[1, 0].plot(t_values, U_values[:, 1, 0].imag, label='Im(U[1,0])')
axs[1, 0].plot(t_values, U_exact_values[:, 1, 0].real, '--', label='Re(U_exact[1,0])')
axs[1, 0].plot(t_values, U_exact_values[:, 1, 0].imag, '--', label='Im(U_exact[1,0])')
axs[1, 0].set_title('U[1,0]')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot U[1,1]
axs[1, 1].plot(t_values, U_values[:, 1, 1].real, label='Re(U[1,1])')
axs[1, 1].plot(t_values, U_values[:, 1, 1].imag, label='Im(U[1,1])')
axs[1, 1].plot(t_values, U_exact_values[:, 1, 1].real, '--', label='Re(U_exact[1,1])')
axs[1, 1].plot(t_values, U_exact_values[:, 1, 1].imag, '--', label='Im(U_exact[1,1])')
axs[1, 1].set_title('U[1,1]')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.show()