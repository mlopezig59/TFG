from scipy.integrate import RK45
import numpy as np
import matplotlib.pyplot as plt

# Definimos la ecuación diferencial
def f(t, y):
    return -2 * y

# Condiciones iniciales
y0 = 1

# Intervalo de integración
t0, tf = 0, 10

# Puntos donde queremos la solución
t_eval = np.linspace(t0, tf, 100)
y_values = []
t_values = []

# Crear el integrador RK45
integrator = RK45(f, t0, [y0], tf)

# Integrar paso a paso
while integrator.status == 'running':
    integrator.step()
    t_values.append(integrator.t)
    y_values.append(integrator.y[0])

# Graficar la solución
plt.plot(t_values, y_values)
plt.plot(t_eval, np.exp(-2 * t_eval), 'r--')
plt.legend(['RK45', 'Solución exacta'])
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solución de la ODE usando RK45')
plt.show()