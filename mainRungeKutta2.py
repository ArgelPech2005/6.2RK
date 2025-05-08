#   Codigo que implementa el metodo de Runge-Kutta
#   de cuarto orden para resolver una ecuacion diferencial
#   
#           Autor:
#   Argel Jesus Pech Manrique
#   argelpech098@gmail.com  
#   Version 1.01 : 06/05/2025
#

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del circuito
V = 10         # Voltaje en voltios
R = 1000       # Resistencia en ohmios
C = 0.001      # Capacitancia en faradios

# EDO: dq/dt = (V - q/C) / R
def f(t, q):
    return (V - q / C) / R

# Método de Runge-Kutta 4to orden
def runge_kutta_4(f, t0, q0, t_end, h):
    t_vals = [t0]
    q_vals = [q0]

    t = t0
    q = q0

    while t < t_end:
        k1 = f(t, q)
        k2 = f(t + h/2, q + h/2 * k1)
        k3 = f(t + h/2, q + h/2 * k2)
        k4 = f(t + h, q + h * k3)

        q += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h

        t_vals.append(round(t, 5))
        q_vals.append(q)

    return t_vals, q_vals

# Condiciones iniciales
t0 = 0
q0 = 0
t_end = 1
h = 0.05

# Ejecutar Runge-Kutta
t_vals, q_vals = runge_kutta_4(f, t0, q0, t_end, h)

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(t_vals, q_vals, 'o-', label='Carga q(t)', color='green')
plt.title('Carga de un Capacitor en un Circuito RC')
plt.xlabel('Tiempo (s)')
plt.ylabel('Carga (Coulombs)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("carga_capacitor_rc.png")
plt.show()
