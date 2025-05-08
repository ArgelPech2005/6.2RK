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

# Definición del sistema de EDOs:
# dy1/dt = y2
# dy2/dt = -2*y2 - 5*y1
def sistema(t, y):
    y1, y2 = y
    dy1 = y2
    dy2 = -2 * y2 - 5 * y1
    return np.array([dy1, dy2])

# Método de Runge-Kutta 4º orden para sistemas
def runge_kutta_sistema(f, t0, y0, t_end, h):
    t_vals = [t0]
    y1_vals = [y0[0]]
    y2_vals = [y0[1]]
    
    t = t0
    y = np.array(y0, dtype=float)


    while t < t_end:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        
        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h
        
        t_vals.append(round(t, 5))
        y1_vals.append(y[0])
        y2_vals.append(y[1])

    return t_vals, y1_vals, y2_vals

# Condiciones iniciales
t0 = 0
y0 = [1, 0]     # y1(0) = 1 (posición), y2(0) = 0 (velocidad)
t_end = 5
h = 0.1

# Resolver el sistema
t_vals, y1_vals, y2_vals = runge_kutta_sistema(sistema, t0, y0, t_end, h)

# Gráfica de la trayectoria de la masa
plt.figure(figsize=(9, 5))
plt.plot(t_vals, y1_vals, 'b-', label='Posición y₁(t)')
plt.title('Dinámica de un Resorte Amortiguado')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición (m)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("trayectoria_resorte.png")
plt.show()
