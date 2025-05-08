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
import pandas as pd

# Definición de la EDO: dT/dx = -0.25(T - 25)
def f(x, T):
    return -0.25 * (T - 25)

# Método de Runge-Kutta de cuarto orden
def runge_kutta_4(f, x0, y0, x_end, h):
    x_vals = [x0]
    y_vals = [y0]

    x = x0
    y = y0

    while x < x_end:
        k1 = f(x, y)
        k2 = f(x + h/2, y + h/2 * k1)
        k3 = f(x + h/2, y + h/2 * k2)
        k4 = f(x + h, y + h * k3)

        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h

        x_vals.append(round(x, 5))
        y_vals.append(y)

    return x_vals, y_vals

# Parámetros iniciales
x0 = 0
T0 = 100
x_end = 2
h = 0.1

# Solución con Runge-Kutta
x_vals, T_rk = runge_kutta_4(f, x0, T0, x_end, h)

# Solución exacta
T_exacta = [25 + 75 * np.exp(-0.25 * x) for x in x_vals]

# Guardar en CSV
df = pd.DataFrame({
    "x (m)": x_vals,
    "T_RK4 (°C)": T_rk,
    "T_Exacta (°C)": T_exacta
})
df.to_csv("transferencia_calor_rk4.csv", index=False)

# Graficar
plt.figure(figsize=(9, 5))
plt.plot(x_vals, T_rk, 'o-', label="RK4 (aproximada)", color="blue")
plt.plot(x_vals, T_exacta, '--', label="Solución exacta", color="red")
plt.xlabel("Distancia x (m)")
plt.ylabel("Temperatura T (°C)")
plt.title("Transferencia de Calor en un Tubo - RK4 vs Exacta")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("perfil_temperatura_rk4.png")
plt.show()


