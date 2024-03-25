#ORIGINAL CODE FROM https://colab.research.google.com/github/jckantor/CBE30338/blob/master/docs/03.09-COVID-19.ipynb#scrollTo=OztoD-uH3ffm

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# parameter values
R0 = 8.9
t_incubation = 2
t_infective = 20

# initial number of infected and recovered individuals
e_initial = 8
i_initial = 1
r_initial = 0.00
s_initial = 1071 - e_initial - i_initial - r_initial
N = 1080. #Total number of individuals, N

c = 1/t_incubation
b = 1/t_infective
a = R0*b

# SEIR model differential equations.
def deriv(x, t, c, a, b):
    s, e, i, r = x
    dsdt = -a * s * i / N
    dedt =  a * s * i / N - c * e
    didt = c * e - b * i
    drdt =  b * i
    return [dsdt, dedt, didt, drdt]

t = np.linspace(0, 300, 300)
x_initial = s_initial, e_initial, i_initial, r_initial
soln = odeint(deriv, x_initial, t, args=(c, a, b))
S, E, I, R = soln.T




plt.figure()
plt.grid()
plt.title("odeint method")
plt.plot(t, S, 'orange', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Removed')
plt.plot(t, E, 'b', label='Exposed')
plt.xlabel('Time t, [days]')
plt.ylabel('Numbers of individuals')
plt.ylim([0,N])
plt.legend()

plt.show();
