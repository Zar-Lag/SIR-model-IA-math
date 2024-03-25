import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numba
from scipy import integrate


N = 1080. #Total number of individuals, N
I0, R0 = 1., 0 #Initial number of infected and recovered individuals
S0 = N - I0 - R0 #Susceptible individuals to infection initially is deduced
a, b = 0.4, 0.05 #Contact rate and mean recovery rate
tmax = 300 #A grid of time points (in days)
Nt = 300
t = np.linspace(0, tmax, Nt+1)

def derivative(X, t):
    S, I, R = X
    dotS = -a*S*I/N # derivative of S(t)
    dotI = a*S*I/N - b*I # derivative of I(t)
    dotR = b*I # derivative of R(t)
    return np.array([dotS, dotI, dotR])
X0 = S0, I0, R0 #Initial conditions vector
res = integrate.odeint(derivative, X0, t)
S, I, R = res.T


plt.figure()
plt.grid()
plt.title("odeint method")
plt.plot(t, S, 'orange', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Removed')
plt.xlabel('Time t, [days]')
plt.ylabel('Numbers of individuals')
plt.ylim([0,N])
plt.legend()

plt.show();