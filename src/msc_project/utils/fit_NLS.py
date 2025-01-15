import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def distribution_function_mesa(z, Gamma, z1, z2):
    """
    Distribution function g(z) with a flat region, as given in Tagantsev et al. (2002).
    It is defined by the parameters z1, z2, and Γ.
    """
    h = 1 / (z2 - z1 + Gamma * np.pi)
    if z < z1:
        return (Gamma**2 * h) / ((z - z1)**2 + Gamma**2)
    elif z > z2:
        return (Gamma**2 * h) / ((z - z2)**2 + Gamma**2)
    else:
        return h
    
def distribution_function_lorentzian(z, A, omega, tl):
    """
    Distribution function g(z) with a Lorentzian shape, as used in Kondratyuk & Chouprik (2022).
    """
    
    return (A / np.pi) * (omega / ((z - np.log10(tl))**2 + omega**2))

def polarization_mesadist(t, Gamma, z1, z2):
    """
    Polarization function P(t) arising from a mesa-like distribution function g(z) as given in Tagantsev et al. (2002).
    """
    z0 = np.log10(t)
    h = 1 / (z2 - z1 + Gamma * np.pi)
    if z0 < z1:
        return Gamma * h * (np.pi/2 - np.arctan((z1 - z0) / Gamma))
    elif z0 > z2:
        return Gamma * h * (np.pi/2 + (z2-z1)/Gamma + np.arctan((z0 - z2) / Gamma))
    else:
        return Gamma * h * (np.pi/2 + (z0-z1)/Gamma)
    
def polarization_lorentzian(t, A, omega, tl):
    """
    Polarization function P(t) arising from a Lorentzian distribution function g(z) as used in Kondratyuk & Chouprik (2022).
    """
    z0 = np.log10(t)
    return (A/np.pi) * np.arctan((z0 - np.log10(tl)) / omega)

fig, ax = plt.subplots()

t0 = np.logspace(-9, -2, 1000)
z = np.log10(t0)

Gamma = 0.5
z1 = -7
z2 = -6
g = np.vectorize(distribution_function_mesa)(z, Gamma, z1, z2)
gl = np.vectorize(distribution_function_lorentzian)(z, 1, 0.1, 1e-5)
ax.plot(t0, g)
ax.plot(t0, gl)
ax.set_xlabel('t0')
ax.set_ylabel('g(z)')
ax.set_xscale('log')

ax.vlines(10**z1, 0, 1, color='r', linestyle='--')
ax.vlines(10**z2, 0, 1, color='r', linestyle='--')
plt.show()
