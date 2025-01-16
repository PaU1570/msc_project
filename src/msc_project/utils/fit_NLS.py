import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as clr
import argparse

def distribution_function_mesa(z, Gamma, z1, z2):
    """
    Distribution function g(z) with a flat region, as given in Tagantsev et al. (2002).
    It is defined by the parameters z1, z2, and Γ.
    """
    def _distribution_function_mesa(z, Gamma, z1, z2):
        h = 1 / (z2 - z1 + Gamma * np.pi)
        if z < z1:
            return (Gamma**2 * h) / ((z - z1)**2 + Gamma**2)
        elif z > z2:
            return (Gamma**2 * h) / ((z - z2)**2 + Gamma**2)
        else:
            return h
        
    return np.vectorize(_distribution_function_mesa)(z, Gamma, z1, z2)
    
def distribution_function_lorentzian(z, A, omega, tl):
    """
    Distribution function g(z) with a Lorentzian shape, as used in Kondratyuk & Chouprik (2022).
    """
    def _distribution_function_lorentzian(z, A, omega, tl):
        return (A / np.pi) * (omega / ((z - np.log10(tl))**2 + omega**2))
    
    return np.vectorize(_distribution_function_lorentzian)(z, A, omega, tl)

def polarization_mesadist(t, Gamma, z1, z2):
    """
    Polarization function P(t) arising from a mesa-like distribution function g(z) as given in Tagantsev et al. (2002).
    """
    def _polarization_mesadist(t, Gamma, z1, z2):
        z0 = np.log10(t)
        h = 1 / (z2 - z1 + Gamma * np.pi)
        if z0 < z1:
            return Gamma * h * (np.pi/2 - np.arctan((z1 - z0) / Gamma))
        elif z0 > z2:
            return Gamma * h * (np.pi/2 + (z2-z1)/Gamma + np.arctan((z0 - z2) / Gamma))
        else:
            return Gamma * h * (np.pi/2 + (z0-z1)/Gamma)
        
    return np.vectorize(_polarization_mesadist)(t, Gamma, z1, z2)
    
def polarization_lorentzian(t, A, omega, tl):
    """
    Polarization function P(t) arising from a Lorentzian distribution function g(z) as used in Kondratyuk & Chouprik (2022).
    """
    def _polarization_lorentzian(t, A, omega, tl):
        z0 = np.log10(t)
        return (A/np.pi) * np.arctan((z0 - np.log10(tl)) / omega)
    
    return np.vectorize(_polarization_lorentzian)(t, A, omega, tl)

def fit_polarization(x, y, p0=None, type="mesa"):
    """
    Fit the polarization data to a distribution function.
    """
    x = np.array(x)
    y = np.array(y)

    if type == "mesa":
        if p0 is None:
            p0 = [0.5, -7, -5]
        popt, pcov = curve_fit(polarization_mesadist, x, y, p0=p0)
        return popt
    elif type == "lorentzian":
        popt, pcov = curve_fit(polarization_lorentzian, x, y)
        return popt
    else:
        raise ValueError("Unknown type of distribution function")

def plot_polarization(data, ax=None, cmap=plt.cm.plasma, fit=False):
    if ax is None:
        fig, ax = plt.subplots()

    norm = clr.Normalize()       
    colors = cmap(norm(data.columns[1:].astype(float)))
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    cbar.set_label("V")
    
    for col, c in zip(data.columns[1:], colors):
        if fit:
            ax.scatter(data['Pulse Width'], data[col], color=c, marker='o')
        else:
            ax.plot(data['Pulse Width'], data[col], color=c, marker='o')
        if fit:
            try:
                popt = fit_polarization(data['Pulse Width'], data[col])
                t = np.logspace(np.log10(data['Pulse Width'].min()), np.log10(data['Pulse Width'].max()), 1000)
                p = polarization_mesadist(t, *popt)
                ax.plot(t, p, color=c, ls='--')
            except:
                print("Failed to fit", col)
                pass

    if ax is None:
        ax.set(xlabel = 't', ylabel = 'p', xscale='log')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input file')

    args = parser.parse_args()

    data = pd.read_csv(args.input)
    # keep only even columns
    data = data.iloc[:, ::2]
    # fix column names
    data.columns = [col.split('V')[0] for col in data.columns]
    # convert all the numeric values to float
    data = data.apply(pd.to_numeric, errors='coerce')

    plt.style.use('ggplot')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_polarization(data, ax=ax, fit=True)

    ax.set(xlabel = 't', ylabel = 'p', xscale='log')
    
    plt.show()
