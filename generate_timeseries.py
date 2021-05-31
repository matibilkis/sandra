import numpy as np
from scipy.integrate import odeint
from scipy.special import legendre
from tqdm import tqdm

def generate_lorenz_fulldata(n_ics):
    """
    Generate initial conditions for Lorenz system.
    Create toy data with Lorenz dynamics and
    Legendre polinomials in a 1D grid

    Arguments:
    n_ics - number of initial conditions
    Returns:
    X, dX - Array of n_ics time series and their derivatives with 250 samples in time and 128 in the spatial dimension

    """
    ic_means = np.array([0,0,25])
    ic_widths = 2*np.array([36,48,41])

    # training data
    ics = ic_widths*(np.random.rand(n_ics, 3)-.5) + ic_means

    d = 3 #Dimension del sistema

    dt = 0.02; ti = 0.; tf = 5.
    t = np.arange(ti,tf,dt)
    n_steps = len(t)

    Z = np.zeros((n_ics,n_steps,d))
    dZ = np.zeros(Z.shape)
    for i in tqdm(range(n_ics)):
        Z[i], dZ[i]= simulate_lorenz(ics[i], t)

    n_points = 128

    space_modes = legendre_polinomials(n_points)

    X = np.zeros((n_ics,n_steps,n_points))
    x1 = np.zeros(X.shape)
    x2 = np.zeros(X.shape)
    x3 = np.zeros(X.shape)
    x4 = np.zeros(X.shape)
    x5 = np.zeros(X.shape)
    x6 = np.zeros(X.shape)
    for i in range(n_ics):
        for j in range(n_steps):
            x1[i,j] = space_modes[0] * Z[i,j,0]
            x2[i,j] = space_modes[1] * Z[i,j,1]
            x3[i,j] = space_modes[2] * Z[i,j,2]
            x4[i,j] = space_modes[3] * Z[i,j,0]**3
            x5[i,j] = space_modes[4] * Z[i,j,1]**3
            x6[i,j] = space_modes[5] * Z[i,j,2]**3

    X = x1 + x2 + x3 + x4 + x5 + x6

    dX = np.zeros((n_ics,n_steps,n_points))
    dx1 = np.zeros(X.shape)
    dx2 = np.zeros(X.shape)
    dx3 = np.zeros(X.shape)
    dx4 = np.zeros(X.shape)
    dx5 = np.zeros(X.shape)
    dx6 = np.zeros(X.shape)
    for i in range(n_ics):
        for j in range(n_steps):
            dx1[i,j] = space_modes[0] * dZ[i,j,0]
            dx2[i,j] = space_modes[1] * dZ[i,j,1]
            dx3[i,j] = space_modes[2] * dZ[i,j,2]
            dx4[i,j] = space_modes[3] * 2 * Z[i,j,0] * dZ[i,j,0]
            dx5[i,j] = space_modes[4] * 2 * Z[i,j,1] * dZ[i,j,1]
            dx6[i,j] = space_modes[5] * 2 * Z[i,j,2] * dZ[i,j,2]

    dX = dx1 + dx2 + dx3 + dx4 + dx5 + dx6

    return X, dX


def simulate_lorenz(z0, t, sigma=10., beta=8/3, rho=28.):
    """
    Simulate the Lorenz dynamics.
    Arguments:
        z0 - Initial condition in the form of a 3-value list or array.
        t - Array of time points at which to simulate.
        sigma, beta, rho - Lorenz parameters
    Returns:
        z, dz - Arrays of the trajectory values and their 1st and 2nd derivatives.
    """
    f = lambda z,t : [sigma*(z[1] - z[0]), z[0]*(rho - z[2]) - z[1], z[0]*z[1] - beta*z[2]]

    z = odeint(f, z0, t)

    dt = t[1] - t[0]
    dz = np.zeros(z.shape)
    for i in range(t.size):
        dz[i] = f(z[i],dt*i)
    return z, dz


def legendre_polinomials(n_points):
    n = n_points
    L = 1
    y_array = np.linspace(-L,L,128)

    modes = np.zeros((6,n))
    for i in range(6):
        modes[i] = legendre(i)(y_array)

    return modes

if __name__ == '__main__':
    import os
    x, x_dot = generate_lorenz_fulldata(2048)
    os.makedirs("datalorenz",exist_ok=True)
    np.save("datalorenz/2048",[x,x_dot])
# Test
#X, dX = generate_lorenz_fulldata(2048)
