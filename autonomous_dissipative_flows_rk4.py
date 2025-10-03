
import numpy as np

def rk4_step(f, y, t, dt):
    k1 = np.asarray(f(y, t))
    k2 = np.asarray(f(y + dt/2 * k1, t + dt/2))
    k3 = np.asarray(f(y + dt/2 * k2, t + dt/2))
    k4 = np.asarray(f(y + dt * k3, t + dt))
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def rk4_integrate(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        y[i] = rk4_step(f, y[i-1], t[i-1], dt)
    return y

def autonomous_dissipative_flows_rk4(system, dynamic_state=None, L=None, fs=None,
                                 SampleSize=None, parameters=None,
                                 InitialConditions=None):
    if fs is None:
        fs = 100
    if SampleSize is None:
        SampleSize = 2000
    if L is None:
        L = 100.0
    t = np.linspace(0, L, int(L * fs))

    if system == 'lorenz':
        if parameters is not None and len(parameters) == 3:
            rho, sigma, beta = parameters
        else:
            rho, sigma, beta = 28.0, 10.0, 8.0 / 3.0

        def lorenz(state, t):
            x, y, z = state
            dxdt = sigma * (y - x)
            dydt = x * (rho - z) - y
            dzdt = x * y - beta * z
            return dxdt, dydt, dzdt

        if InitialConditions is None:
            InitialConditions = [1.0, 1.0, 1.0]

        states = rk4_integrate(lorenz, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:], (states[:, 2])[-SampleSize:]]
        t = t[-SampleSize:]

    elif system == 'van_der_pol':
        if parameters is not None and len(parameters) == 1:
            mu = parameters[0]
        else:
            mu = 1.0

        def van_der_pol(state, t):
            x, y = state
            dxdt = y
            dydt = -x + mu * (1 - x**2) * y 
            return dxdt, dydt

        if InitialConditions is None:
            InitialConditions = [1.0, 0.0]

        states = rk4_integrate(van_der_pol, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
        t = t[-SampleSize:]

    elif system == 'duffing':
        if parameters is not None and len(parameters) == 5:
            delta, alpha, beta, gamma, omega = parameters
        else:
            delta, alpha, beta, gamma, omega = 0.1, -1, 1, 0.3, 1.25

        def duffing(state, t):
            x, y = state
            dxdt = y
            dydt = - alpha * x - beta * x**3 - delta * y + gamma * np.cos(omega * t)
            return dxdt, dydt

        if InitialConditions is None:
            InitialConditions = [1.0, 0.0]

        states = rk4_integrate(duffing, InitialConditions, t)
        ts = [(states[:, 0])[-SampleSize:], (states[:, 1])[-SampleSize:]]
        t = t[-SampleSize:]

    else:
        raise ValueError("El sistema solicitado no está disponible. Usa 'lorenz', 'van_der_pol' o 'duffing'.")

    return t, ts
