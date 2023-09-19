"""a module for constructing time-series data
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import h5py

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

#-------------------------------------------------

def load_data(path, verbose=False):
    """read time-series from hdf file
    """
    if verbose:
        print('loading time-series data from: '+path)

    with h5py.File(path, 'r') as obj:
        to = obj.attrs['t0']
        dt = obj.attrs['dt']
        data = obj['data'][:]

    return to + np.arange(len(data))*dt, data

#------------------------

def write_data(path, t, data, verbose=False):
    """write time-series to hdf file
    """
    if verbose:
        print('writing time-series data to: '+path)

    with h5py.File(path, 'w') as obj:
        obj.attrs.create('t0', t[0])
        obj.attrs.create('dt', t[1]-t[0])
        obj.create_dataset('data', data=data)

#-------------------------------------------------

def load_params(path, verbose=False):
    """read signal params from hdf file
    """
    if verbose:
        print('loading signal parameters from: '+path)

    with h5py.File(path, 'r') as obj:
        params = obj['params'][...]
    return params

#------------------------

def write_params(path, params, verbose=False):
    """write signal params to hdf file
    """
    if verbose:
        print('writing signal parameters to: '+path)

    with h5py.File(path, 'w') as obj:
        obj.create_dataset('params', data=params)

#-------------------------------------------------

def time_array(t0, dur, sample_rate, verbose=False):
    """construct an array of uniformly spaced times
    """
    if verbose:
        print('constructing time array starting at %d with duration %d and %d samples (sample_rate = %d)' \
            % (t0, dur, dur*sample_rate, sample_rate))
    dt = 1./sample_rate
    return t0 + np.arange(dur*sample_rate)* dt

#-------------------------------------------------

def sine_gaussian(t, A, to, fo, phio, tau):
    """h(t) = A * cos(2*pi*fo*(t-to) + phio) * exp(-0.5*(t-to)**2/tau**2)
    """
    return A * np.cos(2*np.pi*fo*(t-to) + phio) * np.exp(-0.5*(t-to)**2/tau**2)

#-------------------------------------------------

def draw_noise(t, sigma, verbose=False):
    """stationary, white Gaussian noise
    """
    if verbose:
        print('drawing %d iid noise samples from zero-mean Gaussian with sigma=%.3f' % (len(t), sigma))
    return np.random.normal(size=len(t)) * sigma

#------------------------

def draw_signals(
        num,
        A=(0.0, 1.0),
        to=(0.0, 100.0),
        fo=(10.0, 100.0),
        phio=(-np.pi, +np.pi),
        tau=(1.0, 10.0),
        verbose=False,
    ):
    """draw sine-Gaussian parameters from the relevant priors \
(assumed to be Pareto distributions or Uniform distributions)
    """
    if verbose:
        print('drawing %d signals' % num)
        print('    A ~ uniform(%.3f, %.3f)' % tuple(A))
        print('    to ~ uniform(%.3f, %.3f)' % tuple(to))
        print('    fo ~ uniform(%.3f, %.3f)' % tuple(fo))
        print('    phio ~ uniform(%.3f, %.3f)' % tuple(phio))
        print('    tau ~ uniform(%.3f, %.3f)' % tuple(tau))

    # make data structure
    params = np.empty((num,), dtype=[('A', float), ('to', float), ('fo', float), ('phio', float), ('tau', float)])

    # draw parameters
    params['A'] = A[0] + (A[1]-A[0])*np.random.random(size=num)
    params['to'] = to[0] + (to[1]-to[0])*np.random.random(size=num)
    params['fo'] = fo[0] + (fo[1]-fo[0])*np.random.random(size=num)
    params['phio'] = phio[0] + (phio[1]-phio[0])*np.random.random(size=num)
    params['tau'] = tau[0] + (tau[1]-tau[0])*np.random.random(size=num)

    # return
    return params

#------------------------

def draw_data(t, sigma, rate, verbose=False, **priors):
    """make a fake data stream with Poisson distrib of the number of sine-Gaussians given the rate with \
stationary, white Gaussian noise described by sigma
    """
    dt = t[1] - t[0]
    dur = len(t) * dt

    # generate noise
    data = draw_noise(t, sigma, verbose=False)

    # draw the number of signals
    if verbose:
        print('expected number of signals = %.3f' % (rate*dur))
    num = int(np.random.poisson(lam=rate*dur))

    # draw the parameters of signals
    params = draw_signals(num, verbose=verbose, to=(t[0], dur), **priors)

    # iterate and add to data
    for ind in range(num):
        data += sine_gaussian(
            t,
            params['A'][ind],
            params['to'][ind],
            params['fo'][ind],
            params['phio'][ind],
            params['tau'][ind],
        )

    # return
    return data, params

#------------------------

def plot_data(t, data, params):
    """make a time-domain plot of the data with the location of signals overlaid
    """

    fig = plt.figure(figsize=(10,2))
    ax = fig.add_axes([0.08, 0.20, 0.90, 0.78])

    ax.plot(t, data, color='k', alpha=0.25)

    for ind in range(len(params)):
        sel = np.abs(t - params['to'][ind]) <= 6*params['tau'][ind]
        h = sine_gaussian(
            t[sel],
            params['A'][ind],
            params['to'][ind],
            params['fo'][ind],
            params['phio'][ind],
            params['tau'][ind],
        )
        ax.plot(t[sel], h, alpha=0.75)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$h$')

    ax.tick_params(
        left=True,
        right=True,
        top=True,
        bottom=True,
        direction='in',
        which='both',
    )

    ax.set_xlim(t[0], t[-1])

    return fig

#------------------------

def savefig(path, fig, verbose=False):
    if verbose:
        print('saving: '+path)
    fig.savefig(path)
    plt.close(fig)
