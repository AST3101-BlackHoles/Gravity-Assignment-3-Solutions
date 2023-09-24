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

def tukey_window(N, alpha=0.5):
    """
    generate a tukey window

    The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2 that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and at \alpha = 0 it becomes a Hann window.
        """
    # Special cases
    if alpha <= 0:
        return np.ones(N) #rectangular window
    elif alpha >= 1:
        return np.hanning(N)

    # Normal case
    x = np.linspace(0, 1, N)
    w = np.ones(x.shape)

    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))

    return w

#------------------------

def dft(vec, dt):
    """
    computes the DFT of vec
    returns the one-sides spectrum
    """
    N = len(vec)

    dft_vec = np.fft.rfft(vec)*dt
    freqs = np.fft.rfftfreq(N, d=dt)

    return dft_vec, freqs

def idft(dft_vec, dt):
    """
    computes the inverse DFT of vec
    takes in the one-sided spectrum
    """
    vec = np.fft.irfft(dft_vec) / dt # undo the multiplicative factor added in dft(vec, dt)
    time = np.arange(0, len(vec))*dt

    return vec, time

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

def sine_gaussian_time_domain(t, A, to, fo, phio, tau):
    """h(t) = A * cos(2*pi*fo*(t-to) + phio) * exp(-0.5*(t-to)**2/tau**2)
    """
    return A * np.cos(2*np.pi*fo*(t-to) + phio) * np.exp(-0.5*(t-to)**2/tau**2)

#------------------------

def sine_gaussian_freq_domain(f, A, to, fo, phio, tau):
    """fourier transform of the sine_gaussian
    """
    return A * (np.pi/2)**0.5 * tau * np.exp(-2j*np.pi*f*to) \
        * (np.exp(-1j*phio - 2*np.pi**2*tau**2*(f+fo)**2) + np.exp(+1j*phio - 2*np.pi**2*tau**2*(f-fo)**2))

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

    # make data structure
    params = np.empty(
        (num,),
        dtype=[('log10A', float), ('A', float), ('to', float), ('fo', float), ('phio', float), ('tau', float)],
    )

    # draw parameters
    for name, vals in [('log10A', np.log10(A)), ('to', to), ('fo', fo), ('phio', phio), ('tau', tau)]:
        if len(vals) == 1:
            if verbose:
                print('    %s = %.3f' % (name, vals[0]))
            params[name] = vals[0]

        elif len(vals) == 2:
            if verbose:
                print('    %s ~ uniform(%.3f, %.3f)' % (name, vals[0], vals[1]))
            params[name] = vals[0] + (vals[1]-vals[0])*np.random.random(size=num)

        else:
            raise RuntimeError('prior for %s not understood : %s' % (name, vals))

    params['A'] = 10**params['log10A']

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
        data += sine_gaussian_time_domain(
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

def plot_data(t, data, params, color='k', alpha=0.50, ylabel='$h$', fig=None):
    """make a time-domain plot of the data with the location of signals overlaid
    """

    if fig is None:
        fig = plt.figure(figsize=(10,2))
        ax = fig.add_axes([0.08, 0.20, 0.90, 0.78])
    else:
        ax = fig.gca()

    ax.plot(t, data, color=color, alpha=alpha)

    for ind in range(len(params)):
        sel = np.abs(t - params['to'][ind]) <= 6*params['tau'][ind]
        h = sine_gaussian_time_domain(
            t[sel],
            params['A'][ind],
            params['to'][ind],
            params['fo'][ind],
            params['phio'][ind],
            params['tau'][ind],
        )
        ax.plot(t[sel], h, alpha=alpha)

    ax.set_xlabel('$t$')
    ax.set_ylabel(ylabel)

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
        print('    saving: '+path)
    fig.savefig(path)
    plt.close(fig)
