import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.linalg import sqrtm

# Signal generators
def sawtooth(x, period=0.2, amp=1.0, phase=0.):
    return (((x / period - phase - 0.5) % 1) - 0.5) * 2 * amp


def sine_wave(x, period=0.2, amp=1.0, phase=0.):
    return np.sin((x / period - phase) * 2 * np.pi) * amp


def square_wave(x, period=0.2, amp=1.0, phase=0.):
    return ((np.floor(2 * x / period - 2 * phase - 1) % 2 == 0).astype(float) - 0.5) * 2 * amp


def triangle_wave(x, period=0.2, amp=1.0, phase=0.):
    return (sawtooth(x, period, 1., phase) * square_wave(x, period, 1., phase) + 0.5) * 2 * amp

num_sources = 5
signal_length = 500
t = np.linspace(0, 1, signal_length)
#S = np.c_[sawtooth(t), sine_wave(t, 0.3), square_wave(t, 0.4), triangle_wave(t, 0.25), np.random.randn(t.size)].T
dummyKurtosis = []
Sawtooth = np.c_[sawtooth(t)]
SineWave = np.c_[sine_wave(t, 0.3)]
SquareWave = np.c_[square_wave(t, 0.4)]
TriangleWave= np.c_[triangle_wave(t, 0.25)]
dummyKurtosis.append(scipy.stats.kurtosis(Sawtooth, fisher=True, bias=True))
dummyKurtosis.append(scipy.stats.kurtosis(SineWave, fisher=True, bias=True))
dummyKurtosis.append(scipy.stats.kurtosis(SquareWave, fisher=True, bias=True))
dummyKurtosis.append(scipy.stats.kurtosis(TriangleWave, fisher=True, bias=True))
print dummyKurtosis
