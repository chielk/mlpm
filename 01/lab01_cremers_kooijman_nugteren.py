import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy
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


# 1.4
def whiten(data):
    s = np.std(data, axis=0)
    return data / s

"""
def whiten(X):
    # zero mean
    mean = X.mean(axis=1)
    X = X - mean
    evs, phi = np.linalg.eig(X.dot(X.T))
    lam = np.sqrt(np.diag(evs))
    w = np.matrix(lam).I.dot(phi.T.dot(X))
    return w
"""



def random_nonsingular_matrix(d=2):
    """
    Generates a random nonsingular (invertible) matrix if shape d*d
    """
    epsilon = 0.1
    A = np.random.rand(d, d)
    while abs(np.linalg.det(A)) < epsilon:
        A = np.random.rand(d, d)
    return A


def plot_signals(X):
    """
    Plot the signals contained in the rows of X.
    """
    plt.figure()
    for i in range(X.shape[0]):
        ax = plt.subplot(X.shape[0], 1, i + 1)
        plt.plot(X[i, :])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def test_plot():
    # Generate data
    num_sources = 5
    signal_length = 500
    t = np.linspace(0, 1, signal_length)
    S = np.c_[sawtooth(t), sine_wave(t, 0.3), square_wave(t, 0.4), triangle_wave(t, 0.25), np.random.randn(t.size)].T

    plot_signals(S)


# 1.1
def make_mixtures(S, A):
    return A.dot(S)


def one_point_one():
    S = np.random.random((3, 5))
    A = np.random.random((3, 3))
    print make_mixtures(S, A)


# 1.2
def plot_histogram(X, bins=7):
    plt.hist(X.T, bins=bins)
    plt.show()


def apply(fun, X):
    f = numpy.vectorize(fun)
    return f(X)


def ICA(X, activation_function=lambda x: -np.tanh(x), learning_rate=0.001,
        min_delta=0.0001):
    G = np.matrix(np.identity(X.shape[0]))
    W = G.I
    i = 0
    while i < 10000:  # max number of iterations
        if i % 10 == 0:
            unmixed = W.I.dot(X)
            plt.scatter(unmixed[0], unmixed[1])
            plt.show()
        i += 1
        A = W.dot(X)
        Z = apply(activation_function, A)
        XI = W.T.dot(A)
        DeltaW = learning_rate * (W + Z.dot(XI.T))

        W += DeltaW
        if abs(DeltaW).sum() < min_delta:
            break
    return W


"""
C = np.eye(5)  # Dummy matrix; compute covariance here
ax = plt.imshow(C, cmap='gray', interpolation='nearest')


"""
import scipy.io.wavfile
def save_wav(data, out_file, rate):
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    scipy.io.wavfile.write(out_file, rate, scaled)


# Load audio sources
source_files = ['beet.wav', 'beet9.wav', 'beet92.wav', 'mike.wav', 'street.wav']
wav_data = []
sample_rate = None
for f in source_files:
    sr, data = scipy.io.wavfile.read(f)
    if sample_rate is None:
        sample_rate = sr
    else:
        assert(sample_rate == sr)
    #wav_data.append(data[:190000])  # cut off the last part so that all signals have same length
    wav_data.append(list(data[10000:10500]))  # cut off the last part so that all signals have same length

def one_point_two():
    plot_histogram(X)

A = np.random.random((2, 2))

S = np.c_[wav_data[:2]]
#plt.scatter(S[0], S[1])
#plt.show()

M = make_mixtures(S, A)

Whitened = whiten(M)
plt.scatter(Whitened[0], Whitened[1])
plt.show()
#plot_signals(M)
#one_point_two(np.matrix(wav_data))
#plt.scatter(M[0], Mw[0])
#plt.show()

W = ICA(Whitened)
#print W.dot(A)
print W
unmixed = W.I.dot(M)
plt.scatter(unmixed[0], unmixed[1])
plt.show()

"""
# Create source and measurement data
S = np.c_[wav_data]
plot_signals(S)

# Requires your function make_mixtures
#X = make_mixtures(S, make_random_nonsingular(S.shape[0]))
#plot_signals(X)
# Save mixtures to disk, so you can listen to them in your audio player
#for i in range(X.shape[0]):
#    save_wav(X[i, :], 'X' + str(i) + '.wav', sample_rate)
"""
