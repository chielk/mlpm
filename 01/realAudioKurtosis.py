import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import sqrtm
import scipy.io.wavfile
from scipy import stats

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



S = np.c_[wav_data]
RealAudioKurtosis = []
for i in range(len(S[:,0])):
    RealAudioKurtosis.append(scipy.stats.kurtosis(S[:,i], fisher=True, bias=True))
    
print RealAudioKurtosis
