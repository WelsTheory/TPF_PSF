import matplotlib.pyplot as plt
import numpy as np
import wave

data = wave.open("hola.wav","rb")

sample_freq = data.getframerate()
n_samples = data.getnframes()
signal = data.readframes(-1)

data.close()

t_audio = n_samples/sample_freq

print(t_audio)

signal_array = np.frombuffer(signal,dtype=np.int16)

times = np.linspace(0,t_audio,num=n_samples)
# 
plt.figure(figsize=(15,5))
plt.plot(times,signal_array)
plt.title("Audio signal")
plt.ylabel("Signal wave")
plt.xlabel("Time (s)")
plt.xlim(0,t_audio)
plt.show()

# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.io import wavfile
# import numpy as np

# sample_rate, samples = wavfile.read('hola.wav')
# frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

# sample_freq = samples.getframerate()
# n_samples = samples.getnframes()

# t_audio = n_samples/sample_freq
# audio = samples.readframes(-1)

# # times = np.linspace(0,t_audio,num=samples)

# plt.figure(figsize=(15,5))
# plt.plot(t_audio,audio)
# plt.title("Audio signal")
# plt.ylabel("Signal wave")
# plt.xlabel("Time (s)")
# plt.xlim(0,t_audio)
# plt.show()
