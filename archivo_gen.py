import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc
import wave
import sys
np.set_printoptions(threshold=sys.maxsize)

data = wave.open("hola.wav","rb")

sample_freq = data.getframerate()
print(sample_freq)
n_samples = data.getnframes()
print(n_samples)
signal = data.readframes(-1)

data.close()

t_audio = n_samples/sample_freq

print(t_audio)

signal_array = np.frombuffer(signal,dtype=np.int16)
print(signal_array[:])
signal_array.tofile('hello.csv', sep=',')

times = np.linspace(0,t_audio,num=n_samples)

fs   = 44100
sec  = 1
t    = np.arange(0,sec,1/fs)

note=np.zeros(len(t))
L=110  #tiene que matchear con la señal creada en la CIAA
OFFSET=1000
for i in range(L):
    note[i+OFFSET]=-(2**15-1)*i/L
for i in range(L,L+L):
    note[i+OFFSET]=(2**15-1)*(i-L)/L

#probar agregarndo ruido
# note+=np.random.normal(0,((2**15)-1)/5,len(t))

# fig=plt.figure(1)
# plt.plot(t,note)
# plt.show()

plt.figure(figsize=(15,5))
plt.plot(times,signal_array)
plt.title("Audio signal")
plt.ylabel("Signal wave")
plt.xlabel("Time (s)")
plt.xlim(0,t_audio)
plt.show()