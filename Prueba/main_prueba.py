import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.animation import FuncAnimation

import serial
import time

from _fixedInt2 import DeFixedInt, arrayFixedInt

import wave

# generador de onda senoidal
def sin_signal(fs, f0, amp, N, fase=0, n=None):
    """\
    fs:   frecuencia de sampleo [Hz]
    f0:   frecuencia de la senoidal [Hz]
    fase: fase de la señal [rad]
    amp:  amplitud de la señal [0 a 1]
    N:    cantidad de muestras
    n:    numero de muestra a retornar.
          Si es None devuelve todo el arreglo, caso contrario devuelve solamente
          el valor para el instante de tiempo correspondiente a esa muestra
    """
    if n is not None:
        return amp * np.sin(fase + 2 * np.pi * f0 * n * (1/fs))
    
    discrete_time = np.arange(0, N/fs, 1/fs)
    discrete_signal = amp * np.sin(fase + 2 * np.pi * f0 * discrete_time)
    return discrete_signal, discrete_time

def plot_time_and_freq(signal, time, fs, f0, N):

    signal_fft = np.fft.fft(signal)
    signal_fft = np.concatenate((signal_fft[N//2:N],signal_fft[0:N//2]))/N
    freq_fft = np.arange(-fs/2,fs/2,fs/N)
    #freq_fft = np.arange(0, fs, fs/N)

    fig = plt.figure(figsize=(12,6))
    fig.add_subplot(1, 2, 1)
    plt.plot(time, signal, 'b-o', label="f0={}".format(f0))
    plt.title("Señal en el tiempo")
    plt.xlabel("tiempo")
    plt.ylabel("amplitud")
    plt.legend()
    plt.grid()

    fig.add_subplot(1, 2, 2)
    plt.plot(freq_fft, np.abs(signal_fft)**2, 'b-o')
    plt.title("Señal en frecuencia (magnitud)")
    plt.xlabel("frecuencia [Hz]")
    plt.ylabel("amplitud")
    plt.grid()

    plt.show()
    
    return signal_fft


def twos_comp(value, total_bits):
    if (value & (1 << (total_bits - 1))) != 0:   # chequeo de signo
        value = value - (1 << total_bits)        # negativo
    return value & ((2 ** total_bits) - 1)



serial_interface = serial.Serial('COM16', 460800)


N    = 512
fs   = 512
amp  = 1
fase = 0
fft_length = 64
f0   = 16           # fs / fft_length

sine_signal, sin_time = sin_signal(fs=fs, f0=f0, amp=amp, N=N, fase=fase)
sine_signal2, sin_time = sin_signal(fs=fs, f0=f0*4, amp=amp, N=N, fase=fase)
sine_signal3, sin_time = sin_signal(fs=fs, f0=f0*10, amp=amp, N=N, fase=fase)
# noise = np.random.normal(0, 1, len(sine_signal)) * 0.2

# final_signal = list(np.array(sine_signal) + np.array(sine_signal2) + np.array(sine_signal3))
# final_signal = final_signal + noise
# final_signal = sine_signal

def hola_array():
    hola = wave.open("hola.wav","rb")
    sample_freq = hola.getframerate()
    n_samples = hola.getnframes()
    signal = hola.readframes(-1)
    signal_array = np.frombuffer(signal,dtype=np.int16)
    discrete_time = np.arange(0, n_samples/sample_freq, 1/sample_freq)
    discrete_signal = 1 * np.sin(0 + 2 * np.pi * 20 * discrete_time)

    t_audio = n_samples/sample_freq
    print(t_audio)
    return discrete_signal, t_audio


final_signal = hola_array()

sine_fxp = arrayFixedInt(intWidth=0, fractWidth=15, N=final_signal)  # Q(1,15)

# ANIMATION
fig  = plt.figure()

ax1  = fig.add_subplot(2, 1, 1)
ln1, = plt.plot([0], [0], 'r-')
ax1.grid(True)
ax1.set_title("Señal en el tiempo enviada/recibida a la CIAA")

ax2  = fig.add_subplot(2, 2, 4)
ln2, = plt.plot([0], [0], 'b-o')
ax2.grid(True)
ax2.set_title("FFT en la CIAA")

ax3  = fig.add_subplot(2, 2, 3)
ln3, = plt.plot([0], [0], 'g-o')
ax3.grid(True)
ax3.set_title("FFT en Python")


datos_recibidos = []

def init_animation():
    global datos_recibidos, previous_header
    previous_header = 160000
    datos_recibidos = []
    ax1.set_xlim(0, N/fs)
    ax2.set_xlim(-fs/2, fs/2)
    ax3.set_xlim(-fs/2, fs/2)
    ax1.set_ylim(-1.2, 1.2)
    ax2.set_ylim(0, 0.3)
    ax3.set_ylim(0, 0.3)
    plt.draw()
    return ln1,ln2,ln3


def update_animation(t):
    global datos_recibidos, sine_fxp, final_signal, previous_header
    if len(datos_recibidos) >= len(sine_fxp):
        datos_recibidos = []


    fft_out_recibida = []
    max_value = []
    max_index = []
    max_freq = 0
    for index, num in enumerate(sine_fxp[t*fft_length:t*fft_length+fft_length]):
        # send data
        serial_interface.write(twos_comp(num.value, 16).to_bytes(2, byteorder='little', signed=False))
        # receive signal data
        raw_read_data = serial_interface.read(2)
        int_read_data = int.from_bytes(raw_read_data, byteorder='little', signed=True)
        datos_recibidos.append(int_read_data / 2**15)
        # receive fft data (real part)
        raw_read_data = serial_interface.read(2)
        int_read_data = int.from_bytes(raw_read_data, byteorder='little', signed=True)
        fft_real = int_read_data / (2**15)
        # receive fft data (imag part)
        raw_read_data = serial_interface.read(2)
        int_read_data = int.from_bytes(raw_read_data, byteorder='little', signed=True)
        fft_imag = 1j * int_read_data / (2**15)

        fft_out_recibida.append(fft_real + fft_imag)

        if index not in [0,1] and ((index+1) % fft_length == 0):

            raw_read_data = serial_interface.read(4)
            header = int.from_bytes(raw_read_data, byteorder='little', signed=False)
            # print(header)
            if previous_header != header:
                previous_header = header
                if header == 160000:
                    final_signal = list(np.array(sine_signal))
                elif header == 163200:
                    final_signal = list(0.8 * np.array(sine_signal) + 0.5 * np.array(sine_signal2))
                elif header == 163264:
                    final_signal = list(0.5 * np.array(sine_signal) + 0.7 * np.array(sine_signal2) + 0.6 * np.array(sine_signal3))
                else:
                    print("Sync error")
                    break

                sine_fxp = arrayFixedInt(intWidth=0, fractWidth=15, N=final_signal)  # Q(1,15)


            # receive max value in FFT
            raw_read_data = serial_interface.read(2)
            int_read_data = int.from_bytes(raw_read_data, byteorder='little', signed=True)
            max_value.append(int_read_data / (2**13))   # arm_cmplx_mag_squared_q15 output is Q3.13
            # receive max index in FFT
            raw_read_data = serial_interface.read(2)
            int_read_data = int.from_bytes(raw_read_data, byteorder='little', signed=True)
            max_index.append(int_read_data)
            # receive max freq in FFT
            raw_read_data = serial_interface.read(2)
            int_read_data = int.from_bytes(raw_read_data, byteorder='little', signed=True)
            # max_freq.append(int_read_data)
            max_freq = int_read_data


        # print(int_read_data, hex(int_read_data), twos_comp(num.value, 16), num.value)
        # if index % 25 == 0:
        #     print(index)


    ln1.set_data(sin_time[0:t*fft_length+fft_length], datos_recibidos)
    
    if fft_out_recibida != []:

        freq_fft = np.arange(-fs/2,fs/2,fs/fft_length)
        fft_mag = np.abs(fft_out_recibida)**2 * 2**6
        fft_mag = np.concatenate((fft_mag[fft_length//2:fft_length],fft_mag[0:fft_length//2]))/fft_length

        ln2.set_data(freq_fft, fft_mag)
        # ax2.set_ylim(0, max(fft_mag)*1.3)
        # ax2.legend(["Max freq = {}".format(max_freq[-1])])
        ax2.legend(["Max freq = {}".format(max_freq)])

        non_zero_bins = [index*(fs/fft_length) for index,fft_sample \
                in enumerate(np.abs(fft_out_recibida[0:0+fft_length//2])**2) if fft_sample > max_value[0]/10]

        signal_fft = np.fft.fft(datos_recibidos[t*fft_length:t*fft_length+fft_length])
        signal_fft = np.concatenate((signal_fft[fft_length//2:fft_length],signal_fft[0:fft_length//2]))/fft_length
        signal_mag = np.abs(signal_fft)**2
        ln3.set_data(freq_fft, signal_mag)
        # ax3.set_ylim(0, max(signal_mag)*1.3)
        # ax3.legend(["Max freq = {}".format(max(non_zero_bins))])

        print("Max freq CIAA = {}".format(max_freq))
        print("Max freq Python = {}".format(max(non_zero_bins)))

    # print(max_value)
    # print(max_index)
    # print(max_freq)

    # non_zero_bins = [index*(fs/fft_length) for index,fft_sample \
    #         in enumerate(np.abs(fft_out_recibida[0:0+fft_length//2])**2) if fft_sample > max_value[0]/10]
    # print(non_zero_bins)

    return ln1, ln2, ln3


real_time_animation = FuncAnimation(fig, update_animation, 30, init_func=init_animation, blit=True, interval=20, repeat=True)
plt.show()


serial_interface.close()





