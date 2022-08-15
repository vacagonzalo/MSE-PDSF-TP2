import simpleaudio as sa
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    fs = 8000
    noise = np.load('chapu_noise.npy')
    N = len(noise)

    filter_test = np.array(np.load('filter.npy')[0]).astype(float)
    filtered = np.convolve(filter_test, noise)
    playObj = sa.play_buffer(np.array(np.real(filtered), np.int16), 1, 2, fs * 1)

    N_ = len(filter_test)
    nData_ = np.arange(0, N, 1)
    fData_ = nData_ * (fs / N) - (fs / 2)

    nData = np.arange(0, N, 1)
    fData = nData * (fs / N) - (fs / 2)

    fig = plt.figure()
    noise_ax = fig.add_subplot(3, 1, 1)
    noise_ax.set_xlim(0, N / fs)
    noise_ax.set_title("noisy signal", rotation = 0, fontsize = 10, va ="center")
    plt.plot(nData / fs, noise, 'b-', linewidth = 1, alpha = 0.75)
    plt.grid()

    spectrum = np.fft.fft(noise)
    spectrum_ax = fig.add_subplot(3, 1, 2)
    spectrum_ax.set_xlim((-fs / 2) - (fs / N), (fs / 2) + (fs / N))
    spectrum_ax.set_title("FFT(noisy signal)", rotation = 0, fontsize = 10, va ="center")
    plt.plot(fData, np.abs(np.fft.fftshift(spectrum) / N ** 2), 'r-', linewidth = 1, alpha = 0.75)
    plt.grid()

    convAxe         = fig.add_subplot(3,1,3)
    convolveNData = np.arange(0, len(filtered), 1)
    convolveTData = convolveNData/fs
    convLn,       = plt.plot(convolveTData, filtered, 'g-', label ="clean signal", linewidth = 1, alpha = 0.5)

    convAxe.grid(True)
    convAxe.set_xlim(0,convolveTData[-1])

    plt.show()