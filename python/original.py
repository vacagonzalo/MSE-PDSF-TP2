import matplotlib.pyplot as plt
import numpy as np
import simpleaudio as sa


if __name__ == '__main__':
    noise = np.load('chapu_noise.npy')
    fs = 8000
    N = len(noise)
    playObj = sa.play_buffer(
        audio_data=noise, num_channels=1, bytes_per_sample=2, sample_rate=fs)

    nData = np.arange(0, N, 1)
    fData = nData * (fs / N) - (fs / 2)

    fig = plt.figure()
    noise_ax = fig.add_subplot(3, 1, 1)
    noise_ax.set_xlim(0, N / fs)
    noise_ax.set_title("noisy signal", rotation = 0, fontsize = 10, va ="center")
    plt.plot(nData / fs, noise, 'b-', linewidth = 2, alpha = 0.75)
    plt.grid()

    spectrum = np.fft.fft(noise)
    spectrum_ax = fig.add_subplot(3, 1, 2)
    spectrum_ax.set_xlim((-fs / 2) - (fs / N), (fs / 2) + (fs / N))
    spectrum_ax.set_title("FFT(noisy signal)", rotation = 0, fontsize = 10, va ="center")
    plt.plot(fData, np.abs(np.fft.fftshift(spectrum) / N ** 2), 'r-', linewidth = 2, alpha = 0.75)
    plt.grid()

    plt.show()