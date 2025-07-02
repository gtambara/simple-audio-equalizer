import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog
from pydub import AudioSegment
import threading
import queue

# Constantes
FS = 44100
N = 1024
M = 201
BANDS = [100, 330, 1000, 3300, 10000]
CUTOFFS = [np.sqrt(BANDS[i] * BANDS[i+1]) for i in range(len(BANDS)-1)]
CUTOFFS = [0] + CUTOFFS + [FS/2]

# FIR com janela de Hamming
def design_bandpass(low, high, fs, M):
    h = []
    for n in range(M):
        m = n - (M-1)/2
        if m == 0:
            val = 2*(high-low)/fs
        else:
            val = (np.sin(2*np.pi*high*m/fs) - np.sin(2*np.pi*low*m/fs)) / (np.pi*m)
        w = 0.54 - 0.46*np.cos(2*np.pi*n/(M-1))
        h.append(val*w)
    return np.array(h)

filters = [design_bandpass(CUTOFFS[i], CUTOFFS[i+1], FS, M) for i in range(len(BANDS))]

root = tk.Tk()
root.title("Spectre and Equalizer")

gains_dB = [tk.DoubleVar(value=0.0) for _ in range(5)]
def dB_to_linear(db): return 10**(db/20)

input_buffer = np.zeros(N+M-1)
imported_audio = np.array([])
audio_pos = 0
spec_queue = queue.Queue()

def fast_convolve(x, h):
    x = np.asarray(x)
    h = np.asarray(h)[::-1]  # reverse filter

    strides = np.lib.stride_tricks.sliding_window_view(x, len(h))
    return np.sum(strides * h, axis=1)

def fft(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)

    if N == 0:
        return np.array([], dtype=complex)
    if N == 1:
        return x

    if N % 2 != 0:
        # Pad to next power of 2
        next_pow2 = 1 << (N - 1).bit_length()
        x = np.pad(x, (0, next_pow2 - N), mode='constant')
        N = next_pow2

    even = fft(x[::2])
    odd = fft(x[1::2])

    # Protect against mismatched sizes
    min_len = min(len(even), len(odd))
    even = even[:min_len]
    odd = odd[:min_len]
    terms = np.exp(-2j * np.pi * np.arange(min_len) / (2 * min_len)) * odd

    return np.concatenate([even + terms, even - terms])

def rfft(x):
    """Real-input FFT returning only the positive frequencies"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    X = fft(x)
    return X[:N//2 + 1]

def rfftfreq(n, d=1.0):
    """Return frequency bins"""
    return np.arange(n//2 + 1) / (n * d)

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

def compute_fft_bands(samples):
    window = np.hamming(len(samples))
    windowed = samples * window

    # Ensure length is a power of 2 by padding with zeros
    n = len(windowed)
    if (n & (n - 1)) != 0:  # not a power of 2
        target_len = next_power_of_2(n)
        pad_width = target_len - n
        windowed = np.pad(windowed, (0, pad_width), mode='constant')

    spectrum = rfft(windowed)
    freqs = rfftfreq(len(windowed), 1 / FS)
    magnitudes = np.abs(spectrum)

    bands = np.logspace(np.log10(20), np.log10(FS / 2), num=11)
    bar_values = []
    for i in range(10):
        idx = np.where((freqs >= bands[i]) & (freqs < bands[i+1]))[0]
        if len(idx) > 0:
            bar_values.append(np.sqrt(np.mean(magnitudes[idx] ** 2)))
        else:
            bar_values.append(0)
    return bar_values

def audio_callback(outdata, frames, time, status):
    global input_buffer, audio_pos, imported_audio

    if len(imported_audio) > 0:
        if audio_pos + N > len(imported_audio):
            audio_pos = 0
        block = imported_audio[audio_pos:audio_pos+N]
        audio_pos += N
    else:
        block = np.random.normal(0, 1e-6, N).astype(np.float32)

    #Shift
    input_buffer[:-N] = input_buffer[N:]
    input_buffer[-N:] = block

    # Combined filter
    combined_filter = np.zeros(M)
    for i in range(5):
        gain = dB_to_linear(gains_dB[i].get())
        combined_filter += gain * filters[i]

    # Convolution
    y = fast_convolve(input_buffer, combined_filter)
    y = np.clip(y, -1, 1).astype(np.float32)

    outdata[:, 0] = y
    spec_queue.put(y.copy())


# Visualizer
def spectrum_visualizer(canvas, bars):
    while True:
        try:
            samples = spec_queue.get(timeout=1)
            if len(samples) == 0 or (len(samples) & (len(samples) - 1)) != 0:
                continue  # skip if not power of 2 or empty
            mags = compute_fft_bands(samples)
            max_mag = max(mags) + 1e-6
            for i, mag in enumerate(mags):
                height = 100 * (mag / max_mag)
                canvas.coords(bars[i], i*15+10, 110, i*15+20, 110 - height)
        except queue.Empty:
            for bar in bars:
                canvas.coords(bar, 0, 0, 0, 0)
            continue

# MP3
def load_mp3():
    global imported_audio, audio_pos
    file_path = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3")])
    if file_path:
        print("Importando:", file_path)
        audio = AudioSegment.from_file(file_path).set_channels(1).set_frame_rate(FS)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
        imported_audio = samples
        audio_pos = 0

# GUI sliders
tk.Label(root, text="Ganhos (dB)").pack()
frame = tk.Frame(root)
frame.pack()
for i, freq in enumerate(BANDS):
    ttk.Label(frame, text=f"{freq} Hz").grid(row=0, column=i)
    scale = tk.Scale(frame, from_=24, to=-24, variable=gains_dB[i])
    scale.grid(row=1, column=i)

# Canvas and bars
canvas = tk.Canvas(root, width=170, height=120, bg="#005555")
canvas.pack()
bars = [canvas.create_rectangle(i*15+10, 110, i*15+20, 110, fill='white') for i in range(10)]

# MP3 Button
btn = tk.Button(root, text="Importar MP3", command=load_mp3)
btn.pack(pady=10)

# Thread
threading.Thread(target=spectrum_visualizer, args=(canvas, bars), daemon=True).start()

# Audio
stream = sd.OutputStream(samplerate=FS, blocksize=N, dtype='float32', channels=1, callback=audio_callback)
stream.start()
root.mainloop()
stream.stop()