import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog
from pydub import AudioSegment
import threading
import queue
import matplotlib.pyplot as plt

# Constants
FS = 44100
N = 1024
M = 401
BANDS = [100, 330, 1000, 3300, 10000]

# Calculate cutoff frequencies as midpoints (geometric mean) between centers
CUTOFFS = [np.sqrt(BANDS[i] * BANDS[i+1]) for i in range(len(BANDS)-1)]
CUTOFFS = [0] + CUTOFFS + [FS/2]

# FIR Bandpass filter design with Hamming window
def design_bandpass(low, high, fs, M):
    h = []
    for n in range(M):
        m = n - (M-1)/2
        if m == 0:
            val = 2*(high - low)/fs
        else:
            val = (np.sin(2*np.pi*high*m/fs) - np.sin(2*np.pi*low*m/fs)) / (np.pi*m)
        w = 0.54 - 0.46*np.cos(2*np.pi*n/(M-1))
        h.append(val * w)
    return np.array(h)

filters = [design_bandpass(CUTOFFS[i], CUTOFFS[i+1], FS, M) for i in range(len(BANDS))]

root = tk.Tk()
root.title("Spectre and Equalizer")

gains_dB = [tk.DoubleVar(value=0.0) for _ in range(5)]
def dB_to_linear(db): return 10**(db/20)

input_buffer = np.zeros(N + M - 1)
imported_audio = np.array([])
audio_pos = 0
spec_queue = queue.Queue()

def fast_convolve(x, h):
    x = np.asarray(x)
    h = np.asarray(h)[::-1]
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
        next_pow2 = 1 << (N - 1).bit_length()
        x = np.pad(x, (0, next_pow2 - N), mode='constant')
        N = next_pow2
    even = fft(x[::2])
    odd = fft(x[1::2])
    min_len = min(len(even), len(odd))
    even = even[:min_len]
    odd = odd[:min_len]
    terms = np.exp(-2j * np.pi * np.arange(min_len) / (2 * min_len)) * odd
    return np.concatenate([even + terms, even - terms])

def rfft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    X = fft(x)
    return X[:N//2 + 1]

def rfftfreq(n, d=1.0):
    return np.arange(n//2 + 1) / (n * d)

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

def compute_fft_bands(samples):
    window = np.hamming(len(samples))
    windowed = samples * window
    n = len(windowed)
    if (n & (n - 1)) != 0:
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

    input_buffer[:-N] = input_buffer[N:]
    input_buffer[-N:] = block

    combined_filter = np.zeros(M)
    for i in range(5):
        gain = dB_to_linear(gains_dB[i].get())
        combined_filter += gain * filters[i]

    y = fast_convolve(input_buffer, combined_filter)
    y = np.clip(y, -1, 1).astype(np.float32)

    outdata[:, 0] = y
    spec_queue.put((block.copy(), y.copy()))

def spectrum_visualizer(canvas, bars_eq, bars_red):
    while True:
        try:
            # Flush old items, keep only the most recent
            while True:
                item = spec_queue.get(timeout=0.1)
                if spec_queue.empty():
                    break
        except queue.Empty:
            continue

        original, equalized = item

        mags_orig = compute_fft_bands(original)
        mags_eq = compute_fft_bands(equalized)

        for i in range(10):
            # convert magnitude to dB, clamp to Y range
            mag_orig = max(mags_orig[i], 1e-10)
            mag_eq = max(mags_eq[i], 1e-10)

            db_orig = 20 * np.log10(mag_orig)
            db_eq = 20 * np.log10(mag_eq)

            db_orig = max(Y_TICKS_DB[0], min(db_orig, Y_TICKS_DB[-1]))
            db_eq = max(Y_TICKS_DB[0], min(db_eq, Y_TICKS_DB[-1]))

            rel_orig = (db_orig - Y_TICKS_DB[0]) / (Y_TICKS_DB[-1] - Y_TICKS_DB[0])
            rel_eq = (db_eq - Y_TICKS_DB[0]) / (Y_TICKS_DB[-1] - Y_TICKS_DB[0])

            height_orig = rel_orig * (CANVAS_HEIGHT - BOTTOM_MARGIN - 10)
            height_eq = rel_eq * (CANVAS_HEIGHT - BOTTOM_MARGIN - 10)

            x0 = LEFT_MARGIN + i * (BAR_WIDTH + BAR_SPACING)
            x1 = x0 + BAR_WIDTH

            # Blue bar (original)
            canvas.coords(bars_orig[i], x0 + 4, CANVAS_HEIGHT - BOTTOM_MARGIN, x1 - 4, CANVAS_HEIGHT - BOTTOM_MARGIN - height_orig)

            # White part of equalized bar: minimum of eq and orig heights
            white_height = min(height_eq, height_orig)
            canvas.coords(bars_eq[i], x0, CANVAS_HEIGHT - BOTTOM_MARGIN, x1, CANVAS_HEIGHT - BOTTOM_MARGIN - white_height)

            # Red part of equalized bar: excess over original (if any)
            red_height = max(0, height_eq - height_orig)
            canvas.coords(bars_red[i], x0, CANVAS_HEIGHT - BOTTOM_MARGIN - white_height, x1, CANVAS_HEIGHT - BOTTOM_MARGIN - white_height - red_height)

# Load MP3
def load_mp3():
    global imported_audio, audio_pos
    file_path = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3")])
    if file_path:
        print("Importando:", file_path)
        audio = AudioSegment.from_file(file_path).set_channels(1).set_frame_rate(FS)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
        imported_audio = samples
        audio_pos = 0

# GUI setup
tk.Label(root, text="Ganhos (dB)").pack()
frame = tk.Frame(root)
frame.pack()
for i, freq in enumerate(BANDS):
    ttk.Label(frame, text=f"{freq} Hz").grid(row=0, column=i)
    scale = tk.Scale(frame, from_=24, to=-24, variable=gains_dB[i])
    scale.grid(row=1, column=i)

# Canvas setup
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 200
BAR_WIDTH = 25
BAR_SPACING = 10
BOTTOM_MARGIN = 30
LEFT_MARGIN = 40

canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#005555")
canvas.pack()

bars_eq = []
bars_orig = []
bars_red = []

for i in range(10):
    x0 = LEFT_MARGIN + i * (BAR_WIDTH + BAR_SPACING)
    x1 = x0 + BAR_WIDTH

    bars_orig.append(canvas.create_rectangle(x0 + 4, CANVAS_HEIGHT - BOTTOM_MARGIN, x1 - 4, CANVAS_HEIGHT - BOTTOM_MARGIN, fill='skyblue', width=0))
    bars_eq.append(canvas.create_rectangle(x0, CANVAS_HEIGHT - BOTTOM_MARGIN, x1, CANVAS_HEIGHT - BOTTOM_MARGIN, fill='white', width=0))
    bars_red.append(canvas.create_rectangle(x0, CANVAS_HEIGHT - BOTTOM_MARGIN, x1, CANVAS_HEIGHT - BOTTOM_MARGIN, fill='red', width=0))

# Draw axes
canvas.create_line(LEFT_MARGIN, 10, LEFT_MARGIN, CANVAS_HEIGHT - BOTTOM_MARGIN, fill='white')  # Y axis
canvas.create_line(LEFT_MARGIN, CANVAS_HEIGHT - BOTTOM_MARGIN, CANVAS_WIDTH - 10, CANVAS_HEIGHT - BOTTOM_MARGIN, fill='white')  # X axis

# Y-axis ticks and labels in dB
Y_TICKS_DB = list(range(-60, 61, 12))  # Uniform spacing for nice visual
for db in Y_TICKS_DB:
    rel = (db - Y_TICKS_DB[0]) / (Y_TICKS_DB[-1] - Y_TICKS_DB[0])
    y = CANVAS_HEIGHT - BOTTOM_MARGIN - rel * (CANVAS_HEIGHT - BOTTOM_MARGIN - 10)
    canvas.create_line(LEFT_MARGIN - 5, y, LEFT_MARGIN, y, fill='white')
    canvas.create_text(LEFT_MARGIN - 25, y, text=f"{db} dB", fill='white', font=("Arial", 8))

# X-axis frequency labels
bands_labels = ["20", "63", "200", "630", "2k", "6k", "10k", "16k", "22k", "22k+"]
for i in range(10):
    x = LEFT_MARGIN + i * (BAR_WIDTH + BAR_SPACING) + BAR_WIDTH / 2
    canvas.create_text(x, CANVAS_HEIGHT - BOTTOM_MARGIN + 12, text=bands_labels[i], fill='white', font=("Arial", 8))

# MP3 Button
btn = tk.Button(root, text="Importar MP3", command=load_mp3)
btn.pack(pady=10)

# Start visualizer thread and audio stream
threading.Thread(target=spectrum_visualizer, args=(canvas, bars_eq, bars_red), daemon=True).start()
stream = sd.OutputStream(samplerate=FS, blocksize=N, dtype='float32', channels=1, callback=audio_callback)
stream.start()
root.mainloop()
stream.stop()

# Plot filters magnitude response (optional)
for i, filt in enumerate(filters):
    w = np.linspace(0, np.pi, 512)
    H = np.abs(np.fft.fft(filt, 1024))[:512]
    freqs = w * FS / (2 * np.pi)
    plt.plot(freqs, 20 * np.log10(H + 1e-10), label=f"Filtro {i+1} centrado em {BANDS[i]} Hz")

plt.title("Perfil dos filtros")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid()
plt.show()
