# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 09:49:19 2025

@author: msedo
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

from pathlib import Path


def load_inomed_single_channel_export(filepath):
    """
    Loads an inomed single-channel export where:
    - columns are individual signals / trials / measurements
    - rows are samples
    - each measurement time corresponds to one signal column

    Returns
    -------
    signals : np.ndarray
        Shape: n_signals x n_samples
        For your file: 213 x 4000

    measure_times : list[str]
        One time string per signal.
        measure_times[i] corresponds to signals[i, :]

    sampling_rate : int
        Sampling frequency in Hz.

    channel_name : str
        Example: W1-Fz
    """

    filepath = Path(filepath)

    with open(filepath, "r", encoding="latin-1") as f:
        content = f.read()

    # Sampling rate
    sr_match = re.search(r"Sampling rate:\s*(\d+)\s*Hz", content)
    if sr_match is None:
        raise ValueError("Could not find sampling rate.")

    sampling_rate = int(sr_match.group(1))

    # Channel name: Electrode 1 = W1, Electrode 2 = Fz
    e1_match = re.search(r"Electrode\s*1\s*=\s*(\S+)", content)
    e2_match = re.search(r"Electrode\s*2\s*=\s*(\S+)", content)

    if e1_match and e2_match:
        channel_name = f"{e1_match.group(1)}-{e2_match.group(1)}"
    else:
        channel_name = filepath.stem

    # Extract measurement times
    measure_block_match = re.search(
        r"Measure times:(.*?)Data:",
        content,
        flags=re.DOTALL
    )

    if measure_block_match is None:
        raise ValueError("Could not find Measure times block.")

    measure_block = measure_block_match.group(1)

    measure_times = re.findall(
        r"\b\d{1,2}:\d{2}:\d{2}\b",
        measure_block
    )

    # Extract data block
    data_match = re.search(
        r"Data:\s*(.*)$",
        content,
        flags=re.DOTALL
    )

    if data_match is None:
        raise ValueError("Could not find Data block.")

    data_block = data_match.group(1)

    number_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

    rows = []
    sample_indices = []

    for line in data_block.splitlines():
        line = line.strip()

        if ":" not in line:
            continue

        left, right = line.split(":", 1)

        # Only read lines like:
        # 1 : value value value ...
        # 2 : value value value ...
        if not re.match(r"^\d+$", left.strip()):
            continue

        sample_idx = int(left.strip())
        values = [float(v) for v in re.findall(number_pattern, right)]

        if values:
            sample_indices.append(sample_idx)
            rows.append(values)

    if not rows:
        raise ValueError("No numeric data rows were found.")

    # Shape here is:
    # n_samples x n_signals
    data_samples_by_signals = np.array(rows, dtype=float)

    # Transpose so shape becomes:
    # n_signals x n_samples
    signals = data_samples_by_signals.T

    if len(measure_times) != signals.shape[0]:
        print(
            f"Warning: found {len(measure_times)} measure times "
            f"but {signals.shape[0]} signals."
        )

    return signals, measure_times, sampling_rate, channel_name


def plot_overlapping_signals(
    signals,
    sampling_rate,
    measure_times=None,
    channel_name="W1-Fz",
    alpha=0.25,
    linewidth=0.8,
    baseline_correct=True
):
    """
    Plot all signals overlaid on the same axes.

    signals shape must be:
    n_signals x n_samples
    """

    signals = np.asarray(signals, dtype=float)

    if baseline_correct:
        # Remove each signal's own mean so they overlap better visually
        signals_to_plot = signals - np.mean(signals, axis=1, keepdims=True)
    else:
        signals_to_plot = signals.copy()

    n_signals, n_samples = signals_to_plot.shape

    time_ms = np.arange(n_samples) / sampling_rate * 1000

    plt.figure(figsize=(14, 6))

    for i in range(n_signals):
        plt.plot(
            time_ms,
            signals_to_plot[i, :],
            linewidth=linewidth,
            alpha=alpha
        )

    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (μV)")
    plt.title(f"{channel_name}: {n_signals} overlapping signals")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


filepath = r"C:\Users\marti\OneDrive\Documents\HSJD\CCEPs\Martin Garcia\Inomed M2\Data Export\CCEPs export W1 Fz.txt"

signals, measure_times, sr, channel_name = load_inomed_single_channel_export(filepath)

print("Channel:", channel_name)
print("Sampling rate:", sr, "Hz")
print("Signals shape:", signals.shape)
print("Number of measure times:", len(measure_times))

# Save the extracted signals and metadata
np.savez(
    "W1_Fz_213_signals.npz",
    signals=signals,
    measure_times=np.array(measure_times),
    sampling_rate=sr,
    channel_name=channel_name
)

first_half = signals[:signals.shape[0] // 4, :]
first_half_times = measure_times[:signals.shape[0] // 4]

# Plot all 213 signals overlapped
plot_overlapping_signals(
    first_half,
    sampling_rate=sr,
    measure_times=first_half_times,
    channel_name=channel_name,
    alpha=0.25,
    linewidth=0.8,
    baseline_correct=True
)







import re
import numpy as np
import matplotlib.pyplot as plt


def load_inomed_single_channel_export(filepath):
    with open(filepath, "r", encoding="latin-1") as f:
        content = f.read()

    # Sampling rate
    sr_match = re.search(r"Sampling rate:\s*(\d+)\s*Hz", content)
    if sr_match is None:
        raise ValueError("Could not find sampling rate.")
    sampling_rate = int(sr_match.group(1))

    # Channel name
    e1_match = re.search(r"Electrode\s*1\s*=\s*(\S+)", content)
    e2_match = re.search(r"Electrode\s*2\s*=\s*(\S+)", content)
    if e1_match and e2_match:
        channel_name = f"{e1_match.group(1)}-{e2_match.group(1)}"
    else:
        channel_name = "Unknown channel"

    # Measurement times
    measure_block_match = re.search(r"Measure times:(.*?)Data:", content, flags=re.DOTALL)
    if measure_block_match is None:
        raise ValueError("Could not find Measure times block.")
    measure_block = measure_block_match.group(1)
    measure_times = re.findall(r"\b\d{1,2}:\d{2}:\d{2}\b", measure_block)

    # Data block
    data_match = re.search(r"Data:\s*(.*)$", content, flags=re.DOTALL)
    if data_match is None:
        raise ValueError("Could not find Data block.")
    data_block = data_match.group(1)

    number_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    rows = []

    for line in data_block.splitlines():
        line = line.strip()
        if ":" not in line:
            continue

        left, right = line.split(":", 1)

        if not re.match(r"^\d+$", left.strip()):
            continue

        values = [float(v) for v in re.findall(number_pattern, right)]
        if values:
            rows.append(values)

    if not rows:
        raise ValueError("No numeric data rows were found.")

    # rows = samples x signals
    data_samples_by_signals = np.array(rows, dtype=float)

    # transpose -> signals x samples
    signals = data_samples_by_signals.T

    return signals, measure_times, sampling_rate, channel_name


def plot_overlapping_signals(
    signals,
    sampling_rate,
    channel_name="W1-Fz",
    title_suffix="",
    alpha=0.3,
    linewidth=0.8,
    baseline_correct=True
):
    signals = np.asarray(signals, dtype=float)

    if baseline_correct:
        signals_to_plot = signals - np.mean(signals, axis=1, keepdims=True)
    else:
        signals_to_plot = signals.copy()

    n_signals, n_samples = signals_to_plot.shape
    time_ms = np.arange(n_samples) / sampling_rate * 1000

    plt.figure(figsize=(14, 6))
    for i in range(n_signals):
        plt.plot(time_ms, signals_to_plot[i], alpha=alpha, linewidth=linewidth)

    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (μV)")
    plt.title(f"{channel_name}: {n_signals} overlapping signals {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def split_signals_by_initial_polarity(
    signals,
    sampling_rate,
    measure_times=None,
    start_ms=0.0,
    end_ms=3.0
):
    """
    Split signals according to the average value in an initial time window.

    Parameters
    ----------
    signals : array, shape (n_signals, n_samples)
    sampling_rate : float
    measure_times : list or None
    start_ms : float
        start of classification window in ms
    end_ms : float
        end of classification window in ms

    Returns
    -------
    positive_signals, negative_signals,
    positive_times, negative_times,
    positive_idx, negative_idx
    """

    signals = np.asarray(signals)

    start_idx = int(start_ms * sampling_rate / 1000)
    end_idx = int(end_ms * sampling_rate / 1000)

    if end_idx <= start_idx:
        raise ValueError("end_ms must be larger than start_ms")

    # Mean value in the chosen initial window
    initial_mean = np.mean(signals[:, start_idx:end_idx], axis=1)

    positive_idx = np.where(initial_mean >= 0)[0]
    negative_idx = np.where(initial_mean < 0)[0]

    positive_signals = signals[positive_idx]
    negative_signals = signals[negative_idx]

    if measure_times is not None:
        positive_times = [measure_times[i] for i in positive_idx]
        negative_times = [measure_times[i] for i in negative_idx]
    else:
        positive_times = None
        negative_times = None

    return (
        positive_signals,
        negative_signals,
        positive_times,
        negative_times,
        positive_idx,
        negative_idx
    )


# =========================
# MAIN
# =========================
filepath = r"C:\Users\marti\OneDrive\Documents\HSJD\CCEPs\Martin Garcia\Inomed M2\Data Export\CCEPs export W1 Fz.txt"

signals, measure_times, sr, channel_name = load_inomed_single_channel_export(filepath)

print("signals shape:", signals.shape)   # expected: (213, 4000)

(
    positive_signals,
    negative_signals,
    positive_times,
    negative_times,
    positive_idx,
    negative_idx
) = split_signals_by_initial_polarity(
    signals,
    sampling_rate=sr,
    measure_times=measure_times,
    start_ms=0.0,
    end_ms=3.0
)

print(f"Positive signals: {len(positive_signals)}")
print(f"Negative signals: {len(negative_signals)}")

print("Positive indices:", positive_idx + 1)  # +1 to match file numbering
print("Negative indices:", negative_idx + 1)

# Plot separately
plot_overlapping_signals(
    positive_signals,
    sampling_rate=sr,
    channel_name=channel_name,
    title_suffix="(positive initial polarity)",
    alpha=0.35,
    linewidth=0.8,
    baseline_correct=True
)

plot_overlapping_signals(
    negative_signals,
    sampling_rate=sr,
    channel_name=channel_name,
    title_suffix="(negative initial polarity)",
    alpha=0.35,
    linewidth=0.8,
    baseline_correct=True
)