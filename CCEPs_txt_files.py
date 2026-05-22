# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 09:49:19 2025

@author: msedo
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

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
    plt.ylim([-1000, 1000])
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
filepath = r"C:\Users\msedo\Documents\CCEPs\NEW patient data\Hector - 1403487 (MAQ 2)\CCEPs export 7 8.txt"

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
    positive_signals*1000,
    sampling_rate=sr,
    channel_name=channel_name,
    title_suffix="(positive initial polarity)",
    alpha=0.35,
    linewidth=0.8,
    baseline_correct=True
)

plot_overlapping_signals(
    negative_signals*1000,
    sampling_rate=sr,
    channel_name=channel_name,
    title_suffix="(negative initial polarity)",
    alpha=0.35,
    linewidth=0.8,
    baseline_correct=True
)


def add_signal_offsets(signals, offset_step=0.05, center_offsets=True):
    """
    Add a small vertical offset to each signal.

    Parameters
    ----------
    signals : np.ndarray
        Array with shape: n_signals x n_samples

    offset_step : float
        Vertical distance between consecutive signals, in the same units
        as the signal amplitude, usually μV.

    center_offsets : bool
        If True, offsets are centered around zero.
        If False, offsets start at zero and increase upward.

    Returns
    -------
    offset_signals : np.ndarray
        Signals with vertical offsets added.

    offsets : np.ndarray
        Offset value added to each signal.
    """

    signals = np.asarray(signals, dtype=float)

    n_signals = signals.shape[0]

    if center_offsets:
        offsets = (np.arange(n_signals) - (n_signals - 1) / 2) * offset_step
    else:
        offsets = np.arange(n_signals) * offset_step

    offset_signals = signals + offsets[:, np.newaxis]

    return offset_signals, offsets




first_100 = negative_signals[0:29, :]

first_100_offset, offsets = add_signal_offsets(
    first_100,
    offset_step=0.03,
    center_offsets=True
)

plot_overlapping_signals(
    first_100_offset*1000,
    sampling_rate=sr,
    channel_name=channel_name,
    title_suffix="first 100 signals with small offset",
    alpha=0.5,
    linewidth=0.8,
    baseline_correct=False
)



# ============== MEAN GROUPS OF 10 ======================


def plot_group_mean_std_envelopes(
    signals,
    sampling_rate,
    group_size=10,
    channel_name="W1-Fz",
    start_signal=0,
    end_signal=None,
    baseline_correct=True,
    baseline_window_ms=None,
    include_incomplete_group=True,
    separate_figures=True,
    alpha_envelope=0.25,
    linewidth_mean=2.0
):
    """
    Plot average and standard deviation envelope for groups of signals.

    Parameters
    ----------
    signals : np.ndarray
        Shape: n_signals x n_samples

    sampling_rate : float
        Sampling rate in Hz.

    group_size : int
        Number of signals per group. Default is 10.

    start_signal : int
        First signal index to use, Python-style.
        0 means signal 1.

    end_signal : int or None
        Last signal index, Python-style and exclusive.
        None means use all signals.

    baseline_correct : bool
        If True, subtracts baseline from each signal before averaging.

    baseline_window_ms : tuple or None
        Example: (0, 5) subtracts the mean from 0 to 5 ms.
        If None, subtracts the full-signal mean from each signal.

    include_incomplete_group : bool
        If True, includes the final group even if it has fewer than group_size signals.

    separate_figures : bool
        If True, each group is plotted in a different figure.
        If False, all group averages are plotted on the same figure.
    """

    signals = np.asarray(signals, dtype=float)

    if end_signal is None:
        end_signal = signals.shape[0]

    selected_signals = signals[start_signal:end_signal, :].copy()

    n_signals, n_samples = selected_signals.shape
    time_ms = np.arange(n_samples) / sampling_rate * 1000

    # Baseline correction
    if baseline_correct:
        if baseline_window_ms is None:
            baseline = np.mean(selected_signals, axis=1, keepdims=True)
        else:
            b_start_ms, b_end_ms = baseline_window_ms
            b_start = int(b_start_ms * sampling_rate / 1000)
            b_end = int(b_end_ms * sampling_rate / 1000)

            if b_end <= b_start:
                raise ValueError("baseline_window_ms must be something like (0, 5).")

            baseline = np.mean(selected_signals[:, b_start:b_end], axis=1, keepdims=True)

        selected_signals = selected_signals - baseline

    # Create group start indices
    group_starts = list(range(0, n_signals, group_size))

    if not include_incomplete_group:
        group_starts = [
            g for g in group_starts
            if g + group_size <= n_signals
        ]

    if not separate_figures:
        plt.figure(figsize=(14, 6))

    for group_number, group_start in enumerate(group_starts, start=1):
        group_end = min(group_start + group_size, n_signals)

        group = selected_signals[group_start:group_end, :]

        mean_signal = np.mean(group, axis=0)
        std_signal = np.std(group, axis=0)

        real_start_signal = start_signal + group_start + 1
        real_end_signal = start_signal + group_end

        if separate_figures:
            plt.figure(figsize=(14, 6))

        plt.plot(
            time_ms,
            mean_signal,
            linewidth=linewidth_mean,
            label=f"Mean signals {real_start_signal}-{real_end_signal}"
        )

        plt.fill_between(
            time_ms,
            mean_signal - std_signal,
            mean_signal + std_signal,
            alpha=alpha_envelope,
            label="±1 SD"
        )

        if separate_figures:
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (μV)")
            plt.title(
                f"{channel_name}: Mean ± SD, signals {real_start_signal}-{real_end_signal}"
            )
            plt.ylim(-600, 600)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

    if not separate_figures:
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (μV)")
        plt.title(f"{channel_name}: Group mean ± SD envelopes")
        plt.ylim(-500, 500)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

plot_group_mean_std_envelopes(
    positive_signals*1000,
    sampling_rate=sr,
    group_size=20,
    channel_name=channel_name,
    start_signal=10,
    end_signal=30,
    baseline_correct=True,
    baseline_window_ms=None,
    separate_figures=True
)

#======================= SAVE SELECTED SIGNALS =====================


def save_signal_selection(
    signals,
    measure_times,
    sampling_rate,
    channel_name,
    selected_indices,
    output_folder,
    output_name="selected_signals",
    save_csv=True,
    save_npz=True,
    one_based_indices=False
):
    """
    Save a selected group of signals to your computer.

    Parameters
    ----------
    signals : np.ndarray
        Shape: n_signals x n_samples

    measure_times : list
        One measurement time per signal.

    sampling_rate : float
        Sampling rate in Hz.

    channel_name : str
        Example: W1-Fz

    selected_indices : list, array, or np.ndarray
        Indices of signals to save.

    output_folder : str
        Folder where files will be saved.

    output_name : str
        Base name of the saved files.

    save_csv : bool
        If True, saves a CSV file.

    save_npz : bool
        If True, saves a compressed NumPy file.

    one_based_indices : bool
        If True, selected_indices are interpreted as file-style numbers:
        1, 2, 3, ...
        If False, selected_indices are Python-style:
        0, 1, 2, ...
    """

    signals = np.asarray(signals)
    selected_indices = np.asarray(selected_indices)

    if one_based_indices:
        selected_indices = selected_indices - 1

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    selected_signals = signals[selected_indices, :]
    selected_times = [measure_times[i] for i in selected_indices]

    n_selected, n_samples = selected_signals.shape
    time_ms = np.arange(n_samples) / sampling_rate * 1000

    saved_files = []

    if save_npz:
        npz_path = output_folder / f"{output_name}.npz"

        np.savez_compressed(
            npz_path,
            signals=selected_signals,
            selected_indices=selected_indices,
            selected_signal_numbers=selected_indices + 1,
            measure_times=np.array(selected_times, dtype=object),
            sampling_rate=sampling_rate,
            channel_name=channel_name,
            time_ms=time_ms
        )

        saved_files.append(npz_path)

    if save_csv:
        csv_path = output_folder / f"{output_name}.csv"

        df = pd.DataFrame()
        df["time_ms"] = time_ms

        for k, signal_idx in enumerate(selected_indices):
            signal_number = signal_idx + 1
            measure_time = measure_times[signal_idx]
            column_name = f"signal_{signal_number}_{measure_time}"
            df[column_name] = selected_signals[k, :]

        df.to_csv(csv_path, index=False)

        saved_files.append(csv_path)

    print("Saved files:")
    for file in saved_files:
        print(file)

    return selected_signals, selected_times, saved_files



selected_signals, selected_times, saved_files = save_signal_selection(
    signals=positive_signals,
    measure_times=measure_times,
    sampling_rate=sr,
    channel_name=channel_name,
    selected_indices=np.arange(3, 26),  # file-style numbering
    output_folder=r"C:\Users\msedo\Documents\CCEPs\Saved selections",
    output_name="6_7_signals_10_to_30",
    one_based_indices=True
)
