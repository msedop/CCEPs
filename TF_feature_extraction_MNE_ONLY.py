# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:26:19 2024

@author: Martina
"""

globals().clear()

from inomed.inoPatientData import *
from inomed.readEDF import *
import os
import glob

import matplotlib.pyplot as plt
import mplcursors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#from scipy.signal import morlet, cwt
from scipy.signal import find_peaks
from scipy.signal import decimate
from tensorpac.utils import ITC

import pywt
import mne
from mne.time_frequency import morlet, tfr

# ---------------- Importing files from specified folder ----------------------

plt.close('all')

plt.rcParams.update({
    'axes.titlesize': 18,      # Title size
    'axes.labelsize': 18,      # Axis label size (for both x and y axes)
    'xtick.labelsize': 16,     # X-tick label size
    'ytick.labelsize': 16,     # Y-tick label size
    'font.size': 16            # General font size (this affects legends, annotations, etc.)
})

def load_eeg_signals(folder):
    """
    Loads EEG signals from EDF files in a given folder.
    
    Parameters:
    folder: str
        The path to the folder containing the EDF files.
    
    Returns:
    signals: list of np.ndarray
        A list of signals, where each signal corresponds to data from an EDF file.
    fs: float
        The sampling frequency of the signals.
    """
    # Get a list of all EDF files in the folder
    files = glob.glob(os.path.join(folder, '*.edf'))

    # Sort the files by modification time in ascending order
    files.sort(key=os.path.getmtime)

    # Initialize an empty list to store signals
    signals = []

    # Iterate through each file and extract the signal
    for i in range(len(files)):
        # Load EDF file
        ipd = readEDF(file=files[i])

        # Extract metadata and signal data
        info = ipd[0]
        data = ipd[1]

        # Sampling frequency for each channel
        num_samples = info['nSamples'][0]
        fs = num_samples / info['durationRecords']

        # Store the first signal in the list (assuming one signal per file)
        signals.append(data[0] * 1000)  # Multiplying by 1000 to convert units

    return signals, fs


def extract_segment(signals, fs, start_time=0.0, end_time=None):
    """
    Extracts a segment from start_time to end_time seconds and returns the corresponding time vector.
    
    Parameters:
    signals: list of np.ndarray
        List of EEG signals.
    fs: float
        Sampling frequency of the signals.
    start_time: float
        Start of the segment to extract in seconds.
    end_time: float or None
        End of the segment to extract in seconds. If None, extracts until the end of the signal.
    
    Returns:
    tuple:
        - List of np.ndarray: List of extracted signal segments.
        - np.ndarray: Corresponding time vector for the extracted segments.
    """
    # Convert times to sample indices
    start_index = int(start_time * fs)
    end_index = int(end_time * fs) if end_time is not None else len(signals[0])
    
    # Extract the segment for each signal
    extracted_segments = []
    for signal in signals:
        if end_index > len(signal):
            end_index = len(signal)
        extracted_segments.append(signal[start_index:end_index])
    
    # Generate the time vector for the extracted segment
    segment_length = end_index - start_index
    time_vector = np.arange(start_index, end_index) / fs
    
    return extracted_segments, time_vector


def decimate_signals(signals, original_fs, target_fs):
    """
    Decimate the EEG signals to reduce the sampling frequency.
    
    Parameters:
    signals: list of np.ndarray
        The list of EEG signals to be decimated.
    original_fs: float
        The original sampling frequency.
    target_fs: float
        The desired (lower) sampling frequency.
    
    Returns:
    decimated_signals: list of np.ndarray
        The list of decimated EEG signals.
    """
    decimation_factor = int(original_fs / target_fs)
    decimated_signals = [decimate(signal, decimation_factor, zero_phase=True) for signal in signals]
    return decimated_signals


def plot_all_signals(original_signals, decimated_signals, original_fs, decimated_fs):
    """
    Plot all original signals on one subplot and all decimated signals on another subplot.
    
    Parameters:
    original_signals: list of np.ndarray
        List of original signals to be plotted.
    decimated_signals: list of np.ndarray
        List of decimated signals to be plotted.
    original_fs: float
        Original sampling frequency of the signals.
    decimated_fs: float
        Decimated sampling frequency of the signals.
    """
    # Calculate the time axis for the first original and decimated signal
    original_time = np.arange(len(original_signals[0])) / original_fs
    decimated_time = np.arange(len(decimated_signals[0])) / decimated_fs
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)
    
    # Plot all original signals on the first subplot
    for signal in original_signals:
        axs[0].plot(original_time, signal, alpha=0.7)  # Use alpha for transparency to avoid overlap issues
    axs[0].set_title('Original Signals')
    axs[0].set_ylabel('Amplitude')
    
    # Plot all decimated signals on the second subplot
    for signal in decimated_signals:
        axs[1].plot(decimated_time, signal, alpha=0.7)  # Use alpha for transparency to avoid overlap issues
    axs[1].set_title('Decimated Signals')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
    

def compute_itc(signals, fs, f_min, f_max, step, cycle):
    """
    Computes Inter-Trial Coherence (ITC) using the tensorpac ITC function.
    
    Parameters:
    signals: list of np.ndarray
        List of EEG signal segments.
    fs: float
        Sampling frequency of the signals.
    f_min: float
        Minimum frequency for phase calculation.
    f_max: float
        Maximum frequency for phase calculation.
    edges: int
        Number of points to remove due to edge effects.
    cycle: int
        Number of cycles to use to extract the phase.
    
    Returns:
    ITC object
        ITC values for each frequency.
    """
    # Convert list of signals to a 2D numpy array
    signals_array = np.array(signals)

    # Define the frequency range and step
    freqs = np.arange(f_min, f_max + step, step)
    
    # compute ITC for phases between [f_min, f_max]Hz with frequency steps
    itc = ITC(signals_array, fs, f_pha=freqs, dcomplex='wavelet', cycle=cycle, n_jobs=1)
    
    # Access ITC values
    itc_values = itc.itc
    
    # Plot the ITC
    itc.plot()
        
    return itc

def find_top_frequencies(itc, time, top_n=5, time_window=0.5):
    """
    Finds the top N frequencies with the highest ITC values and their corresponding time points.
    
    Parameters:
    itc: ITC object
        The ITC object containing the computed ITC values.
    time: np.ndarray
        Time vector in milliseconds.
    top_n: int
        Number of top frequencies to find.
    
    Returns:
    top_results: np.ndarray
        A 5x3 matrix where each row contains [frequency, time, ITC value].
    """
    itc_values = itc.itc
    freqs = itc.f_pha
    threshold=0.7
    
    # Find indices corresponding to the time window (first 0.15 seconds)
    time_indices = np.where(time <= time_window * 1000)[0]  # Convert seconds to ms

    # Compute the mean ITC value across the specified time window for each frequency
    mean_itc_values = np.mean(itc_values[:, time_indices], axis=1)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, mean_itc_values)
    plt.title('Mean ITC Values vs Frequencies')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mean ITC Value')
    plt.grid(True)
    plt.show()
    
    # Find the indices of the top N frequencies with the highest sustained ITC values
    top_indices = np.argsort(mean_itc_values)[-top_n:][::-1]
    
    # Initialize an empty array to store the results
    top_results = np.zeros((top_n, 4))  # Include 4 columns now for [freq, time, ITC value, threshold crossing time]
    
    for i, idx in enumerate(top_indices):
        # Frequency
        top_freq = freqs[idx][0]
        # Maximum ITC value within the time window
        top_itc_value = np.max(itc_values[idx, time_indices])
        # Time corresponding to the max ITC value within the time window
        top_time = time[time_indices][np.argmax(itc_values[idx, time_indices])]
        
        # Find the time when ITC value crosses the threshold for the first time
        crossing_indices = np.where(itc_values[idx, time_indices] < threshold)[0]
        if len(crossing_indices) > 0:
            threshold_crossing_time = time[time_indices][crossing_indices[0]]
        else:
            threshold_crossing_time = np.nan  # If no crossing occurs
        
        # Store the results in the matrix
        top_results[i, 0] = top_freq                    # Frequency
        top_results[i, 1] = top_time                    # Time of max ITC value
        top_results[i, 2] = top_itc_value               # Max ITC value
        top_results[i, 3] = threshold_crossing_time     # Threshold crossing time
        
        
    avg_itc_values = np.mean(itc_values[top_indices, :], axis=0)
    
    # Plot Average ITC Values Across Top Frequencies
    plt.figure(figsize=(12, 6))
    plt.plot(time, avg_itc_values, label='Average ITC Across Top Frequencies', color='blue')
    plt.xlabel('Time (ms)')
    plt.ylabel('Average ITC Value')
    plt.title('Average ITC over time across selected frequencies')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return top_results, top_indices


def plot_itc_against_time_multiple_frequencies(itc, time, freq_indices):
    """
    Plots ITC values against time for multiple frequency indices on the same plot.
    
    Parameters:
    itc: ITC object
        The ITC object containing the computed ITC values.
    time: np.ndarray
        Time vector.
    freq_indices: list or np.ndarray
        List or array of frequency indices to plot.
    """
    # Extract ITC values
    itc_values = itc.itc
    freqs = itc.f_pha
    
    # Plot ITC values for each specified frequency
    plt.figure(figsize=(12, 8))
    for idx in freq_indices:
        if idx >= itc_values.shape[0]:
            raise ValueError(f"Frequency index {idx} out of bounds.")
        
        itc_values_for_frequency = itc_values[idx, :]
        plt.plot(time, itc_values_for_frequency, label=f'Frequency: {freqs[idx]} Hz')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('ITC Value')
    plt.title('ITC values over time for selected frequencies')
    plt.legend()
    plt.grid(True)
    plt.show()



def apply_complex_morlet(decimated_signals, fs, t, w, top_results):
    """
    Applies the complex Morlet wavelet to compute the phase for each frequency and time point
    in top_results for every individual signal in decimated_signals. Plots the complex phase
    values on a unit circle for the top 5 frequencies.
    
    Parameters:
    decimated_signals: list of np.ndarray
        The list of decimated EEG signals.
    fs: float
        Sampling frequency of the signals.
    top_results: np.ndarray
        A 5x3 matrix where each row contains [frequency, time, ITC value].
    
    Returns:
    phases_mne: np.ndarray
        A Nx5x3 array containing the computed phase values for each trial, frequency, and time point.
    """
    num_trials = len(decimated_signals)
    num_freqs = top_results.shape[0]
    
    # Prepare tensors to store results
    phases_mne = np.zeros((num_trials, num_freqs, 3))
    phase_time_series = {freq: [] for freq in top_results[:, 0]}  # Dictionary to store phase time series
    
    # Extract frequencies and time points from top_results
    freqs = top_results[:, 0]  # Frequency column
    times = top_results[:, 1] / 1000  # Convert time from ms to seconds
    
    t = np.arange(0, 0.5, 1/fs)  # Time vector for signal
    t_centered = np.arange(-0.25, 0.25, 1/fs)  # Adjusted time vector for wavelet
    
    
    for j, freq in enumerate(freqs):
        all_phase_values = []
        time_values = []
        
        for i, signal in enumerate(decimated_signals):
            # Define the wavelet for the specific frequency
            wavelet_mne = morlet(sfreq=fs, freqs=[freq], n_cycles=w)[0]
            
            # Determine if padding is necessary
            M = len(wavelet_mne)
            signal_length = len(signal)
            
            if signal_length < M:
                # Compute padding length
                pad_length = M - signal_length
                # Apply symmetric zero-padding
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                padded_signal = np.pad(signal, (pad_left, pad_right), mode='constant')
            else:
                padded_signal = signal

            # Compute the CWT using MNE with the padded signal
            coefficients_mne = mne.time_frequency.tfr.cwt(padded_signal[np.newaxis, :], [wavelet_mne], use_fft=True, mode='same')[0, 0, :]
            
            # Find phase at a specific time point (e.g., t = 0.049s)
            index_of_interest = int(times[j] * fs)
            
            # Adjust index for padded signal
            if signal_length < M:
                # Index in the padded signal
                index_in_padded_signal = index_of_interest + pad_left
                # Coefficients of original signal only -> removing the padding
                og_sig_coeffs = coefficients_mne[pad_left:len(coefficients_mne)-pad_right]
                
            else:
                # Index in the original signal
                index_in_padded_signal = index_of_interest
                # Coefficients of original signal only -> removing the padding
                og_sig_coeffs = coefficients_mne

            # Ensure the index is within the bounds of the padded signal
            index_in_padded_signal = min(max(index_in_padded_signal, 0), len(coefficients_mne) - 1)

            phase_MNE_at_time = np.angle(coefficients_mne[index_in_padded_signal])
            
            # Store the results
            phases_mne[i, j, 0] = round(freq, 2)
            phases_mne[i, j, 1] = round(times[j] * 1000, 2)  # Store time back in ms, rounded
            phases_mne[i, j, 2] = round(phase_MNE_at_time, 2)
            
            # Collect phase values across time
            phase_values = np.unwrap(np.angle(og_sig_coeffs))
            phase_time_series[freq].append(phase_values)
            time_values = np.linspace(0.01, (len(phase_values) / fs)+0.01, len(phase_values))
            all_phase_values.append(phase_values)
        
        # Convert all phase values into a single array for plotting
        all_phase_values = np.concatenate(all_phase_values)
        time_values = np.tile(time_values, num_trials)  # Repeat time values for all trials
        '''
        # Plot phase values across time for each frequency
        plt.figure(figsize=(8, 4))
        plt.plot(time_values, all_phase_values, 'o-', alpha=0.6)
        plt.title(f'Phase Values Across Time for Frequency {freq} Hz')
        plt.xlabel('Time (s)')
        plt.ylabel('Phase (radians)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        '''
    # Convert lists to arrays
    for freq in phase_time_series:
        phase_time_series[freq] = np.array(phase_time_series[freq])  # Shape: (num_trials, num_time_points)
        
    # Dictionary to store times where std_phase < 50% of max variability
    time_ranges_below_threshold = {freq: [] for freq in top_results[:, 0]}
    
    # Dictionary to store the overall min/max time values
    total_time_range = {freq: [] for freq in top_results[:, 0]}

    # Compare phase values across trials
    for freq in phase_time_series:
        phase_data = phase_time_series[freq]
        num_time_points = phase_data.shape[1]
        
        # Compute mean and standard deviation across trials
        mean_phase = np.mean(phase_data, axis=0)
        std_phase = np.std(phase_data, axis=0)
        
        # Calculate the maximum variability
        max_variability = np.max(std_phase)
        
        # Determine the threshold (50% of max variability)
        threshold = 0.5 * max_variability
        
        # Find indices where std_phase is below the threshold
        below_threshold_indices = np.where(std_phase < threshold)[0]
        
        if len(below_threshold_indices) > 0:
            
            min_time = time_values[below_threshold_indices[0]]
            max_time = time_values[below_threshold_indices[-1]]
            total_time_range[freq] = (min_time, max_time)
            
            # Find continuous time ranges where std_phase is below the threshold
            ranges = []
            start_idx = below_threshold_indices[0]
            for i in range(1, len(below_threshold_indices)):
                if below_threshold_indices[i] > below_threshold_indices[i - 1] + 1:
                    # End of a range
                    ranges.append((start_idx, below_threshold_indices[i - 1]))
                    start_idx = below_threshold_indices[i]
            # Add the last range
            ranges.append((start_idx, below_threshold_indices[-1]))
            
            # Convert ranges to min and max time values
            for start, end in ranges:
                time_ranges_below_threshold[freq].append((time_values[start], time_values[end]))

        '''
        # Plot mean phase and variability
        plt.figure(figsize=(12, 6))
        time_values = np.linspace(0.01, (len(phase_values) / fs)+0.01, len(phase_values))
        
        plt.plot(time_values, mean_phase, label='Mean Phase', color='b')
        plt.fill_between(time_values, mean_phase - std_phase, mean_phase + std_phase, color='b', alpha=0.2, label='Phase Variability')
        
        plt.title(f'Phase Values Across Time for Frequency {freq} Hz')
        plt.xlabel('Time (s)')
        plt.ylabel('Phase (radians)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()        
        '''
        
        # Plot mean phase and variability
        plt.figure(figsize=(12, 6))
        time_values = np.linspace(0.01, (len(phase_values) / fs)+0.01, len(phase_values))
        
        plt.plot(time_values, std_phase, label='Phase Variability', color='b')
        
        plt.title(f'Phase Varibility Across Time for Frequency {freq} Hz')
        plt.xlabel('Time (s)')
        plt.ylabel('Phase Variability')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()       
        
    
    return phases_mne, time_ranges_below_threshold, total_time_range

def average_phase(phases_matrix):
    """
    Compute the average phase for each frequency across trials and plot all complex phases on the unit circle.

    Parameters:
    phases_matrix: np.ndarray
        A 3D numpy array with shape (N, M, 3) where N is the number of trials,
        M is the number of frequencies, and the last dimension contains
        [frequency, time, phase].

    Returns:
    avg_phases: np.ndarray
        A 2D numpy array with shape (M, 3) where M is the number of frequencies,
        and each row contains [frequency, time, average_phase].
    """
    N, M, _ = phases_matrix.shape
    avg_phases = np.zeros((M, 3))  # Prepare an array to store results

    # Prepare subplots for unit circle plots
    fig, axs = plt.subplots(1, M, figsize=(20, 5), subplot_kw=dict(projection='polar'))
    if M == 1:
        axs = [axs]  # Ensure axs is always a list for consistency

    for i in range(M):
        # Extract phases for the i-th frequency across all trials
        phases = phases_matrix[:, i, 2]

        # Convert phases to complex numbers
        complex_phases = np.exp(1j * phases)

        # Compute the mean of complex phases
        mean_complex_phase = np.mean(complex_phases)

        # Compute the average phase
        avg_phase = np.angle(mean_complex_phase)

        # Store the frequency, time, and average phase
        avg_phases[i] = [phases_matrix[0, i, 0], phases_matrix[0, i, 1], avg_phase]

        # Plot the complex phases on the unit circle
        ax = axs[i]
        ax.set_title(f'Frequency {phases_matrix[0, i, 0]} Hz')

        # Plot each phase on the unit circle as a line vector
        for phase in phases:
            ax.plot([0, np.angle(np.exp(1j * phase))], [0, 1], 'o-', color='blue', alpha=0.5)

        # Plot the mean phase as a dashed line
        ax.plot([0, np.angle(mean_complex_phase)], [0, 1], 'k--', label='Mean Phase')

        ax.set_ylim(0, 1.2)  # Adjust y-axis to fit all points
        ax.legend()

    plt.tight_layout()
    plt.show()

    return avg_phases

def circular_variance(phase_values):
    """
    Calculate the circular variance of phase values.
    """
    complex_phases = np.exp(1j * phase_values)
    mean_complex_phase = np.mean(complex_phases)
    mean_resultant_length = np.abs(mean_complex_phase)
    circular_variance = 1 - mean_resultant_length
    return circular_variance

def bootstrap_circular_variance(phase_values, num_bootstrap=1000):
    """
    Perform bootstrapping to estimate the distribution of circular variance.
    
    Parameters:
    phase_values: np.ndarray
        Array of phase values for a single channel.
    num_bootstrap: int
        Number of bootstrap samples to generate.
    
    Returns:
    bootstrapped_variances: np.ndarray
        Array of bootstrapped circular variances.
    """
    bootstrapped_variances = np.zeros(num_bootstrap)
    num_trials = len(phase_values)
    
    for i in range(num_bootstrap):
        # Resample with replacement
        resampled_phases = np.random.choice(phase_values, size=num_trials, replace=True)
        # Compute circular variance for the bootstrap sample
        bootstrapped_variances[i] = circular_variance(resampled_phases)
    
    return bootstrapped_variances

def calculate_phase_errors(phase_tensor, mean_phase_matrix, top_results, num_bootstrap=1000):
    """
    Calculate the error between phase values and mean phase values for each frequency and time point.
    Outputs a tensor with bootstrapped circular variance statistics across trials and ITC values.

    Parameters:
    phase_tensor: np.ndarray
        A tensor of shape (N, 5, 3) where N is the number of trials, 5 is the number of frequencies,
        and 3 corresponds to [frequency, time, phase].
    mean_phase_matrix: np.ndarray
        A matrix of shape (5, 3) where 5 is the number of frequencies, and 3 corresponds to [frequency, time, mean_phase].
    top_results: np.ndarray
        A matrix of shape (5, 3) where 5 is the number of frequencies, and 3 corresponds to [frequency, time, ITC].
    num_bootstrap: int
        Number of bootstrap samples to generate.

    Returns:
    variance_tensor: np.ndarray
        A tensor of shape (5, 6) containing the frequency, time, ITC values, mean resultant length, bootstrapped mean circular variance, and bootstrapped variance standard deviation.
    """
    num_trials, num_freqs, _ = phase_tensor.shape
    
    # Initialize the variance tensor with shape (num_freqs, 7)
    variance_tensor = np.zeros((num_freqs, 7))
    
    # Loop over each frequency
    for freq_idx in range(num_freqs):
        
        # Extract phase values for the current frequency
        phases = phase_tensor[:, freq_idx, 2]
        
        # Perform bootstrapping
        bootstrapped_variances = bootstrap_circular_variance(phases, num_bootstrap)
        
        # Compute statistics from bootstrapped results
        mean_bootstrapped_variance = np.mean(bootstrapped_variances)
        std_bootstrapped_variance = np.std(bootstrapped_variances)
        
        # Extract frequency and time
        freq = mean_phase_matrix[freq_idx, 0]  # Frequency
        time = mean_phase_matrix[freq_idx, 1]  # Time
        itc_value = top_results[freq_idx, 2]  # ITC value
        itc_crossing_time = top_results[freq_idx, 3]  # ITC value

        
        variance_tensor[freq_idx, 0] = freq  # Frequency
        variance_tensor[freq_idx, 1] = time  # Time
        variance_tensor[freq_idx, 2] = itc_value  # ITC value
        variance_tensor[freq_idx, 3] = itc_crossing_time
        variance_tensor[freq_idx, 4] = np.abs(np.mean(np.exp(1j * phases)))  # ITC value
        variance_tensor[freq_idx, 5] = mean_bootstrapped_variance
        variance_tensor[freq_idx, 6] = std_bootstrapped_variance   
        
    # Sort the tensor by circular variance (column index 3) in descending order
    sorted_indices = np.argsort(variance_tensor[:, 5])
    sorted_variance_tensor = variance_tensor[sorted_indices]
    
    return sorted_variance_tensor


def plot_avg_itc_across_frequencies(itc, time):
    """
    Plots the average ITC values across all frequencies against time.
    
    Parameters:
    itc: ITC object
        The ITC object containing the computed ITC values.
    time: np.ndarray
        Time vector.
    """
    # Extract ITC values
    itc_values = itc.itc
    
    # Calculate the average ITC across all frequencies at each time point
    avg_itc_values = np.mean(itc_values, axis=0)
    
    # Plot the average ITC values against time
    plt.figure(figsize=(12, 6))
    plt.plot(time, avg_itc_values, label='Average ITC Across All Frequencies', color='blue')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Average ITC Value')
    plt.title('Average ITC Over Time Across All Frequencies')
    plt.grid(True)
    plt.show()
    
def plot_itc_all_frequencies(itc, time):
    """
    Plots ITC values against time for all frequencies.
    
    Parameters:
    itc: ITC object
        The ITC object containing the computed ITC values.
    time: np.ndarray
        Time vector.
    """
    # Extract ITC values and corresponding frequencies
    itc_values = itc.itc
    freqs = itc.f_pha
    
    # Plot ITC values for all frequencies
    plt.figure(figsize=(12, 8))
    
    # Iterate through each frequency and plot its ITC values over time
    for idx in range(itc_values.shape[0]):
        itc_values_for_frequency = itc_values[idx, :]
        plt.plot(time, itc_values_for_frequency)
    
    plt.xlabel('Time (ms)')
    plt.ylabel('ITC Value')
    plt.title('ITC Values Over Time for All Frequencies')
    plt.grid(True)
    plt.show()



#--------------------------------------------------------------------------------------------

folder = r'C:\Users\msedo\Documents\CCEPs\NEW patient data\Oloriz - 1692187 (MAQ 4)\W3-W4'
signals, fs = load_eeg_signals(folder)

#--------------------------------------------------------------------------------------------
start_time=0.01
es, tes = extract_segment(signals, fs, start_time, end_time=None)
#--------------------------------------------------------------------------------------------
# Decimate the signals from 20000 Hz to 5000 Hz
target_fs = 20000
decimated_signals = decimate_signals(es, fs, target_fs)

#---------------------- Decimating and ITC computation --------------------------------------

# Parameters for ITC computation
f_min = 10
f_max = 100
step = 0.5
cycle = 5       # Number of cycles to use to extract the phase
times = (np.arange(start_time, (len(decimated_signals[0]) / target_fs)+start_time, 1 / target_fs))*1000

itc_full = compute_itc(decimated_signals, target_fs, f_min, f_max, step, cycle)

# Find the top 5 frequencies and respective times with the highest ITC values
top_results, top_indices = find_top_frequencies(itc_full,times, top_n=5)

# Plot the ITC values for the top frequencies
plot_itc_against_time_multiple_frequencies(itc_full, times, top_indices)

# Plot original signals vs decimated signals to make sure signal integrity is maintained after decimating
#plot_all_signals(es, decimated_signals, fs, target_fs)

#plot_itc_all_frequencies(itc_full, times)

    
#------------------------- Signal segmentation and Complex Morlet--------------------------

# Extract the first 0.25 seconds of each signal
signals_segment = extract_segment(decimated_signals, target_fs, start_time, end_time=0.25)

w = 5

# Example usage
phases_mne, times_below_threshold, total_time_range = apply_complex_morlet(decimated_signals, target_fs, times, w, top_results)

# Set NumPy print options to avoid scientific notation and show two decimal places
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

# Print the results
print("\n Phases MNE Tensor:")
print(phases_mne)

avg_phases_mne = average_phase(phases_mne)
print("\n Phases mean mne: ")
print(avg_phases_mne)

# Calculate the error between phase values and the corresponding mean
error_tensor = calculate_phase_errors(phases_mne, avg_phases_mne, top_results)

# Print the error tensor
print("\nError Tensor:")
print(error_tensor)

print("\n Times below threshold: ")
print(times_below_threshold)

print("\n Time ranges: ")
print(total_time_range)


# Export the min_error_matrix to an Excel file
# Define column names
column_names = ['Frequency', 'Time', 'ITC Value', 'ITC crossing Time', 'Mean Phase', 'Circular Variance', 'Circular Variance SD']

# Create a DataFrame
error_tensor_pd = pd.DataFrame(error_tensor, columns=column_names)
error_tensor_pd.to_excel(r'C:\Users\msedo\Documents\CCEPs\CCEP plots\P19\P19CHW3-W4_TF_ANALYSIS.xlsx', index=True)

print(error_tensor_pd)

# Export to Excel
#df_errors.to_excel(r'C:\Users\marti\OneDrive\Documents\UPC\Quart de carrera\8th Cuatrimestre\TFG\SJD\Data Recordings\PATIENT DATA\Patient 14\TF analysis\P14CH3-4_TFA_ITC_E.xlsx', index=True)
def plot_phases_on_unit_circle(phases_matrix):
    """
    Plots phases on the unit circle for each frequency and adds legends.
    
    Parameters:
    phases_matrix: np.ndarray
        A 3D numpy array with shape (N, M, 3) where N is the number of trials,
        M is the number of frequencies, and the last dimension contains
        [frequency, time, phase].
    """
    num_freqs = phases_matrix.shape[1]
    
    plt.figure(figsize=(12, 12))

    for i in range(num_freqs):
        # Extract phases for the i-th frequency across all trials
        phases = phases_matrix[:, i, 2]

        # Convert phases to complex numbers
        complex_phases = np.exp(1j * phases)
        
        # Extract real and imaginary parts
        real_parts = np.real(complex_phases)
        imag_parts = np.imag(complex_phases)

        # Plot all phases on the unit circle
        plt.subplot(num_freqs, 1, i + 1)
        plt.plot(real_parts, imag_parts, 'o', label=f'Frequency {phases_matrix[0, i, 0]} Hz', alpha=0.7)
        
        # Plot unit circle
        circle = plt.Circle((0, 0), 1, color='grey', fill=False, linestyle='--', linewidth=1)
        plt.gca().add_artist(circle)
        
        # Set aspect ratio to equal for unit circle
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Set plot limits
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        
        # Labels and legend
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title(f'Phase Distribution for Frequency {phases_matrix[0, i, 0]} Hz')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage
phases_matrix = np.random.rand(10, 5, 3)  # Replace with your actual phases_matrix
plot_phases_on_unit_circle(phases_matrix)