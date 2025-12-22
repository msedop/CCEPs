# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 10:11:12 2025

@author: msedo
"""

import numpy as np
import matplotlib.pyplot as plt
import re

# Read file
with open(r'C:\Users\msedo\Documents\CCEPs\Martin Garcia\Inomed M2\Data Export\CCEPs export W3 Fz.txt', encoding='latin-1') as f:
    content = f.read()

# Extract metadata
metadata = {}
lines = content.split('\n')

for line in lines[: 50]:  
    line = line.strip()
    
    if '***' in line or 'Raw data' in line:  
        break
    
    if '=' in line and not line.startswith('ASCII'):
        parts = line. split('=', 1)
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip()
            metadata[key] = value

# Extract patient name
patient_match = re.search(r'^\s*([A-Z]+\s+[A-Z]+,\s+[A-Z]+)\s*$', content, re. MULTILINE)
if patient_match:
    metadata['Patient'] = patient_match.group(1)

# Extract measurement info
measurement_match = re.search(r'(\w+\s+\w+\s+\w+)\s+(\d{2}/\d{2}/\d{4})\s*/\s*(\d{2}:\d{2}:\d{2})', content)
if measurement_match: 
    metadata['Measurement Type'] = measurement_match.group(1)
    metadata['Date'] = measurement_match.group(2)
    metadata['Time'] = measurement_match.group(3)

# Extract sampling rate
rate_match = re.search(r'Sampling rate:\s*(\d+)\s*Hz', content)
sampling_rate = int(rate_match.group(1)) if rate_match else None
metadata['Sampling Rate'] = f"{sampling_rate} Hz" if sampling_rate else None

# Extract measure times (recording time for each of the 213 signals)
times_match = re.search(r'Measure times:(.*?)Data:', content, re.DOTALL)
times = re.findall(r'\d{1,2}:\d{2}:\d{2}', times_match.group(1))

# Extract data
data_match = re.search(r'Data:\s*\n(.*?)$', content, re.DOTALL)
data_lines = data_match.group(1).strip().split('\n')

data_points = []
for line in data_lines:
    line = re.sub(r'^\s*\d+\s*:', '', line.strip())
    values = [float(v) for v in re.findall(r'-?\d+\. ?\d*', line)]
    if values:
        data_points.append(values)

# Convert to numpy array and transpose
data_points = np.array(data_points)
signals = -data_points.T

# Create time axis in milliseconds
num_samples = signals.shape[1]
time_ms = np.arange(num_samples) * (1000 / sampling_rate)  # Convert to milliseconds

# Print info
print("=== METADATA ===")
for key, value in metadata.items():
    print(f"{key}: {value}")

print(f"\n=== DATA INFO ===")
print(f"Total number of signals:  {signals.shape[0]}")
print(f"Data points per signal: {signals.shape[1]}")
print(f"Duration per signal: {time_ms[-1]:.2f} ms")
print(f"Number of measurement times:  {len(times)}")
print(f"First signal time: {times[0]}")
print(f"Last signal time: {times[-1]}")

# Filter signals by time range (13:28 to 13:40)
start_time = "13:28:00"
end_time = "13:40:00"

selected_indices = []
selected_times = []

for i, time in enumerate(times):
    if start_time <= time <= end_time:
        selected_indices.append(i)
        selected_times.append(time)

selected_signals = signals[selected_indices, :]

print(f"\n=== TIME FILTER INFO ===")
print(f"Time range: {start_time} to {end_time}")
print(f"Number of signals in range: {len(selected_indices)}")
print(f"Selected signal indices: {selected_indices}")
print(f"Selected times: {selected_times}")

# Plot first 10 signals from the selected time range - overlapping
num_to_plot = min(17, len(selected_indices))

plt.figure(figsize=(16, 10))

for i in range(num_to_plot):
    # Plot the signal without offset
    plt.plot(time_ms, selected_signals[i, :], 
            linewidth=1, 
            alpha=0.7,
            label=f'{selected_times[i]}')

plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude (μV)', fontsize=12)
plt.title(f'EEG Signals Overlapped - Time Range {start_time} to {end_time}\nFirst {num_to_plot} Signals', 
         fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=9)
plt.xlim(time_ms[0], time_ms[-1])
plt.ylim(-0.3, 0.3)
plt.tight_layout()
plt.show()



# Calculate average signal and standard deviation
average_signal = np.mean(selected_signals, axis=0)
std_signal = np.std(selected_signals, axis=0)

print(f"\n=== AVERAGE SIGNAL INFO ===")
print(f"Number of signals averaged: {selected_signals.shape[0]}")
print(f"Average signal shape: {average_signal.shape}")
print(f"Average signal mean: {np.mean(average_signal):.4f} μV")
print(f"Average signal std:  {np.std(average_signal):.4f} μV")
print(f"Average signal min:  {np.min(average_signal):.4f} μV")
print(f"Average signal max: {np.max(average_signal):.4f} μV")

# Plot average signal with standard deviation envelope
plt.figure(figsize=(16, 8))

# Plot standard deviation envelope
plt.fill_between(time_ms, 
                 average_signal - std_signal, 
                 average_signal + std_signal,
                 alpha=0.3, 
                 color='lightblue',
                 label='± SD')

# Plot average signal
plt.plot(time_ms, average_signal, 
        linewidth=2, 
        color='darkblue',
        label=f'Average of {len(selected_indices)} signals')

# Add zero line
plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (0 μV)')

plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude (μV)', fontsize=12)
plt.title(f'Average EEG Signal with SD Envelope - Time Range {start_time} to {end_time}\nAveraged across {len(selected_indices)} signals', 
         fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=10)
plt.ylim(-0.3, 0.3)

plt.tight_layout()
plt.show()