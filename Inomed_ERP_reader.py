# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:53:42 2024

@author: Martina

FOR EVERY PATIENT AND CHANNEL, CHANGE FOLDER DIRECTORY ON LINE 31 AND EXCEL FILE EXPORT NAME ON LINE 324

"""

import sys
sys.path.append(r"C:\Users\msedo\Documents\CCEPs\SJD\Nex Python Toolkit 3.12")

from inomed.inoPatientData import *
from inomed.readEDF import *

import matplotlib.pyplot as plt
import mplcursors

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import coherence

import os
import glob
from pprint import pprint

import seaborn as sns

# ==================== Función parser de metadatos señales inomed ======================

def parse_inomed_metadata(meta_list):
    raw = meta_list[0]
    
    # Separamos por carácter de split '\x14', eliminamos espacios en blanco y strings vacios
    parts = [p.strip() for p in raw.split('\x14') if p.strip()]
    
    # Convertimos a diccionario
    meta_dict = {}
    for item in parts:
        if ':' in item:
            key, val = item.split(':', 1) # separamos contenido de string por ":" y generamos dos elementos
            meta_dict[key.strip()] = val.strip() # Definimos el primer elemento como key y el segundo value,
                                                 # eliminando espacios en blanco que puedan haber en los strings
    
    return meta_dict

# =======================================================================================

# =======================================================================================
#                               Importación de archivos EDF
# =======================================================================================

plt.close('all')

# Directorio de carpeta que contiene los archivos EDF a analizar
folder = r'C:\Users\msedo\Documents\CCEPs\NEW patient data\Genis Gelador - 1981833 (MAQ 4)\W3-REF'

# Lista de todos los archivos EDF presentes en la carpeta
files = glob.glob(os.path.join(folder, '*.edf'))

# Ordenamos los archivos por fecha de modificación en orden ascendente
files.sort(key=os.path.getmtime)

# Creación de array con nombres de archivos
file_names = [os.path.basename(file) for file in files]

# Visualización de metadatos de primer archivo EDF 
meta_patient = readEDF(file=files[0])[0]
pprint(meta_patient) # pprint() mejora la visualización de los datos (multi-line display)

meta = parse_inomed_metadata(readEDF(file=files[0])[1][1])
pprint(meta)


# =======================================================================================
#                           Importación CCEPs máquina 2
# =======================================================================================

# file_path = r"C:\Users\msedo\Documents\CCEPs\Saved selections\Hector\6_7_signals_10_to_30.csv"

# df = pd.read_csv(file_path)

# time_ms = df["time_ms"].to_numpy()

# signal_columns = [col for col in df.columns if col.startswith("signal_")]

# signals = df[signal_columns].to_numpy().T

# signals_list = [signals[i, :].copy()*1000 for i in range(signals.shape[0])]

# signal_numbers = []
# measure_times = []

# for col in signal_columns:
#     match = re.match(r"signal_(\d+)_(.+)", col)
#     signal_numbers.append(int(match.group(1)))
#     measure_times.append(match.group(2))

# sampling_rate = 20000
# channel_name = "W3-Fz"

# print("Signals array shape:", signals.shape)
# print("Number of signals in list:", len(signals_list))
# print("Each signal shape:", signals_list[0].shape)
# print("Signal numbers:", signal_numbers)
# print("Measure times:", measure_times)


# f_signals = signals_list
# t_signals = time_ms



# plt.figure(figsize=(14, 6))

# for signal in signals_list:
#     plt.plot(time_ms, signal, alpha=0.4, linewidth=0.8)

# plt.xlabel("Time (ms)")
# plt.ylabel("Amplitude (μV)")
# plt.title(f"{channel_name}: loaded signals {signal_numbers[0]} to {signal_numbers[-1]}")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()


# =======================================================================================
#                               Visualización de señales
# =======================================================================================

# Inicializamos listas vacias para almacenar las señales
orig_signals = []
f_signals = []
t_signals = []

# Creación de subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

for i in np.array(range(0, len(file_names))):
    
    # cargamos archivo EDF
    ipd = readEDF(file=files[i])

    # Guardamos metadatos
    info = ipd[0]

    # Lista de numpy.ndarray con datos
    data = ipd[1]
    
    # Número de muestras en registro
    num_samples = ipd[0]['nSamples'][0]
    
    # Duración registro (seg)
    duration = ipd[0]['durationRecords']
    
    # Frecuencia de muestreo
    fs = (num_samples/duration)
    
    # Periodo de muestreo
    sample_range = range(num_samples)
    sample_array = np.array(sample_range)
    t = (sample_array / fs) * 1000 # Conversión a milisegundos 
    
    y = -data[0]*1000 # Conversión a uV
    
    orig_signals.append(y)

    #----------------------- Filtramos señal  ---------------------------------
    
    # Frecuencias de corte para los filtros
    low_cutoff = 10  # Frecuencia de corte pasa-altos en Hz
    high_cutoff = 1500.0  # Frecuencia de corte pasa-bajos en Hz
    
    # Creamos un filtro Butterworth
    order = 4  # Orden del filtro
    b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='band', fs=fs)
    
    # Aplicamos el filtro a la señal
    filtered_y = signal.filtfilt(b, a, y)
    
    f_signals.append(y)
    t_signals.append(t)
    
    # ------------------------------- Plots ----------------------------------
    
    # Plot señal original
    ax1.plot(t, y, label='Original Signal')
    # Plot señal filtrada
    ax2.plot(t, filtered_y, label='Filtered Signal')
    


ax1.set_ylabel("Amplitude [uV]")
ax1.set_title('Original Signal')
ax1.grid('minor')

ax2.set_ylabel("Amplitude [uV]")
ax2.set_xlabel("Time [ms]")
ax2.set_title('Filtered Signal')
ax2.grid('minor')

# Ajustamos el layout para evitar solapamiento
plt.tight_layout()

# Añadimos cursores interactivos
mplcursors.cursor(ax1.lines + ax2.lines, hover=True)

plt.show()


# =======================================================================================
#            Feature extraction (latencia + amplitud) componentes P1, N1, P2, N2
# =======================================================================================

# Inicializamos listas para almacenar medidas de latencia y amplitud
data = {
    'Signal': [],
    'P1_Latency': [], 'P1_Amplitude': [],
    'N1_Latency': [], 'N1 matsumoto': [],
    'P2_Latency': [], 'P2_Amplitude': [],
    'N2_Latency': [], 'N2_Amplitude': []
}


# Plot señales
plt.figure(figsize=(10, 6))

# Ploteamos todas las señales en la nueva figura
for idx in range(len(f_signals)):

    t = t_signals[idx]
    
    # Definición de ventanas temporales de cada componente
    mask_section1 = (t >= 10) & (t <= 15)   # P1: 'Little bump', primer pico positivo
    mask_section2 = (t >= 20) & (t <= 50)  # N1: 'Matsumoto N1', primer pico negativo
    mask_section3 = (t >= 50) & (t <= 100)  # P2: segundo pico positivo
    mask_section4 = (t >= 100) & (t <= 200) # N2: segundo pico negativo
    
    # Extraemos las ventanas de tiempo relevantes
    section1 = f_signals[idx][mask_section1]
    section2 = f_signals[idx][mask_section2]
    section3 = f_signals[idx][mask_section3]
    section4 = f_signals[idx][mask_section4]

    # Encontramos el máximo de la sección 1 (P1)
    max_value_section1 = np.max(section1)
    global_indices_section1 = np.where(mask_section1)[0]
    max_idx_local_section1 = np.argmax(section1)
    max_idx_section1 = global_indices_section1[max_idx_local_section1]

    # Encontramos el mínimo de la sección 2 (N1)
    min_value_section2 = np.min(section2)
    global_indices_section2 = np.where(mask_section2)[0]
    min_idx_local_section2 = np.argmin(section2)
    min_idx_section2 = global_indices_section2[min_idx_local_section2]

    # Encontramos el máximo de la sección 3 (P2)
    max_value_section3 = np.max(section3)
    global_indices_section3 = np.where(mask_section3)[0]
    max_idx_local_section3 = np.argmax(section3)
    max_idx_section3 = global_indices_section3[max_idx_local_section3]

    # Encontramos el mínimo de la sección 4 (N2)
    min_value_section4 = np.min(section4)
    global_indices_section4 = np.where(mask_section4)[0]
    min_idx_local_section4 = np.argmin(section4)
    min_idx_section4 = global_indices_section4[min_idx_local_section4]
    
    
    # Encontramos las coordenadas de la intersección entre la línea vertical que passa por N1 y la recta P1-P2
    x1, y1 = t[max_idx_section1], max_value_section1 # Coordenadas P1
    x2, y2 = t[max_idx_section3], max_value_section3 # Coordenadas P2
    x3, y3 = t[min_idx_section2], min_value_section2 # Coordenadas N1

    # Ecuación de la recta que atraviesa (x1, y1) y (x2, y2)
    m = (y2 - y1) / (x2 - x1)  # Pendiente de la recta
    c = y1 - m * x1            # Intercept de la recta

    # Punto de intersección (x3, y4)
    y4 = m * x3 + c

    # Cálculo de amplitud de Matsumoto (N1)
    n1_matsumoto = y4 - y3

    # Cálculo de amplitud N2
    n2_amp = max_value_section3 - min_value_section4

    
    # Añadimos valores de latencia y amplitud en las listas correspondientes
    data['Signal'].append(idx + 1)
    data['P1_Latency'].append(t[max_idx_section1])
    data['P1_Amplitude'].append(max_value_section1)
    data['N1_Latency'].append(t[min_idx_section2])
    data['N1 matsumoto'].append(n1_matsumoto)
    data['P2_Latency'].append(t[max_idx_section3])
    data['P2_Amplitude'].append(max_value_section3)
    data['N2_Latency'].append(t[min_idx_section4])
    data['N2_Amplitude'].append(n2_amp)
    
    
    plt.plot(t, f_signals[idx])
    
    # Plot de valores mínimos en la señal con etiquetas
    plt.scatter([t[mask_section2][np.argmin(f_signals[idx][mask_section2])], t[mask_section4][np.argmin(f_signals[idx][mask_section4])]],
                [min_value_section2, min_value_section4], color='red', zorder=2)

    # Plot de valor máximo P1 en la señal con etiqueta
    plt.scatter(t[mask_section1][np.argmax(f_signals[idx][mask_section1])], max_value_section1, color='green', zorder=2)

    # Plot de valor máximo P2 en la señal con etiqueta
    plt.scatter(t[max_idx_section3], max_value_section3, color='blue', zorder=2)

    
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [uV]')
plt.title('Original Signal with Maximum and Minimum Values in Sections')
plt.legend() 
   
plt.grid('minor')
plt.show()

# Cálculo de media
mean_data = {
    'Signal': ['Mean'],
    'P1_Latency': [np.mean(data['P1_Latency'])],
    'P1_Amplitude': [np.mean(data['P1_Amplitude'])],
    'N1_Latency': [np.mean(data['N1_Latency'])],
    'N1 matsumoto': [np.mean(data['N1 matsumoto'])],
    'P2_Latency': [np.mean(data['P2_Latency'])],
    'P2_Amplitude': [np.mean(data['P2_Amplitude'])],
    'N2_Latency': [np.mean(data['N2_Latency'])],
    'N2_Amplitude': [np.mean(data['N2_Amplitude'])]
}

# Cálculo de desviación estándar
std_data = {
    'Signal': ['STD'],
    'P1_Latency': [np.std(data['P1_Latency'])],
    'P1_Amplitude': [np.std(data['P1_Amplitude'])],
    'N1_Latency': [np.std(data['N1_Latency'])],
    'N1 matsumoto': [np.std(data['N1 matsumoto'])],
    'P2_Latency': [np.std(data['P2_Latency'])],
    'P2_Amplitude': [np.std(data['P2_Amplitude'])],
    'N2_Latency': [np.std(data['N2_Latency'])],
    'N2_Amplitude': [np.std(data['N2_Amplitude'])]
}

# =======================================================================================
#                  Averaged signal with Standard Deviation Envelope
# =======================================================================================
# ------------ Averaged signal with standard deviation envelope ---------------

# ---------------------------------------------------------------
#            ONLY FOR PATIENT 19 WITH DIFF SIGNAL LENGTHS
# ---------------------------------------------------------------

# sig_4000 = [-sig[:4000] for sig in f_signals]
# f_signals = sig_4000
# t_4000 = [t[:4000] for t in t_signals]
# t = t_4000[0]

# # Definición de ventanas temporales de cada componente
# mask_section1 = (t >= 12) & (t <= 27)   # P1: 'Little bump', primer pico positivo
# mask_section2 = (t >= 20) & (t <= 50)  # N1: 'Matsumoto N1', primer pico negativo
# mask_section3 = (t >= 50) & (t <= 110)  # P2: segundo pico positivo
# mask_section4 = (t >= 100) & (t <= 200) # N2: segundo pico negativo

# ------------------- !!!!!!!!!!!!!!!!!!!!!!  -------------------

# Calculate the mean signal and standard deviation
mean_signal = np.mean(f_signals, axis=0)
std_deviation = np.std(f_signals, axis=0)
N = len(f_signals)

# Extraemos las ventanas de tiempo relevantes
section1 = mean_signal[mask_section1]
section2 = mean_signal[mask_section2]
section3 = mean_signal[mask_section3]
section4 = mean_signal[mask_section4]

# Encontramos el máximo de la sección 1 (P1)
max_value_section1 = np.max(section1)
global_indices_section1 = np.where(mask_section1)[0]
max_idx_local_section1 = np.argmax(section1)
max_idx_section1 = global_indices_section1[max_idx_local_section1]

# Encontramos el mínimo de la sección 2 (N1)
min_value_section2 = np.min(section2)
global_indices_section2 = np.where(mask_section2)[0]
min_idx_local_section2 = np.argmin(section2)
min_idx_section2 = global_indices_section2[min_idx_local_section2]

# Encontramos el máximo de la sección 3 (P2)
max_value_section3 = np.max(section3)
global_indices_section3 = np.where(mask_section3)[0]
max_idx_local_section3 = np.argmax(section3)
max_idx_section3 = global_indices_section3[max_idx_local_section3]

# Encontramos el mínimo de la sección 4 (N2)
min_value_section4 = np.min(section4)
global_indices_section4 = np.where(mask_section4)[0]
min_idx_local_section4 = np.argmin(section4)
min_idx_section4 = global_indices_section4[min_idx_local_section4]

# Encontramos las coordenadas de la intersección entre la línea vertical que passa por N1 y la recta P1-P2
x1, y1 = t[max_idx_section1], max_value_section1 # Coordenadas P1
x2, y2 = t[max_idx_section3], max_value_section3 # Coordenadas P2
x3, y3 = t[min_idx_section2], min_value_section2 # Coordenadas N1

# Ecuación de la recta que atraviesa (x1, y1) y (x2, y2)
m = (y2 - y1) / (x2 - x1)  # Pendiente de la recta
c = y1 - m * x1            # Intercept de la recta

# Punto de intersección (x3, y4)
y4 = m * x3 + c

# Cálculo de amplitud de Matsumoto (N1)
n1_matsumoto_avg = y4 - y3

# Cálculo de amplitud N2
n2_amp_avg = max_value_section3 - min_value_section4

# Append N1, P1, and N2 values to the data table
data['Signal'].append('Averaged')
data['P1_Latency'].append(t[max_idx_section1])
data['P1_Amplitude'].append(max_value_section1)
data['N1_Latency'].append(t[min_idx_section2])
data['N1 matsumoto'].append(n1_matsumoto_avg)
data['P2_Latency'].append(t[max_idx_section3])
data['P2_Amplitude'].append(max_value_section3)
data['N2_Latency'].append(t[min_idx_section4])
data['N2_Amplitude'].append(n2_amp_avg)

# Plot the mean signal with standard deviation
plt.figure(figsize=(10, 6))
plt.plot(t, mean_signal, label='Averaged Signal')
plt.fill_between(t, mean_signal - std_deviation, mean_signal + std_deviation, color='gray', alpha=0.2, label='Standard Deviation')

# Plot the maximum and minimum values on the mean signal
plt.scatter(t[max_idx_section1], max_value_section1, color='red', label='P1')
plt.scatter(t[min_idx_section2], min_value_section2, color='green', label='N1')
plt.scatter(t[max_idx_section3], max_value_section3, color='blue', label='P2')
plt.scatter(t[min_idx_section4], min_value_section4, color='orange', label='N2')

# Add text labels beneath the legend
plt.text(75, 40, f'P1: lat.={t[max_idx_section1]:.2f} ms, amp.={max_value_section1:.2f} uV', ha='left')
plt.text(75, 50, f'N1: lat.={t[min_idx_section2]:.2f} ms, amp.={n1_matsumoto_avg:.2f} uV', ha='left')
plt.text(75, 60, f'P2: lat.={t[max_idx_section3]:.2f} ms, amp.={max_value_section3:.2f} uV', ha='left')
plt.text(75, 70, f'N2: lat.={t[min_idx_section4]:.2f} ms, amp.={n2_amp_avg:.2f} uV', ha='left')


# ===================== Intersection lines plot =============================

# ------------------------ Plotting P1-P2 intersection with N1 ----------------

# --- P1-P2 segment as dashed line ---
x_line = np.linspace(min(x1, x2), max(x1, x2), 200)
y_line = m * x_line + c
plt.plot(x_line, y_line, color='grey', linestyle='--', linewidth=2, label='P1-P2 line')

# --- Intersection on the P1-P2 line (x3, y4) ---
plt.scatter(x3, y4, color='grey', zorder=6)

# --- Ensure N1 point is visible (re-plot if needed) ---
plt.scatter(x3, y3, color='green', zorder=7)  # N1 already plotted earlier, this ensures it's on top

# --- Vertical line between intersection and N1 ---
ymin_v, ymax_v = min(y3, y4), max(y3, y4)
plt.vlines(x3, ymin=ymin_v, ymax=ymax_v, color='grey', linestyle=':', linewidth=1.5, zorder=5)

# --- Annotate the amplitude difference along the vertical line ---
mid_y = (y3 + y4) / 2
plt.text(x3 + 3, mid_y, f'{n1_matsumoto_avg:.2f} µV', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

# ---------------------------- N2 amplitude line --------------------------------

# Horizontal line at P2 (min of section 3) from P2 time to the x position of section4 max
x_start = t[max_idx_section3]
x_end = t[min_idx_section4]
y_h = max_value_section3

# Draw the horizontal line (no legend entry)
plt.hlines(y=y_h, xmin=x_start, xmax=x_end, colors='red', linestyles='--', linewidth=2, zorder=4)

# Optional small vertical ticks at the ends to make endpoints clear
plt.vlines([x_start, x_end], ymin=y_h - 2, ymax=y_h + 2, colors='red', linestyle=':', linewidth=1.5, zorder=5)

# Annotate the N2 amplitude (n2_amp_avg) near the middle of the horizontal line
x_mid = x_end + 1
y_text = (y_h + max_value_section3) / 2
plt.text(x_mid, y_text, f'{n2_amp_avg:.2f} µV', color='red',
         fontsize=9, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

# If you want a connecting vertical line from the horizontal end to the section4 max point:
# this draws a vertical line at x_end from y_h up to the section4 max value (max_value_section4)
plt.vlines(x_end, ymin=y_h, ymax=min_value_section4, colors='red', linestyles=':', linewidth=1, alpha=0.8, zorder=3)


plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (µV)')
plt.title(f'Averaged signal ± Standard deviation envelope (N= {N})')
plt.legend()
plt.xlim([0, 250])
plt.ylim([-50, 100])
plt.grid(True)

plt.gca().invert_yaxis()
plt.show()

# # =======================================================================================
# #                            Dataframe with extracted parameters
# # =======================================================================================

# Append mean data to the dataframe
df = pd.DataFrame(data)

# Create a DataFrame for the mean data
mean_df = pd.DataFrame(mean_data)

# Create a DataFrame for the mean data
std_df = pd.DataFrame(std_data)

# Concatenate mean_df with df
df = pd.concat([df, mean_df], ignore_index=True)

# Concatenate std_df with df
df = pd.concat([df, std_df], ignore_index=True)

# Set pandas display options to show the full DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Display the DataFrame
print(df)

# Export DataFrame to Excel file
df.to_excel(r'C:\Users\msedo\Documents\CCEPs\CCEP plots\P17CHW3-REF.xlsx', index=True)


# # # --------------------------- Start - end analysis ----------------------------

# earliest_index = 0
# latest_index = len(f_signals)-1

# # Plot the mean signal with standard deviation
# plt.figure(figsize=(10, 6))
# plt.plot(t, f_signals[earliest_index], label='Initial Signal')
# plt.plot(t, f_signals[latest_index], label='Final Signal')
# plt.xlabel('Time (ms)')
# plt.ylabel('Amplitude (µV)')
# plt.title('Initial and Final CCEPs')
# plt.legend()
# plt.xlim([0, 250])
# plt.ylim([-200, 500])
# plt.grid(True)
# plt.show()

# row1 = df.iloc[earliest_index]
# row2 = df.iloc[latest_index]
# new_df = pd.concat([row1, row2], axis=1).T.reset_index(drop=True)

# # Add mean values to the data
# mean_data_se = {
#     'Signal': ['Mean'],
#     'P1_Latency': [np.mean(new_df['P1_Latency'])],
#     'P1_Amplitude': [np.mean(new_df['P1_Amplitude'])],
#     'N1_Latency': [np.mean(new_df['N1_Latency'])],
#     'N1 matsumoto': [np.mean(new_df['N1 matsumoto'])],
#     'P2_Latency': [np.mean(new_df['P2_Latency'])],
#     'P2_Amplitude': [np.mean(new_df['P2_Amplitude'])],
#     'N2_Latency': [np.mean(new_df['N2_Latency'])],
#     'N2_Amplitude': [np.mean(new_df['N2_Amplitude'])]
# }

# # Add std values to the data
# std_data_se = {
#     'Signal': ['STD'],
#     'P1_Latency': [np.std(new_df['P1_Latency'])],
#     'P1_Amplitude': [np.std(new_df['P1_Amplitude'])],
#     'N1_Latency': [np.std(new_df['N1_Latency'])],
#     'N1 matsumoto': [np.std(new_df['N1 matsumoto'])],
#     'P2_Latency': [np.std(new_df['P2_Latency'])],
#     'P2_Amplitude': [np.std(new_df['P2_Amplitude'])],
#     'N2_Latency': [np.std(new_df['N2_Latency'])],
#     'N2_Amplitude': [np.std(new_df['N2_Amplitude'])]
# }

# # Calculate the percentage difference
# percentage_difference = ((row2 - row1) / row1) * 100

# # Convert to DataFrame for better readability
# pdiff_df = pd.DataFrame(percentage_difference, columns=['Percentage Difference']).T

# pdiff_df['Signal']= 'Diff'

# # Create a DataFrame for the mean data
# mean_df_se = pd.DataFrame(mean_data_se)

# # Create a DataFrame for the mean data
# std_df_se = pd.DataFrame(std_data_se)

# # Concatenate mean_df with df
# new_df = pd.concat([new_df, mean_df_se], ignore_index=True)

# # Concatenate std_df with df
# new_df = pd.concat([new_df, std_df_se], ignore_index=True)

# # Concatenate percentage_difference with df
# new_df = pd.concat([new_df, pdiff_df], ignore_index=True)

# # Set pandas display options to show the full DataFrame
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# # Display the DataFrame
# print(new_df)

# # Export DataFrame to Excel file
# new_df.to_excel(os.path.join(folder, "P4CH5-6_inv_SE.xlsx"), index=True)

# # Compute the coherence
# f, Cxy = coherence(f_signals[earliest_index], f_signals[latest_index], fs=fs, window='hann', nperseg=1250, noverlap=625, nfft=2048, detrend='linear')

# # Plot the coherence
# plt.figure(figsize=(10, 6))
# plt.semilogy(f, Cxy)
# plt.title('Coherence between initial and final CCEPs')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Coherence')
# plt.grid()
# plt.show()

# # Identify peak coherence frequencies
# peaks, _ = find_peaks(Cxy, height=0.35)  # Adjust height for your threshold
# peak_freqs = f[peaks]
# peak_coherences = Cxy[peaks]

# print("")
# print("Peak Coherence Frequencies:", peak_freqs)
# print("Peak Coherence Values:", peak_coherences)
