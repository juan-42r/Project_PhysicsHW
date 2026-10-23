"""
================================================================================
PROYECTO: Registro de Ondas Sonoras y Análisis de su Espectro
ASIGNATURA: Física Ondulatoria y Procesamiento de Señales
================================================================================
Este script contiene las 4 funciones canónicas solicitadas en la rúbrica
integradas en un único archivo ejecutable:
1. capturar_audio(duracion, fs)
2. calcular_fft(senal, fs)
3. graficar_resultados(tiempo, senal, frecs, espectro)
4. analizar_parametros_fisicos(frecuencia_fundamental)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, windows
import sounddevice as sd

def capturar_audio(duracion: float = 3.0, fs: int = 44100) -> tuple[np.ndarray, np.ndarray]:
    """
    Captura la señal de audio del micrófono y devuelve un arreglo de NumPy y el vector de tiempo.
    
    Parámetros:
    -----------
    duracion : float
        Tiempo de grabación en segundos.
    fs : int
        Frecuencia de muestreo en Hz (estándar: 44100 Hz).
        
    Retorna:
    --------
    tiempo : np.ndarray
        Vector temporal con valores en segundos.
    senal : np.ndarray
        Vector con las muestras de amplitud normalizadas y sin offset DC.
    """
    total_muestras = int(duracion * fs)
    print(f"\n[GRABACIÓN] Capturando {duracion} segundos a {fs} Hz...")
    print(">>> Toque la nota o emita el sonido hacia el micrófono...")
    
    grabacion = sd.rec(total_muestras, samplerate=fs, channels=1, dtype='float32')
    sd.wait() # Bloquea hasta que la grabación termine
    print("[GRABACIÓN] Finalizada con éxito.")
    
    senal = grabacion.flatten()
    # Eliminar componente de polarización DC
    senal = senal - np.mean(senal)
    # Normalizar por el pico absoluto
    max_amp = np.max(np.abs(senal))
    if max_amp > 1e-6:
        senal = senal / max_amp
        
    tiempo = np.linspace(0.0, duracion, total_muestras, endpoint=False)
    return tiempo, senal


def calcular_fft(senal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica la FFT, calcula las magnitudes y genera el vector de frecuencias de 0 a fs/2.
    
    Parámetros:
    -----------
    senal : np.ndarray
        Arreglo 1D de muestras de audio.
    fs : int
        Frecuencia de muestreo en Hz.
        
    Retorna:
    --------
    frecs : np.ndarray
        Vector de frecuencias discretas desde 0 hasta la frecuencia de Nyquist (fs/2).
    espectro : np.ndarray
        Magnitud normalizada del espectro unilateral de amplitudes.
    """
    N = len(senal)
    # Aplicar ventana de Hann para mitigar la fuga espectral (Spectral Leakage)
    ventana = windows.hann(N)
    senal_ventaneada = senal * ventana
    
    # Transformada Rápida de Fourier para señales reales (rfft)
    fft_vals = rfft(senal_ventaneada)
    frecs = rfftfreq(N, d=1.0 / fs)
    
    # Corrección de amplitud por ganancia coherente de la ventana de Hann
    espectro = np.abs(fft_vals) * (2.0 / np.sum(ventana))
    if len(espectro) > 0:
        espectro[0] /= 2.0 # Componente DC sin duplicar
        
    return frecs, espectro


def analizar_parametros_fisicos(frecuencia_fundamental: float) -> tuple[float, float, float]:
    """
    Calcula e imprime los parámetros físicos fundamentales:
    - Periodo T (s)
    - Frecuencia f (Hz)
    - Frecuencia angular omega (rad/s)
    
    Parámetros:
    -----------
    frecuencia_fundamental : float
        Frecuencia del primer pico dominante (f0) en Hz.
        
    Retorna:
    --------
    T : float
        Periodo de oscilación en segundos.
    f : float
        Frecuencia de oscilación en Hz.
    omega : float
        Frecuencia angular en rad/s.
    """
    f = float(frecuencia_fundamental)
    if f <= 0:
        raise ValueError("La frecuencia fundamental debe ser estrictamente positiva.")
        
    T = 1.0 / f
    omega = 2.0 * np.pi * f
    
    print("\n" + "="*50)
    print("          PARAMETROS FISICOS DE LA ONDA          ")
    print("="*50)
    print(f" * Frecuencia (f)          : {f:8.2f} Hz")
    print(f" * Periodo (T)             : {T*1000.0:8.3f} ms ({T:.6f} s)")
    print(f" * Frecuencia Angular (w)  : {omega:8.2f} rad/s")
    print("="*50 + "\n")
    
    return T, f, omega


def graficar_resultados(tiempo: np.ndarray, senal: np.ndarray, frecs: np.ndarray, espectro: np.ndarray):
    """
    Genera la figura científica con las tres gráficas requeridas:
    - Gráfica 1: Señal temporal completa (envolvente).
    - Gráfica 2: Zoom de la señal temporal (para contar ciclos y medir el periodo T).
    - Gráfica 3: Zoom del espectro de frecuencias identificando el pico fundamental f0 y armónicos.
    """
    # 1. Encontrar el pico fundamental f0 y armónicos en el rango audible
    mascara_audible = (frecs >= 60.0) & (frecs <= 3500.0)
    sub_frecs = frecs[mascara_audible]
    sub_esp = espectro[mascara_audible]
    
    picos_idx, _ = find_peaks(sub_esp, height=np.max(sub_esp) * 0.1, distance=int(len(sub_frecs)/80) + 1)
    
    if len(picos_idx) > 0:
        f0 = sub_frecs[picos_idx[0]]
        picos_f = sub_frecs[picos_idx]
        picos_a = sub_esp[picos_idx]
    else:
        idx_max = np.argmax(sub_esp)
        f0 = sub_frecs[idx_max]
        picos_f = np.array([f0])
        picos_a = np.array([sub_esp[idx_max]])
        
    # 2. Calcular parámetros físicos
    T, f, omega = analizar_parametros_fisicos(f0)
    
    # 3. Configurar figura con 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    fig.subplots_adjust(hspace=0.45)
    
    # --- GRÁFICA 1: Señal temporal completa (Envolvente) ---
    ax1.plot(tiempo, senal, color='#1f77b4', lw=0.7)
    ax1.set_title("Gráfica 1: Señal Temporal Completa (Envolvente de Amplitud)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Tiempo $t$ (s)")
    ax1.set_ylabel("Amplitud normalizada")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Resaltar la zona de zoom en la gráfica temporal completa
    t_centro = tiempo[-1] * 0.5
    dur_zoom = 5 * T # Mostrar 5 ciclos
    t_ini = max(0.0, t_centro - dur_zoom / 2.0)
    t_fin = min(tiempo[-1], t_centro + dur_zoom / 2.0)
    ax1.axvspan(t_ini, t_fin, color='orange', alpha=0.3, label='Región de Zoom')
    ax1.legend(loc='upper right')
    
    # --- GRÁFICA 2: Zoom de la señal temporal (Conteo de ciclos y medición de T) ---
    mascara_z = (tiempo >= t_ini) & (tiempo <= t_fin)
    t_zoom = (tiempo[mascara_z] - t_ini) * 1000.0 # En milisegundos
    s_zoom = senal[mascara_z]
    
    ax2.plot(t_zoom, s_zoom, color='#d62728', marker='.', markersize=3, lw=1.5)
    ax2.set_title("Gráfica 2: Zoom de la Señal Temporal (Medición directa del Periodo $T$)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Tiempo $\\Delta t$ (ms)")
    ax2.set_ylabel("Amplitud")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Cota ilustrativa del periodo
    ax2.annotate('', xy=(0.5 * T * 1000, 0.8), xytext=(1.5 * T * 1000, 0.8),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax2.text(1.0 * T * 1000, 0.86, f"$T \\approx {T*1000:.2f}$ ms", ha='center', color='green', fontweight='bold')
    
    # --- GRÁFICA 3: Zoom del espectro de frecuencias (FFT) ---
    f_corte = min(3000.0, frecs[-1])
    mascara_esp = (frecs >= 0) & (frecs <= f_corte)
    
    ax3.plot(frecs[mascara_esp], espectro[mascara_esp], color='#9467bd', lw=1.2)
    ax3.fill_between(frecs[mascara_esp], espectro[mascara_esp], color='#9467bd', alpha=0.2)
    ax3.set_title("Gráfica 3: Espectro de Frecuencias (FFT) con Identificación de Armónicos", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Frecuencia $f$ (Hz)")
    ax3.set_ylabel("Magnitud")
    ax3.set_xlim(0, f_corte)
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # Anotaciones de picos
    for i, (pf, pa) in enumerate(zip(picos_f, picos_a)):
        if pf <= f_corte:
            tag = f"$f_0={pf:.1f}$ Hz" if i == 0 else f"$f_{i+1}={pf:.1f}$ Hz"
            ax3.plot(pf, pa, 'ro', markersize=5)
            ax3.annotate(tag, xy=(pf, pa), xytext=(pf, pa + np.max(sub_esp)*0.08),
                         ha='center', fontsize=8, fontweight='bold',
                         arrowprops=dict(arrowstyle='->', lw=0.8, color='black'))
                         
    # Recuadro con resumen
    ax3.text(0.76, 0.75, f"$f_0 = {f:.2f}$ Hz\n$T = {T*1000:.2f}$ ms\n$\\omega = {omega:.1f}$ rad/s",
             transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
             
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Iniciando análisis de ondas sonoras...")
    # Ejemplo de uso directo
    try:
        t, s = capturar_audio(duracion=3.0, fs=44100)
    except Exception as err:
        print(f"[AVISO] Micrófono no disponible ({err}). Generando onda de prueba (440 Hz)...")
        fs = 44100
        t = np.linspace(0, 3.0, 3 * fs, endpoint=False)
        s = np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 880.0 * t)
        
    frecs, esp = calcular_fft(s, 44100)
    graficar_resultados(t, s, frecs, esp)
