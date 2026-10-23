"""
Módulo: signal_processing.py
Descripción: Algoritmos de análisis espectral mediante Transformada Rápida
             de Fourier (FFT), detección de picos armónicos y cálculo de
             parámetros físicos fundamentales (T, f, omega).
Autor: Proyecto Acústica y Procesamiento de Señales (UTB)
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, windows

def calcular_fft(senal: np.ndarray, fs: int, aplicar_ventana: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula el espectro de amplitud unilateral (Single-Sided Spectrum) de la señal
    mediante la Transformada Rápida de Fourier para señales reales (rfft).
    
    Parámetros:
    -----------
    senal : np.ndarray
        Vector 1D con las muestras discretas de la señal de audio.
    fs : int
        Frecuencia de muestreo en Hz.
    aplicar_ventana : bool, opcional
        Si es True, aplica una ventana de Hann para mitigar la fuga espectral (spectral leakage).
        
    Retorna:
    --------
    frecs : np.ndarray
        Vector de frecuencias discretas desde 0 hasta fs/2 (Frecuencia de Nyquist).
    espectro_amplitud : np.ndarray
        Amplitudes físicas normalizadas asociadas a cada frecuencia (espectro unilateral).
    """
    N = len(senal)
    
    if aplicar_ventana:
        # Ventana de Hann para suavizar discontinuidades en los bordes temporales
        ventana = windows.hann(N)
        senal_procesada = senal * ventana
        # Corrección de ganancia coherente debida a la ventana (factor de atenuación)
        factor_escala = 2.0 / np.sum(ventana)
    else:
        senal_procesada = senal
        factor_escala = 2.0 / N
        
    # Cálculo de la FFT para valores reales (frecuencias positivas 0 a fs/2)
    fft_compleja = rfft(senal_procesada)
    
    # Vector de frecuencias asociado: frecs = [0, 1*(fs/N), 2*(fs/N), ..., fs/2]
    frecs = rfftfreq(N, d=1.0 / fs)
    
    # Magnitud del espectro unilateral escalado físicamente
    espectro_amplitud = np.abs(fft_compleja) * factor_escala
    
    # El componente continuo (DC, k=0) no se duplica por el factor 2
    if len(espectro_amplitud) > 0:
        espectro_amplitud[0] /= 2.0
        
    return frecs, espectro_amplitud


def detectar_picos(frecs: np.ndarray, espectro: np.ndarray, 
                   f_min: float = 60.0, f_max: float = 5000.0, 
                   umbral_relativo: float = 0.08, 
                   dist_min_hz: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifica los picos espectrales dominantes correspondientes a la frecuencia
    fundamental y sus respectivos sobretonos/armónicos.
    
    Parámetros:
    -----------
    frecs : np.ndarray
        Arreglo de frecuencias del espectro.
    espectro : np.ndarray
        Magnitud de las amplitudes espectrales.
    f_min, f_max : float
        Rango de interés acústico en Hz para descartar ruidos de baja o muy alta frecuencia.
    umbral_relativo : float
        Fracción del pico máximo para considerar un pico relevante (filtro de piso de ruido).
    dist_min_hz : float
        Distancia mínima en Hz entre picos consecutivos para evitar falsos positivos adyacentes.
        
    Retorna:
    --------
    picos_frecs : np.ndarray
        Frecuencias en Hz de los picos detectados, ordenados ascendentemente.
    picos_amps : np.ndarray
        Amplitudes correspondientes a dichos picos.
    """
    df = frecs[1] - frecs[0] if len(frecs) > 1 else 1.0
    distancia_muestras = max(1, int(dist_min_hz / df))
    
    # Enmascarar solo el rango audible de interés
    mascara = (frecs >= f_min) & (frecs <= f_max)
    sub_frecs = frecs[mascara]
    sub_espectro = espectro[mascara]
    
    if len(sub_espectro) == 0:
        return np.array([]), np.array([])
        
    max_amp = np.max(sub_espectro)
    altura_minima = max_amp * umbral_relativo
    
    indices_rel, propiedades = find_peaks(
        sub_espectro, 
        height=altura_minima, 
        distance=distancia_muestras,
        prominence=altura_minima * 0.5
    )
    
    if len(indices_rel) == 0:
        # Fallback al valor máximo global en el rango
        idx_max = np.argmax(sub_espectro)
        return np.array([sub_frecs[idx_max]]), np.array([sub_espectro[idx_max]])
        
    picos_frecs = sub_frecs[indices_rel]
    picos_amps = sub_espectro[indices_rel]
    
    # Ordenar por frecuencia
    orden = np.argsort(picos_frecs)
    return picos_frecs[orden], picos_amps[orden]


def estimar_nota_musical(frecuencia: float) -> tuple[str, float]:
    """
    Determina la nota musical temperada estándar (A4 = 440 Hz) más cercana
    y la desviación en centésimas de semitono (cents).
    """
    if frecuencia <= 0:
        return "N/A", 0.0
    
    nombres_notas = ['Do', 'Do#', 'Re', 'Re#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si']
    # Número de semitonos respecto a A4 (440 Hz)
    n_semitonos = 12 * np.log2(frecuencia / 440.0)
    semitono_redondeado = int(round(n_semitonos))
    
    # Desviación en cents
    desviacion_cents = (n_semitonos - semitono_redondeado) * 100.0
    
    # A4 es la nota 9 (La) en la octava 4
    indice_nota_global = semitono_redondeado + 69  # MIDI note number (A4 = 69)
    nombre = nombres_notas[indice_nota_global % 12]
    octava = (indice_nota_global // 12) - 1
    
    return f"{nombre}{octava}", desviacion_cents


def analizar_parametros_fisicos(frecuencia_fundamental: float, 
                                picos_frecs: np.ndarray = None, 
                                picos_amps: np.ndarray = None) -> dict:
    """
    Calcula los parámetros cinemáticos y ondulatorios del tono acústico:
    - Periodo T (s y ms)
    - Frecuencia f (Hz)
    - Frecuencia angular omega (rad/s)
    - Identificación de armónicos teóricos vs experimentales
    - Estimación de nota musical
    
    Retorna:
    --------
    parametros : dict
        Diccionario estructurado con los valores físicos calculados.
    """
    if frecuencia_fundamental <= 0:
        raise ValueError("La frecuencia fundamental debe ser un valor positivo no nulo.")
        
    T = 1.0 / frecuencia_fundamental
    omega = 2.0 * np.pi * frecuencia_fundamental
    nota, cents = estimar_nota_musical(frecuencia_fundamental)
    
    armonicos = []
    if picos_frecs is not None and len(picos_frecs) > 0:
        for f, amp in zip(picos_frecs, picos_amps):
            ratio = f / frecuencia_fundamental
            n_armonico = int(round(ratio))
            armonicos.append({
                "armonico_n": n_armonico,
                "frecuencia_hz": float(f),
                "amplitud_relativa": float(amp),
                "error_respecto_multiplo": float(abs(f - n_armonico * frecuencia_fundamental))
            })
            
    parametros = {
        "frecuencia_fundamental_hz": frecuencia_fundamental,
        "periodo_s": T,
        "periodo_ms": T * 1000.0,
        "frecuencia_angular_rad_s": omega,
        "nota_estimada": nota,
        "desviacion_cents": cents,
        "armonicos": armonicos
    }
    
    # Impresión elegante por consola compatible con terminales Windows (cp1252)
    print("\n" + "="*65)
    print("        PARAMETROS FISICOS Y ESPECTRALES ESTIMADOS        ")
    print("="*65)
    print(f" * Frecuencia fundamental (f0) : {frecuencia_fundamental:8.2f} Hz")
    print(f" * Periodo temporal (T)        : {T*1000.0:8.3f} ms ({T:8.6f} s)")
    print(f" * Frecuencia angular (omega)  : {omega:8.2f} rad/s")
    print(f" * Nota musical mas cercana    : {nota} ({cents:+.1f} cents)")
    
    if armonicos:
        print("\n Componentes Armonicos Detectados:")
        print(f" {'Armonico':<10} | {'Frecuencia (Hz)':<16} | {'Amplitud':<12} | {'Ratio f/f0':<10}")
        print(" " + "-"*56)
        for arm in armonicos:
            ratio = arm['frecuencia_hz'] / frecuencia_fundamental
            print(f" n = {arm['armonico_n']:<6} | {arm['frecuencia_hz']:14.2f} Hz | {arm['amplitud_relativa']:10.4f} | {ratio:8.2f}")
    print("="*65 + "\n")
    
    return parametros
