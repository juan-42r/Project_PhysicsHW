"""
Módulo: audio_recorder.py
Descripción: Captura de señales acústicas en tiempo real desde el micrófono
             del ordenador y lectura de archivos de audio WAV.
Autor: Proyecto Acústica y Procesamiento de Señales (UTB)
"""

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import os

def capturar_audio(duracion: float, fs: int = 44100, canales: int = 1, guardar_wav: str = None, device: int | str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Captura una señal acústica en tiempo real desde el micrófono especificado o predeterminado.
    
    Parámetros:
    -----------
    duracion : float
        Duración de la captura en segundos (p. ej. 2.0 o 3.0 s).
    fs : int, opcional
        Frecuencia de muestreo en Hz (por defecto 44100 Hz, estándar acústico).
    canales : int, opcional
        Número de canales de audio (1 para mono, 2 para estéreo).
    guardar_wav : str, opcional
        Ruta del archivo para almacenar la grabación en formato WAV.
    device : int o str, opcional
        ID o nombre del dispositivo de hardware de entrada en sounddevice.
        
    Retorna:
    --------
    tiempo : np.ndarray
        Vector temporal discretizado t = [0, 1/fs, 2/fs, ..., (N-1)/fs].
    senal : np.ndarray
        Arreglo 1D de amplitudes normalizadas en el rango [-1.0, 1.0] con
        eliminación previa de componente DC (offset de polarización).
    """
    total_muestras = int(duracion * fs)
    print(f"\n[INFO] Iniciando grabación por micrófono ({duracion} s a {fs} Hz, dev={device})...")
    print("[INFO] ¡Emite o toca la nota musical ahora!")
    
    try:
        # Grabación síncrona con sounddevice especificando el dispositivo de entrada
        grabacion = sd.rec(total_muestras, samplerate=fs, channels=canales, dtype='float32', device=device)
        sd.wait() # Espera hasta completar el tiempo de muestreo
        print("[INFO] Grabación finalizada exitosamente.")
    except Exception as e:
        raise RuntimeError(f"Error al acceder al hardware de audio: {e}\n"
                           f"Verifique que el micrófono esté conectado y con permisos concedidos.")
    
    # Aplanado a vector 1D si es monocanal
    senal = grabacion.flatten() if canales == 1 else grabacion[:, 0]
    
    # Eliminación de offset DC (polarización continua del convertidor ADC)
    senal = senal - np.mean(senal)
    
    # Normalización por el pico máximo para evitar saturación y facilitar el análisis
    max_val = np.max(np.abs(senal))
    if max_val > 1e-6:
        senal = senal / max_val
        
    # Vector de tiempo continuo discretizado
    tiempo = np.linspace(0.0, duracion, total_muestras, endpoint=False)
    
    # Almacenamiento opcional en formato WAV para reproducibilidad
    if guardar_wav:
        os.makedirs(os.path.dirname(guardar_wav) or '.', exist_ok=True)
        # Convertir a formato PCM de 16 bits para guardar WAV estándar
        senal_pcm = np.int16(senal * 32767)
        wavfile.write(guardar_wav, fs, senal_pcm)
        print(f"[INFO] Audio guardado para auditoría en: {guardar_wav}")
        
    return tiempo, senal


def cargar_audio(ruta_wav: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Carga una señal de audio desde un archivo WAV pregrabado.
    
    Parámetros:
    -----------
    ruta_wav : str
        Ruta del archivo de audio WAV.
        
    Retorna:
    --------
    tiempo : np.ndarray
        Vector temporal de la señal en segundos.
    senal : np.ndarray
        Vector de amplitud normalizado sin offset DC.
    fs : int
        Frecuencia de muestreo extraída de los metadatos del WAV.
    """
    if not os.path.exists(ruta_wav):
        raise FileNotFoundError(f"No se encontró el archivo de audio: {ruta_wav}")
        
    fs, datos = wavfile.read(ruta_wav)
    
    # Conversión a mono si es estéreo
    if datos.ndim > 1:
        datos = datos[:, 0]
        
    # Conversión a float32 normalizado
    if datos.dtype == np.int16:
        senal = datos.astype(np.float32) / 32768.0
    elif datos.dtype == np.int32:
        senal = datos.astype(np.float32) / 2147483648.0
    elif datos.dtype == np.uint8:
        senal = (datos.astype(np.float32) - 128.0) / 128.0
    else:
        senal = datos.astype(np.float32)
        
    # Remover DC offset y normalizar
    senal = senal - np.mean(senal)
    max_val = np.max(np.abs(senal))
    if max_val > 1e-6:
        senal = senal / max_val
        
    duracion = len(senal) / fs
    tiempo = np.linspace(0.0, duracion, len(senal), endpoint=False)
    
    return tiempo, senal, fs
