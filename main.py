"""
================================================================================
PROYECTO: Registro de Ondas Sonoras y Análisis de su Espectro
ASIGNATURA: Física Ondulatoria / Procesamiento de Señales
INSTITUCIÓN: Universidad Tecnológica de Bolívar (UTB)
================================================================================
Descripción:
  Este programa captura o carga audio, procesa la señal en el dominio temporal
  y de la frecuencia mediante la Transformada Rápida de Fourier (FFT),
  extrae la frecuencia fundamental (f₀) y los armónicos (fₙ), calcula los
  parámetros físicos (T, f, ω) y genera gráficas comparativas listas para el
  informe experimental en PDF.
================================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Importar funciones de los módulos del proyecto
from audio_recorder import capturar_audio, cargar_audio
from signal_processing import calcular_fft, detectar_picos, analizar_parametros_fisicos
from visualizer import graficar_resultados

def generar_senal_sintetica(tipo: str = "flauta", duracion: float = 2.0, fs: int = 44100) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera una señal sintética realista modelada físicamente para pruebas y demostración
    cuando no se dispone de micrófono físico o instrumentos a mano.
    - Flauta: Tono puro predominante en el armónico fundamental con sobretonos muy débiles.
    - Guitarra: Sonido rico en armónicos pares e impares con decaimiento exponencial característico.
    """
    t = np.linspace(0.0, duracion, int(duracion * fs), endpoint=False)
    
    if tipo.lower() == "flauta":
        # Flauta dulce tocando un La4 (A4 ~ 440 Hz) o Do5 (C5 ~ 523.25 Hz)
        f0 = 523.25 # Do5
        # Envolvente suave de aire
        envolvente = np.sin(np.pi * t / duracion)**0.5
        # Contenido espectral: fundamental muy fuerte (90%), segundo armónico suave (10%)
        senal = (1.0 * np.sin(2 * np.pi * f0 * t) +
                 0.12 * np.sin(2 * np.pi * 2 * f0 * t) +
                 0.03 * np.sin(2 * np.pi * 3 * f0 * t)) * envolvente
                 
    elif tipo.lower() == "guitarra":
        # Guitarra acústica tocando un La3 (A3 ~ 220 Hz)
        f0 = 220.0
        # Decaimiento percusivo exponencial de cuerda pulsada (Attack-Decay)
        envolvente = np.exp(-1.8 * t)
        # Riqueza armónica de cuerda tensa
        senal = (1.00 * np.sin(2 * np.pi * 1 * f0 * t) +
                 0.65 * np.sin(2 * np.pi * 2 * f0 * t) +
                 0.45 * np.sin(2 * np.pi * 3 * f0 * t) +
                 0.30 * np.sin(2 * np.pi * 4 * f0 * t) +
                 0.18 * np.sin(2 * np.pi * 5 * f0 * t)) * envolvente
    else:
        # Tono senoidal puro 440 Hz
        f0 = 440.0
        senal = np.sin(2 * np.pi * f0 * t)

    # Agregar leve ruido gaussiano ambiental realista (-50 dB)
    ruido = np.random.normal(0, 0.003, len(t))
    senal = senal + ruido
    
    # Normalizar
    senal = senal / np.max(np.abs(senal))
    return t, senal


def procesar_y_analizar(tiempo: np.ndarray, senal: np.ndarray, fs: int, 
                        nombre_instrumento: str, ruta_salida_figura: str = None) -> dict:
    """
    Pipeline completo de análisis: FFT -> Detección de picos -> Parámetros físicos -> Gráficas.
    """
    print(f"\n---> Analizando espectro y dinámica de: {nombre_instrumento}")
    
    # 1. Cálculo de FFT unilateral con ventana de Hann
    frecs, espectro = calcular_fft(senal, fs, aplicar_ventana=True)
    
    # 2. Detección automática de la frecuencia fundamental y armónicos
    picos_frecs, picos_amps = detectar_picos(frecs, espectro, f_min=70.0, f_max=4000.0)
    
    if len(picos_frecs) == 0:
        print("[ADVERTENCIA] No se detectaron picos con suficiente energía. Usando pico global.")
        idx_max = np.argmax(espectro[frecs >= 60])
        f0 = frecs[frecs >= 60][idx_max]
    else:
        # La frecuencia fundamental f0 corresponde al primer pico significativo
        f0 = picos_frecs[0]
        
    # 3. Parámetros físicos ondulatorios
    parametros = analizar_parametros_fisicos(f0, picos_frecs, picos_amps)
    
    # 4. Generación y renderizado de gráficas científicas
    graficar_resultados(
        tiempo=tiempo,
        senal=senal,
        frecs=frecs,
        espectro=espectro,
        picos_frecs=picos_frecs,
        picos_amps=picos_amps,
        f0=parametros['frecuencia_fundamental_hz'],
        T=parametros['periodo_s'],
        omega=parametros['frecuencia_angular_rad_s'],
        titulo_instrumento=nombre_instrumento,
        guardar_figura=ruta_salida_figura
    )
    
    return parametros


def menu_interactivo():
    """
    Menú de consola intuitivo y amigable para el usuario y el docente.
    """
    print("=" * 70)
    print("   SISTEMA DE ANÁLISIS ACÚSTICO Y TRANSFORMADA DE FOURIER (FFT)   ")
    print("             Universidad Tecnológica de Bolívar (UTB)             ")
    print("=" * 70)
    print("Seleccione una opción de ejecución:")
    print(" 0. Iniciar Interfaz Gráfica Moderna (GUI CustomTkinter)")
    print(" 1. Grabar instrumento en tiempo real desde el micrófono (Consola)")
    print(" 2. Cargar archivo de audio WAV existente (Consola)")
    print(" 3. Modo Demostración Rápida (Flauta vs Guitarra sintéticas)")
    print(" 4. Salir")
    print("-" * 70)
    
    opcion = input("Ingrese el número de opción (0-4): ").strip()
    
    fs = 44100 # Frecuencia de muestreo estándar CD
    
    if opcion == "0":
        print("\n[INFO] Lanzando Interfaz Gráfica Moderna...")
        from gui_app import main as launch_gui
        launch_gui()
        
    elif opcion == "1":
        instrumento = input("\nIngrese el nombre del instrumento (ej: Flauta / Guitarra / Voz): ").strip() or "Instrumento"
        try:
            duracion = float(input("Ingrese la duración de grabación en segundos [por defecto 3.0]: ") or "3.0")
        except ValueError:
            duracion = 3.0
            
        nombre_archivo = f"audio_samples/{instrumento.lower().replace(' ', '_')}.wav"
        salida_figura = f"figures/analisis_{instrumento.lower().replace(' ', '_')}.png"
        
        tiempo, senal = capturar_audio(duracion, fs=fs, guardar_wav=nombre_archivo)
        procesar_y_analizar(tiempo, senal, fs, instrumento, salida_figura)
        plt.show()
        
    elif opcion == "2":
        ruta_wav = input("\nIngrese la ruta del archivo WAV: ").strip()
        if not os.path.exists(ruta_wav):
            print(f"[ERROR] El archivo '{ruta_wav}' no existe.")
            return
        nombre_base = os.path.splitext(os.path.basename(ruta_wav))[0]
        salida_figura = f"figures/analisis_{nombre_base}.png"
        
        tiempo, senal, fs_cargado = cargar_audio(ruta_wav)
        procesar_y_analizar(tiempo, senal, fs_cargado, nombre_base.capitalize(), salida_figura)
        plt.show()
        
    elif opcion == "3":
        print("\n[MODO DEMO] Generando datos de alta fidelidad para Flauta y Guitarra...")
        os.makedirs("audio_samples", exist_ok=True)
        os.makedirs("figures", exist_ok=True)
        
        # 1. Análisis de Flauta
        t_flauta, s_flauta = generar_senal_sintetica("flauta", duracion=2.5, fs=fs)
        # Guardar muestra WAV para auditoría
        from scipy.io import wavfile
        wavfile.write("audio_samples/flauta_demo.wav", fs, np.int16(s_flauta * 32767))
        param_flauta = procesar_y_analizar(t_flauta, s_flauta, fs, "Flauta Dulce (Tono Puro)", "figures/analisis_flauta.png")
        
        # 2. Análisis de Guitarra
        t_guitarra, s_guitarra = generar_senal_sintetica("guitarra", duracion=2.5, fs=fs)
        wavfile.write("audio_samples/guitarra_demo.wav", fs, np.int16(s_guitarra * 32767))
        param_guitarra = procesar_y_analizar(t_guitarra, s_guitarra, fs, "Guitarra Acústica (Rica en Armónicos)", "figures/analisis_guitarra.png")
        
        print("\n" + "#"*70)
        print("          RESUMEN COMPARATIVO FISICO: FLAUTA vs GUITARRA          ")
        print("#"*70)
        print(f" FLAUTA DULCE:")
        print(f"   * f0 = {param_flauta['frecuencia_fundamental_hz']:.2f} Hz | T = {param_flauta['periodo_ms']:.3f} ms | w = {param_flauta['frecuencia_angular_rad_s']:.2f} rad/s")
        print(f"   * Nota musical: {param_flauta['nota_estimada']}")
        print(f"   * Comportamiento: Tono casi puramente senoidal con sobretonos minimos.")
        print("-" * 70)
        print(f" GUITARRA ACUSTICA:")
        print(f"   * f0 = {param_guitarra['frecuencia_fundamental_hz']:.2f} Hz | T = {param_guitarra['periodo_ms']:.3f} ms | w = {param_guitarra['frecuencia_angular_rad_s']:.2f} rad/s")
        print(f"   * Nota musical: {param_guitarra['nota_estimada']}")
        print(f"   * Comportamiento: Multiples armonicos equiespaciados (n*f0) determinantes del timbre.")
        print("#"*70)
        print(f"\n[EXITO] Graficas exportadas con exito en la carpeta 'figures/':")
        print("  1. figures/analisis_flauta.png")
        print("  2. figures/analisis_guitarra.png")
        print("Puede abrirlas para adjuntarlas directamente a su informe en PDF.")
        plt.show()
        
    elif opcion == "4":
        print("Finalizando programa. ¡Éxitos en la entrega del proyecto!")
        sys.exit(0)
    else:
        print("[ERROR] Opción inválida. Intente de nuevo.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--gui", "-g"]:
        from gui_app import main as launch_gui
        launch_gui()
    else:
        menu_interactivo()
