"""
Módulo: visualizer.py
Descripción: Generación de visualizaciones científicas de alta calidad para
             señales de audio: envolvente temporal, zoom de oscilación periódica
             y espectro de densidad de Fourier con marcado de armónicos.
Autor: Proyecto Acústica y Procesamiento de Señales (UTB)
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def graficar_resultados(tiempo: np.ndarray, senal: np.ndarray, 
                        frecs: np.ndarray, espectro: np.ndarray,
                        picos_frecs: np.ndarray = None, picos_amps: np.ndarray = None,
                        f0: float = None, T: float = None, omega: float = None,
                        titulo_instrumento: str = "Instrumento Acústico",
                        guardar_figura: str = None,
                        t_zoom_ciclos: int = 5,
                        f_max_espectro: float = 3000.0) -> plt.Figure:
    """
    Crea una figura compuesta de 3 paneles para el análisis físico y espectral:
    1. Señal temporal global (muestra la envolvente ADSR: ataque, sostenimiento, decaimiento).
    2. Zoom temporal centrado (para inspección visual y cálculo manual de T crestas a crestas).
    3. Espectro de frecuencias (FFT) con identificación de f0 y armónicos fn.
    
    Parámetros:
    -----------
    tiempo : np.ndarray
        Vector temporal completo (s).
    senal : np.ndarray
        Vector de amplitud normalizado [-1, 1].
    frecs : np.ndarray
        Vector de frecuencias de la FFT (Hz).
    espectro : np.ndarray
        Magnitud de las amplitudes espectrales.
    picos_frecs, picos_amps : np.ndarray, opcional
        Frecuencias y magnitudes de los armónicos detectados.
    f0, T, omega : float, opcional
        Parámetros físicos para rotular la figura.
    titulo_instrumento : str
        Nombre del instrumento analizado (ej: 'Flauta Dulce', 'Guitarra Acústica').
    guardar_figura : str, opcional
        Ruta para guardar la imagen en formato PNG/PDF a 300 DPI.
    t_zoom_ciclos : int
        Número de ciclos a mostrar en el panel de zoom temporal.
    f_max_espectro : float
        Límite superior del eje X en la gráfica de frecuencia (Hz).
    """
    # Configuración de estilo visual científico y elegante
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.subplots_adjust(hspace=0.42)
    
    # -------------------------------------------------------------
    # PANEL 1: SEÑAL TEMPORAL COMPLETA (ENVOLVENTE GLOBAL)
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(tiempo, senal, color='#1f77b4', linewidth=0.8, alpha=0.9, label='Amplitud s(t)')
    ax1.set_title(f"1. Registro Temporal Completo — {titulo_instrumento}", fontsize=12, fontweight='bold', pad=8)
    ax1.set_xlabel("Tiempo $t$ (s)", fontsize=10)
    ax1.set_ylabel("Amplitud normalizada", fontsize=10)
    ax1.set_ylim(-1.15, 1.15)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Subrayar el tramo estacionario donde se extrae el zoom
    duracion = tiempo[-1]
    t_centro = duracion * 0.5
    if T is not None and T > 0:
        ventana_dur = t_zoom_ciclos * T
        t_ini_zoom = max(0.0, t_centro - ventana_dur / 2.0)
        t_fin_zoom = min(duracion, t_centro + ventana_dur / 2.0)
        ax1.axvspan(t_ini_zoom, t_fin_zoom, color='#ff7f0e', alpha=0.25, label=f'Región de Zoom ({t_zoom_ciclos} ciclos)')
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    
    # -------------------------------------------------------------
    # PANEL 2: ZOOM TEMPORAL (MEDICIÓN DIRECTA DE PERIODO T)
    # -------------------------------------------------------------
    ax2 = axes[1]
    if T is not None and T > 0:
        mascara_zoom = (tiempo >= t_ini_zoom) & (tiempo <= t_fin_zoom)
        t_z = tiempo[mascara_zoom]
        s_z = senal[mascara_zoom]
    else:
        # Si T no está definido, tomar un intervalo por defecto de 20 ms
        t_ini_zoom = duracion * 0.49
        t_fin_zoom = t_ini_zoom + 0.02
        mascara_zoom = (tiempo >= t_ini_zoom) & (tiempo <= t_fin_zoom)
        t_z = tiempo[mascara_zoom]
        s_z = senal[mascara_zoom]

    # Convertir tiempo a milisegundos para legibilidad intuitiva
    t_z_ms = (t_z - t_z[0]) * 1000.0 if len(t_z) > 0 else []
    
    ax2.plot(t_z_ms, s_z, color='#d62728', marker='o', markersize=2.5, linewidth=1.5, label='Onda acústica')
    ax2.set_title(f"2. Zoom Temporal en Régimen Estacionario (Medición de Periodo $T$)", fontsize=12, fontweight='bold', pad=8)
    ax2.set_xlabel("Tiempo relativo $\\Delta t$ (ms)", fontsize=10)
    ax2.set_ylabel("Amplitud", fontsize=10)
    ax2.set_ylim(-1.15, 1.15)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Marcador de periodo en el panel zoom
    if T is not None and T > 0 and len(t_z_ms) > 0:
        T_ms = T * 1000.0
        # Dibujar una flecha de cota mostrando el periodo T
        ax2.annotate(
            '', xy=(0.5 * T_ms, 0.85), xytext=(1.5 * T_ms, 0.85),
            arrowprops=dict(arrowstyle='<->', color='#2ca02c', lw=2)
        )
        ax2.text(
            1.0 * T_ms, 0.92, f"$T \\approx {T_ms:.2f}$ ms",
            ha='center', va='bottom', color='#2ca02c', fontweight='bold', fontsize=10
        )
        
    ax2.legend(loc='upper right', frameon=True, fontsize=9)
    
    # -------------------------------------------------------------
    # PANEL 3: ESPECTRO DE FOURIER (FFT) Y PICOS ARMÓNICOS
    # -------------------------------------------------------------
    ax3 = axes[2]
    # Limitar visualización hasta f_max_espectro para máxima claridad
    mascara_frec = (frecs >= 0) & (frecs <= f_max_espectro)
    f_plot = frecs[mascara_frec]
    esp_plot = espectro[mascara_frec]
    
    ax3.plot(f_plot, esp_plot, color='#9467bd', linewidth=1.2, label='Espectro unilateral $|X(f)|$')
    ax3.fill_between(f_plot, esp_plot, color='#9467bd', alpha=0.2)
    ax3.set_title("3. Espectro de Frecuencias (Transformada Rápida de Fourier — FFT)", fontsize=12, fontweight='bold', pad=8)
    ax3.set_xlabel("Frecuencia $f$ (Hz)", fontsize=10)
    ax3.set_ylabel("Magnitud espectral", fontsize=10)
    ax3.set_xlim(0, f_max_espectro)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Anotar picos armónicos detectados
    if picos_frecs is not None and len(picos_frecs) > 0:
        for idx, (pf, pa) in enumerate(zip(picos_frecs, picos_amps)):
            if pf <= f_max_espectro:
                etiqueta = f"$f_0={pf:.1f}$ Hz" if idx == 0 else f"$f_{idx+1}={pf:.1f}$ Hz"
                ax3.plot(pf, pa, 'ro', markersize=6)
                ax3.annotate(
                    etiqueta,
                    xy=(pf, pa),
                    xytext=(pf, pa + (np.max(esp_plot) * 0.08)),
                    ha='center', fontsize=8.5, fontweight='bold',
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1, shrinkB=4)
                )
                
    # Cuadro de texto resumen con parámetros físicos calculados
    if f0 is not None and T is not None and omega is not None:
        texto_resumen = (
            f"Parámetros Físicos:\n"
            f"• $f_0 = {f0:.2f}$ Hz\n"
            f"• $T = {T*1000.0:.3f}$ ms\n"
            f"• $\\omega = {omega:.2f}$ rad/s"
        )
        ax3.text(
            0.78, 0.70, texto_resumen,
            transform=ax3.transAxes,
            fontsize=9.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffdd', edgecolor='#ccccaa', alpha=0.95)
        )
        
    ax3.legend(loc='upper right', frameon=True, fontsize=9)
    
    # Guardar figura en disco si se solicita
    if guardar_figura:
        os.makedirs(os.path.dirname(guardar_figura) or '.', exist_ok=True)
        fig.savefig(guardar_figura, dpi=300, bbox_inches='tight')
        print(f"[INFO] Gráfica científica exportada a: {guardar_figura}")
        
    return fig
