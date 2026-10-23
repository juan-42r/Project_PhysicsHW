"""
================================================================================
PROYECTO: Sistema de Análisis Acústico y Transformada de Fourier (FFT)
INSTITUCIÓN: Universidad Tecnológica de Bolívar (UTB)
ASIGNATURA: Física Ondulatoria / Procesamiento Digital de Señales
AUTOR: Equipo de Desarrollo - Proyecto Acústico UTB
================================================================================
Módulo: gui_app.py
Descripción:
  Interfaz gráfica moderna (Dark Mode) con CustomTkinter y Matplotlib.
  Permite la captura en tiempo real por micrófono en hilo secundario (multithreading),
  carga de archivos WAV, generación de demos sintéticos (flauta, guitarra, diapasón),
  cálculo automático de FFT, detección de armónicos, visualización interactiva de
  3 gráficas científicas sincronizadas y métricas físicas (T, f, omega, nota).
================================================================================
"""

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

# Librerías científicas y de hardware
import sounddevice as sd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Framework moderno de GUI
try:
    import customtkinter as ctk
except ModuleNotFoundError:
    import subprocess
    print("[INFO] 'customtkinter' no está detectado en este entorno de Python. Instalando automáticamente...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter"])
        import customtkinter as ctk
        print("[ÉXITO] 'customtkinter' instalado correctamente.")
    except Exception as err:
        import tkinter.messagebox as mb
        mb.showerror(
            "Dependencia Faltante",
            f"Falta instalar la librería 'customtkinter'.\n\n"
            f"Por favor ejecute en su terminal:\n"
            f"pip install customtkinter\n\n"
            f"Error: {err}"
        )
        sys.exit(1)

# Módulos del proyecto
from audio_recorder import capturar_audio, cargar_audio
from signal_processing import calcular_fft, detectar_picos, analizar_parametros_fisicos

# ==============================================================================
# CONFIGURACIÓN DE TEMA Y ESTÉTICA VISUAL (DARK MODE HIGH-TECH)
# ==============================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

PALETA = {
    "bg_principal": "#0d1117",     # Fondo oscuro general
    "bg_sidebar": "#161b22",       # Barra lateral de control
    "bg_card": "#21262d",          # Tarjetas de datos y resultados
    "border": "#30363d",           # Bordes sutiles
    "cyan": "#00d2ff",             # Color acento 1 (Señal temporal)
    "red": "#ff4d6d",              # Color acento 2 (Zoom temporal y grabación)
    "purple": "#b388ff",           # Color acento 3 (FFT Espectro)
    "green": "#00e676",            # Color acento 4 (Periodo T y éxito)
    "yellow": "#ffd166",           # Color acento 5 (Advertencias y notas)
    "text_primary": "#f0f6fc",     # Texto principal
    "text_secondary": "#8b949e",   # Texto secundario
    "plot_bg": "#12171f",          # Fondo de las gráficas Matplotlib
    "plot_axis": "#1f242c"         # Fondo de los ejes Matplotlib
}


class InterfazAcusticaUTB(ctk.CTk):
    """
    Ventana principal de la aplicación con arquitectura desacoplada por eventos
    y multithreading para captura/procesamiento no bloqueante.
    """

    def __init__(self):
        super().__init__()

        # -------------------------------------------------------------
        # 1. Configuración de la Ventana Principal
        # -------------------------------------------------------------
        self.title("UTB | Análisis Acústico y Espectro FFT — Física Ondulatoria")
        self.geometry("1400x900")
        self.minsize(1100, 750)
        self.configure(fg_color=PALETA["bg_principal"])

        # Estado interno de la aplicación
        self.tiempo_senal = None
        self.amplitud_senal = None
        self.fs_actual = 44100
        self.hilo_grabacion = None
        self.esta_grabando = False
        self.esta_reproduciendo = False

        # Configuración del layout base (2 columnas: Sidebar izquierda + Panel principal derecho)
        self.grid_columnconfigure(0, weight=0, minsize=320) # Sidebar de control
        self.grid_columnconfigure(1, weight=1)              # Gráficas y Métricas
        self.grid_rowconfigure(0, weight=1)

        # -------------------------------------------------------------
        # 2. Construcción de Componentes Visuales
        # -------------------------------------------------------------
        self._construir_sidebar()
        self._construir_panel_principal()

        # Poblar lista de micrófonos detectados
        self._actualizar_dispositivos_audio()

        # Generar señal de bienvenida/demostración inicial (Flauta pura)
        self.after(300, self.generar_audio_demo, "flauta")

    # ==========================================================================
    # CONSTRUCCIÓN DE LA BARRA LATERAL (CONTROLES Y CONFIGURACIÓN)
    # ==========================================================================
    def _construir_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self,
            fg_color=PALETA["bg_sidebar"],
            corner_radius=0,
            border_width=1,
            border_color=PALETA["border"]
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(20, weight=1) # Espaciador inferior

        # Encabezado Institucional UTB
        header_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header_frame.pack(fill="x", padx=18, pady=(20, 15))

        lbl_institucion = ctk.CTkLabel(
            header_frame,
            text="UNIVERSIDAD TECNOLÓGICA DE BOLÍVAR",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=PALETA["cyan"]
        )
        lbl_institucion.pack(anchor="w")

        lbl_titulo = ctk.CTkLabel(
            header_frame,
            text="Laboratorio Acústico FFT",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=PALETA["text_primary"]
        )
        lbl_titulo.pack(anchor="w", pady=(2, 0))

        lbl_subtitulo = ctk.CTkLabel(
            header_frame,
            text="Física Ondulatoria • Proyecto de Aula",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=PALETA["text_secondary"]
        )
        lbl_subtitulo.pack(anchor="w")

        # Separador visual
        ctk.CTkFrame(self.sidebar, height=1, fg_color=PALETA["border"]).pack(fill="x", padx=18, pady=5)

        # ----------------- SECCIÓN 1: HARDWARE DE AUDIO -----------------
        lbl_sec1 = ctk.CTkLabel(
            self.sidebar,
            text="⚙️ CONFIGURACIÓN DE AUDIO",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=PALETA["text_secondary"]
        )
        lbl_sec1.pack(anchor="w", padx=18, pady=(10, 4))

        # Selector de Micrófono
        lbl_mic = ctk.CTkLabel(self.sidebar, text="Micrófono de entrada:", font=ctk.CTkFont(size=12))
        lbl_mic.pack(anchor="w", padx=18, pady=(4, 2))

        mic_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        mic_frame.pack(fill="x", padx=18, pady=(0, 8))

        self.combo_dispositivos = ctk.CTkOptionMenu(
            mic_frame,
            values=["Detectando dispositivos..."],
            fg_color=PALETA["bg_card"],
            button_color=PALETA["border"],
            button_hover_color=PALETA["cyan"],
            text_color=PALETA["text_primary"],
            dropdown_fg_color=PALETA["bg_card"],
            dynamic_resizing=False,
            height=32
        )
        self.combo_dispositivos.pack(side="left", fill="x", expand=True, padx=(0, 6))

        btn_refrescar = ctk.CTkButton(
            mic_frame,
            text="🔄",
            width=36,
            height=32,
            fg_color=PALETA["bg_card"],
            hover_color=PALETA["border"],
            command=self._actualizar_dispositivos_audio
        )
        btn_refrescar.pack(side="right")

        # Frecuencia de muestreo (fs)
        lbl_fs = ctk.CTkLabel(self.sidebar, text="Frecuencia de Muestreo (fs):", font=ctk.CTkFont(size=12))
        lbl_fs.pack(anchor="w", padx=18, pady=(4, 2))

        self.combo_fs = ctk.CTkSegmentedButton(
            self.sidebar,
            values=["22050 Hz", "44100 Hz", "48000 Hz"],
            selected_color=PALETA["cyan"],
            selected_hover_color="#00b4d8",
            unselected_color=PALETA["bg_card"],
            unselected_hover_color=PALETA["border"]
        )
        self.combo_fs.set("44100 Hz")
        self.combo_fs.pack(fill="x", padx=18, pady=(0, 10))

        # Duración de captura en segundos
        dur_header = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        dur_header.pack(fill="x", padx=18, pady=(4, 2))

        lbl_dur = ctk.CTkLabel(dur_header, text="Duración de captura:", font=ctk.CTkFont(size=12))
        lbl_dur.pack(side="left")

        self.lbl_dur_valor = ctk.CTkLabel(
            dur_header,
            text="3.0 s",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=PALETA["cyan"]
        )
        self.lbl_dur_valor.pack(side="right")

        self.slider_duracion = ctk.CTkSlider(
            self.sidebar,
            from_=1.0,
            to=6.0,
            number_of_steps=10,
            button_color=PALETA["cyan"],
            button_hover_color="#00b4d8",
            progress_color=PALETA["cyan"],
            fg_color=PALETA["bg_card"],
            command=self._al_cambiar_slider_duracion
        )
        self.slider_duracion.set(3.0)
        self.slider_duracion.pack(fill="x", padx=18, pady=(0, 12))

        # ----------------- SECCIÓN 2: ACCIONES DE ENTRADA -----------------
        lbl_sec2 = ctk.CTkLabel(
            self.sidebar,
            text="🎙️ CAPTURA Y ENTRADA DE DATOS",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=PALETA["text_secondary"]
        )
        lbl_sec2.pack(anchor="w", padx=18, pady=(10, 4))

        # Botón Iniciar Grabación (Estilo Prominente)
        self.btn_grabar = ctk.CTkButton(
            self.sidebar,
            text="🔴 Iniciar Grabación en Vivo",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=42,
            fg_color=PALETA["red"],
            hover_color="#d90429",
            command=self.iniciar_grabacion_asincrona
        )
        self.btn_grabar.pack(fill="x", padx=18, pady=(4, 8))

        # Barra de progreso para la grabación
        self.barra_progreso = ctk.CTkProgressBar(
            self.sidebar,
            height=6,
            progress_color=PALETA["red"],
            fg_color=PALETA["bg_card"]
        )
        self.barra_progreso.set(0)
        self.barra_progreso.pack(fill="x", padx=18, pady=(0, 10))

        # Botón Cargar Archivo WAV
        self.btn_cargar_wav = ctk.CTkButton(
            self.sidebar,
            text="📁 Cargar Archivo de Audio (WAV)",
            font=ctk.CTkFont(size=13),
            height=36,
            fg_color=PALETA["bg_card"],
            hover_color=PALETA["border"],
            border_width=1,
            border_color=PALETA["border"],
            command=self.cargar_archivo_wav
        )
        self.btn_cargar_wav.pack(fill="x", padx=18, pady=(0, 8))

        # Menú desplegable para generar tonos demostrativos
        lbl_demo = ctk.CTkLabel(self.sidebar, text="Generador de Demos Sintéticos:", font=ctk.CTkFont(size=12))
        lbl_demo.pack(anchor="w", padx=18, pady=(4, 2))

        self.combo_demo = ctk.CTkOptionMenu(
            self.sidebar,
            values=[
                "Flauta Dulce (Do5 ~ 523 Hz, Tono Puro)",
                "Guitarra Acústica (La3 ~ 220 Hz, Armónicos)",
                "Diapasón de Concierto (La4 = 440 Hz)"
            ],
            fg_color=PALETA["bg_card"],
            button_color=PALETA["border"],
            button_hover_color=PALETA["cyan"],
            height=34,
            command=self._al_seleccionar_demo
        )
        self.combo_demo.set("Flauta Dulce (Do5 ~ 523 Hz, Tono Puro)")
        self.combo_demo.pack(fill="x", padx=18, pady=(0, 14))

        # ----------------- SECCIÓN 3: REPRODUCCIÓN Y AUDITORÍA -----------------
        lbl_sec3 = ctk.CTkLabel(
            self.sidebar,
            text="💾 HERRAMIENTAS Y EXPORTACIÓN",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=PALETA["text_secondary"]
        )
        lbl_sec3.pack(anchor="w", padx=18, pady=(10, 4))

        # Botón Reproducir / Detener Audio
        self.btn_reproducir = ctk.CTkButton(
            self.sidebar,
            text="🔊 Reproducir Señal Actual",
            height=36,
            fg_color=PALETA["bg_card"],
            hover_color=PALETA["border"],
            border_width=1,
            border_color=PALETA["border"],
            command=self.alternar_reproduccion_audio
        )
        self.btn_reproducir.pack(fill="x", padx=18, pady=(0, 8))

        # Botón Exportar Gráfica a Alta Resolución (PNG/PDF)
        self.btn_exportar_fig = ctk.CTkButton(
            self.sidebar,
            text="📊 Exportar Gráficas (PNG / PDF)",
            height=36,
            fg_color=PALETA["bg_card"],
            hover_color=PALETA["border"],
            border_width=1,
            border_color=PALETA["border"],
            command=self.exportar_grafica_dialogo
        )
        self.btn_exportar_fig.pack(fill="x", padx=18, pady=(0, 8))

        # Botón Limpiar Gráficas
        self.btn_limpiar = ctk.CTkButton(
            self.sidebar,
            text="🧹 Limpiar Interfaz",
            height=34,
            fg_color="transparent",
            text_color=PALETA["text_secondary"],
            hover_color=PALETA["bg_card"],
            command=self.limpiar_interfaz
        )
        self.btn_limpiar.pack(fill="x", padx=18, pady=(0, 10))

        # Estado en pie de página
        status_frame = ctk.CTkFrame(self.sidebar, fg_color=PALETA["bg_card"], corner_radius=8)
        status_frame.pack(side="bottom", fill="x", padx=18, pady=18)

        self.lbl_estado = ctk.CTkLabel(
            status_frame,
            text="● Sistema Listo",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=PALETA["green"]
        )
        self.lbl_estado.pack(anchor="w", padx=12, pady=8)

    # ==========================================================================
    # CONSTRUCCIÓN DEL PANEL PRINCIPAL (KPI CARDS + CANVAS MATPLOTLIB)
    # ==========================================================================
    def _construir_panel_principal(self):
        self.panel_principal = ctk.CTkFrame(self, fg_color="transparent")
        self.panel_principal.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.panel_principal.grid_columnconfigure(0, weight=1)
        self.panel_principal.grid_rowconfigure(1, weight=1) # El área de gráficas se expande

        # -------------------------------------------------------------
        # FILA 1: DASHBOARD DE PARÁMETROS FÍSICOS (KPI CARDS)
        # -------------------------------------------------------------
        kpi_container = ctk.CTkFrame(self.panel_principal, fg_color="transparent")
        kpi_container.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        # Permitir que las 4 tarjetas se distribuyan uniformemente
        for col in range(4):
            kpi_container.grid_columnconfigure(col, weight=1)

        # 1. Card: Frecuencia Fundamental (f0)
        self.card_f0 = self._crear_tarjeta_kpi(
            kpi_container, 0, "FRECUENCIA FUNDAMENTAL (f₀)", "-- Hz", "Tono dominante", PALETA["cyan"]
        )
        # 2. Card: Periodo Temporal (T)
        self.card_T = self._crear_tarjeta_kpi(
            kpi_container, 1, "PERIODO DE OSCILACIÓN (T)", "-- ms", "T = 1 / f₀", PALETA["green"]
        )
        # 3. Card: Frecuencia Angular (omega)
        self.card_omega = self._crear_tarjeta_kpi(
            kpi_container, 2, "FRECUENCIA ANGULAR (ω)", "-- rad/s", "ω = 2π·f₀", PALETA["purple"]
        )
        # 4. Card: Nota Musical y Afinación
        self.card_nota = self._crear_tarjeta_kpi(
            kpi_container, 3, "NOTA MUSICAL ESTIMADA", "--", "Escala temperada A4", PALETA["yellow"]
        )

        # -------------------------------------------------------------
        # FILA 2: CONTENEDOR DE GRÁFICAS CIENTÍFICAS MATPLOTLIB
        # -------------------------------------------------------------
        self.chart_frame = ctk.CTkFrame(
            self.panel_principal,
            fg_color=PALETA["bg_sidebar"],
            corner_radius=10,
            border_width=1,
            border_color=PALETA["border"]
        )
        self.chart_frame.grid(row=1, column=0, sticky="nsew")
        self.chart_frame.grid_columnconfigure(0, weight=1)
        self.chart_frame.grid_rowconfigure(0, weight=1)

        # Inicialización de la figura Matplotlib con tema Dark
        self._inicializar_figura_matplotlib()

    def _crear_tarjeta_kpi(self, parent, col, titulo, valor_inicial, subtitulo, color_acento):
        card = ctk.CTkFrame(
            parent,
            fg_color=PALETA["bg_card"],
            corner_radius=10,
            border_width=1,
            border_color=PALETA["border"]
        )
        card.grid(row=0, column=col, sticky="ew", padx=6, pady=2)

        lbl_t = ctk.CTkLabel(
            card,
            text=titulo,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=PALETA["text_secondary"]
        )
        lbl_t.pack(anchor="w", padx=14, pady=(10, 2))

        lbl_val = ctk.CTkLabel(
            card,
            text=valor_inicial,
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=color_acento
        )
        lbl_val.pack(anchor="w", padx=14, pady=0)

        lbl_sub = ctk.CTkLabel(
            card,
            text=subtitulo,
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=PALETA["text_secondary"]
        )
        lbl_sub.pack(anchor="w", padx=14, pady=(0, 10))

        return {"card": card, "lbl_valor": lbl_val, "lbl_sub": lbl_sub}

    # ==========================================================================
    # INTEGRACIÓN DE MATPLOTLIB CON TEMA OSCURO CIENTÍFICO
    # ==========================================================================
    def _inicializar_figura_matplotlib(self):
        # Crear Figura y subplots (3 filas x 1 columna)
        plt.style.use("dark_background")
        self.fig = Figure(figsize=(9, 7.5), dpi=100, facecolor=PALETA["bg_sidebar"])
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.07, hspace=0.48)

        self.ax1 = self.fig.add_subplot(3, 1, 1)
        self.ax2 = self.fig.add_subplot(3, 1, 2)
        self.ax3 = self.fig.add_subplot(3, 1, 3)

        self._estilizar_eje(self.ax1, "1. Registro Temporal Completo s(t)", "Tiempo t (s)", "Amplitud [-1, 1]")
        self._estilizar_eje(self.ax2, "2. Zoom Temporal — Medición de Periodo T", "Tiempo relativo Δt (ms)", "Amplitud")
        self._estilizar_eje(self.ax3, "3. Espectro de Frecuencias (FFT)", "Frecuencia f (Hz)", "Magnitud |X(f)|")

        # Embeber Canvas de Matplotlib en Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(8, 0))

        # Barra de herramientas interactiva (Zoom, Pan, Guardar) adaptada al modo oscuro
        toolbar_frame = ctk.CTkFrame(self.chart_frame, fg_color="transparent")
        toolbar_frame.pack(fill="x", padx=8, pady=(2, 6))

        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.config(background=PALETA["bg_sidebar"])
        self.toolbar._message_label.config(background=PALETA["bg_sidebar"], foreground=PALETA["text_secondary"])
        for button in self.toolbar.winfo_children():
            try:
                button.config(background=PALETA["bg_card"])
            except Exception:
                pass
        self.toolbar.update()

    def _estilizar_eje(self, ax, titulo, xlabel, ylabel):
        ax.set_facecolor(PALETA["plot_axis"])
        ax.set_title(titulo, fontsize=10.5, fontweight="bold", color=PALETA["text_primary"], pad=6)
        ax.set_xlabel(xlabel, fontsize=9, color=PALETA["text_secondary"])
        ax.set_ylabel(ylabel, fontsize=9, color=PALETA["text_secondary"])
        ax.tick_params(colors=PALETA["text_secondary"], labelsize=8)
        ax.grid(True, linestyle="--", linewidth=0.6, color=PALETA["border"], alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(PALETA["border"])

    # ==========================================================================
    # LÓGICA DE DETECCIÓN Y GESTIÓN DE HARDWARE DE AUDIO
    # ==========================================================================
    def _actualizar_dispositivos_audio(self):
        try:
            dispositivos = sd.query_devices()
            mics = []
            self.mapa_dispositivos = {}

            for idx, dev in enumerate(dispositivos):
                if dev.get("max_input_channels", 0) > 0:
                    nombre = dev.get("name", f"Dispositivo {idx}").strip()
                    # Sanear caracteres problemáticos en Windows
                    nombre_limpio = nombre.encode("ascii", "ignore").decode("ascii") or f"Dispositivo {idx}"
                    etiqueta = f"[{idx}] {nombre_limpio[:28]}"
                    mics.append(etiqueta)
                    self.mapa_dispositivos[etiqueta] = idx

            if mics:
                self.combo_dispositivos.configure(values=mics)
                # Seleccionar el micrófono predeterminado del sistema si es posible
                def_dev = sd.default.device[0]
                seleccionado = next((m for m in mics if m.startswith(f"[{def_dev}]")), mics[0])
                self.combo_dispositivos.set(seleccionado)
                self.lbl_estado.configure(text=f"● Micrófono listo: {seleccionado[:22]}", text_color=PALETA["green"])
            else:
                self.combo_dispositivos.configure(values=["No se hallaron micrófonos"])
                self.combo_dispositivos.set("No se hallaron micrófonos")
                self.lbl_estado.configure(text="⚠️ Sin micrófono (Use Modo Demo)", text_color=PALETA["yellow"])
        except Exception as e:
            self.combo_dispositivos.configure(values=["Error al detectar"])
            self.lbl_estado.configure(text=f"⚠️ Hardware: {str(e)[:25]}", text_color=PALETA["red"])

    def _al_cambiar_slider_duracion(self, valor):
        self.lbl_dur_valor.configure(text=f"{valor:.1f} s")

    # ==========================================================================
    # MULTITHREADING: CAPTURA DE AUDIO SIN BLOQUEAR LA GUI
    # ==========================================================================
    def iniciar_grabacion_asincrona(self):
        """
        Lanza un hilo de ejecución secundario para la captura síncrona con sounddevice,
        manteniendo la interfaz gráfica 100% responsiva y animando la barra de progreso.
        """
        if self.esta_grabando:
            return

        try:
            fs_str = self.combo_fs.get().replace(" Hz", "").strip()
            fs = int(fs_str)
        except ValueError:
            fs = 44100

        duracion = float(self.slider_duracion.get())

        # Obtener ID del dispositivo seleccionado
        sel_mic = self.combo_dispositivos.get()
        dev_id = self.mapa_dispositivos.get(sel_mic, None)

        self.esta_grabando = True
        self.btn_grabar.configure(
            text="⏹ Grabando señal...",
            fg_color="#6c757d",
            state="disabled"
        )
        self.btn_cargar_wav.configure(state="disabled")
        self.lbl_estado.configure(
            text=f"🎙️ Grabando {duracion:.1f}s por micrófono...",
            text_color=PALETA["red"]
        )
        self.barra_progreso.set(0)

        # Iniciar hilo de captura física
        self.hilo_grabacion = threading.Thread(
            target=self._hilo_captura_worker,
            args=(duracion, fs, dev_id),
            daemon=True
        )
        self.hilo_grabacion.start()

        # Iniciar actualización periódica de la barra de progreso en el hilo principal
        self._animar_barra_progreso(inicio=time.time(), duracion=duracion)

    def _animar_barra_progreso(self, inicio, duracion):
        if not self.esta_grabando:
            return

        transcurrido = time.time() - inicio
        progreso = min(1.0, transcurrido / duracion)
        self.barra_progreso.set(progreso)

        if progreso < 1.0:
            self.after(50, self._animar_barra_progreso, inicio, duracion)

    def _hilo_captura_worker(self, duracion, fs, dev_id):
        """Worker en segundo plano."""
        try:
            # Asegurar directorio de almacenamiento
            os.makedirs("audio_samples", exist_ok=True)
            ruta_wav = "audio_samples/captura_reciente.wav"

            tiempo, senal = capturar_audio(
                duracion=duracion,
                fs=fs,
                canales=1,
                guardar_wav=ruta_wav,
                device=dev_id
            )

            # Notificar de forma segura a la GUI principal mediante after()
            self.after(0, self._al_terminar_grabacion, tiempo, senal, fs, "Grabación de Micrófono")
        except Exception as e:
            self.after(0, self._al_fallar_grabacion, str(e))

    def _al_terminar_grabacion(self, tiempo, senal, fs, titulo):
        self.esta_grabando = False
        self.btn_grabar.configure(
            text="🔴 Iniciar Grabación en Vivo",
            fg_color=PALETA["red"],
            state="normal"
        )
        self.btn_cargar_wav.configure(state="normal")
        self.barra_progreso.set(1.0)
        self.lbl_estado.configure(
            text=f"● Grabación completada ({len(senal)} muestras)",
            text_color=PALETA["green"]
        )

        # Procesar y graficar la señal capturada
        self.procesar_y_renderizar(tiempo, senal, fs, titulo)

    def _al_fallar_grabacion(self, mensaje_error):
        self.esta_grabando = False
        self.btn_grabar.configure(
            text="🔴 Iniciar Grabación en Vivo",
            fg_color=PALETA["red"],
            state="normal"
        )
        self.btn_cargar_wav.configure(state="normal")
        self.barra_progreso.set(0)
        self.lbl_estado.configure(
            text="❌ Error de hardware al grabar",
            text_color=PALETA["red"]
        )

        messagebox.showerror(
            "Error de Audio",
            f"No se pudo acceder al dispositivo de captura:\n{mensaje_error}\n\n"
            f"Sugerencia: Verifique los permisos del micrófono en Windows o "
            f"seleccione otro dispositivo de la lista desplegable."
        )

    # ==========================================================================
    # CARGA DE ARCHIVO WAV Y GENERACIÓN DE DEMOS SINTÉTICOS
    # ==========================================================================
    def cargar_archivo_wav(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo de audio WAV",
            filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
        )
        if not ruta:
            return

        try:
            tiempo, senal, fs = cargar_audio(ruta)
            nombre_base = os.path.splitext(os.path.basename(ruta))[0]
            self.lbl_estado.configure(
                text=f"● Audio cargado: {nombre_base[:25]}",
                text_color=PALETA["green"]
            )
            self.procesar_y_renderizar(tiempo, senal, fs, f"Archivo: {nombre_base}")
        except Exception as e:
            messagebox.showerror("Error al cargar WAV", f"Ocurrió un error al leer el archivo:\n{e}")

    def _al_seleccionar_demo(self, eleccion):
        if "Flauta" in eleccion:
            self.generar_audio_demo("flauta")
        elif "Guitarra" in eleccion:
            self.generar_audio_demo("guitarra")
        elif "Diapasón" in eleccion:
            self.generar_audio_demo("diapason")

    def generar_audio_demo(self, tipo: str):
        """Genera señales modelo para experimentación inmediata sin requerir hardware."""
        fs = 44100
        duracion = float(self.slider_duracion.get())
        t = np.linspace(0.0, duracion, int(duracion * fs), endpoint=False)

        if tipo == "flauta":
            # Flauta: Do5 (523.25 Hz) con fundamental dominante y timbre suave
            f0 = 523.25
            envolvente = np.sin(np.pi * t / duracion)**0.5
            s = (1.0 * np.sin(2 * np.pi * f0 * t) +
                 0.12 * np.sin(2 * np.pi * 2 * f0 * t) +
                 0.03 * np.sin(2 * np.pi * 3 * f0 * t)) * envolvente
            titulo = "Demo: Flauta Dulce (Do5 ~ 523 Hz)"
        elif tipo == "guitarra":
            # Guitarra: La3 (220 Hz) rica en armónicos con decaimiento percusivo
            f0 = 220.0
            envolvente = np.exp(-1.8 * t)
            s = (1.00 * np.sin(2 * np.pi * 1 * f0 * t) +
                 0.65 * np.sin(2 * np.pi * 2 * f0 * t) +
                 0.45 * np.sin(2 * np.pi * 3 * f0 * t) +
                 0.30 * np.sin(2 * np.pi * 4 * f0 * t) +
                 0.18 * np.sin(2 * np.pi * 5 * f0 * t)) * envolvente
            titulo = "Demo: Guitarra Acústica (La3 ~ 220 Hz)"
        else:
            # Diapasón puro: La4 (440 Hz)
            f0 = 440.0
            envolvente = np.exp(-0.8 * t)
            s = np.sin(2 * np.pi * f0 * t) * envolvente
            titulo = "Demo: Diapasón Estándar (La4 = 440 Hz)"

        # Añadir sutil ruido de cuantización realista
        ruido = np.random.normal(0, 0.002, len(t))
        s = s + ruido
        s = s / np.max(np.abs(s))

        self.procesar_y_renderizar(t, s, fs, titulo)
        self.lbl_estado.configure(text=f"● Sintetizado: {titulo[:30]}", text_color=PALETA["green"])

    # ==========================================================================
    # PIPELINE DE PROCESAMIENTO Y RENDERIZADO DE LAS 3 GRÁFICAS
    # ==========================================================================
    def procesar_y_renderizar(self, tiempo: np.ndarray, senal: np.ndarray, fs: int, titulo: str):
        """
        Calcula la FFT, detecta picos armónicos, actualiza las tarjetas de resultados
        y redibuja los 3 paneles en el Canvas de Matplotlib con estética Dark Mode.
        """
        self.tiempo_senal = tiempo
        self.amplitud_senal = senal
        self.fs_actual = fs

        # 1. Cálculo de FFT unilateral con ventana de Hann
        frecs, espectro = calcular_fft(senal, fs, aplicar_ventana=True)

        # 2. Detección automática de la frecuencia fundamental y armónicos
        picos_frecs, picos_amps = detectar_picos(frecs, espectro, f_min=70.0, f_max=4000.0)

        if len(picos_frecs) == 0:
            idx_max = np.argmax(espectro[frecs >= 60])
            f0 = float(frecs[frecs >= 60][idx_max])
        else:
            f0 = float(picos_frecs[0])

        # 3. Parámetros físicos (T, f, omega, nota)
        params = analizar_parametros_fisicos(f0, picos_frecs, picos_amps)

        # 4. Actualizar tarjetas KPI numéricas
        self.card_f0["lbl_valor"].configure(text=f"{f0:.2f} Hz")
        self.card_f0["lbl_sub"].configure(text="Pico dominante")

        self.card_T["lbl_valor"].configure(text=f"{params['periodo_ms']:.3f} ms")
        self.card_T["lbl_sub"].configure(text=f"{params['periodo_s']:.6f} s")

        self.card_omega["lbl_valor"].configure(text=f"{params['frecuencia_angular_rad_s']:.2f} rad/s")
        self.card_omega["lbl_sub"].configure(text="2π · f₀")

        desv = params["desviacion_cents"]
        signo = "+" if desv >= 0 else ""
        self.card_nota["lbl_valor"].configure(text=f"{params['nota_estimada']}")
        self.card_nota["lbl_sub"].configure(text=f"Afinación: {signo}{desv:.1f} cents")

        # 5. RENDERIZADO DE LAS 3 GRÁFICAS CIENTÍFICAS
        self._dibujar_graficas(tiempo, senal, frecs, espectro, picos_frecs, picos_amps, params, titulo)

    def _dibujar_graficas(self, tiempo, senal, frecs, espectro, picos_frecs, picos_amps, params, titulo):
        T = params["periodo_s"]
        f0 = params["frecuencia_fundamental_hz"]

        # Limpiar ejes anteriores
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Re-estilizar fondos y etiquetas
        self._estilizar_eje(self.ax1, f"1. Registro Temporal Completo — {titulo}", "Tiempo t (s)", "Amplitud [-1, 1]")
        self._estilizar_eje(self.ax2, "2. Zoom Temporal en Régimen Estacionario (Medición de Periodo T)", "Tiempo relativo Δt (ms)", "Amplitud")
        self._estilizar_eje(self.ax3, "3. Espectro de Frecuencias (FFT Unilateral) y Detección de Armónicos", "Frecuencia f (Hz)", "Magnitud |X(f)|")

        # -------------------------------------------------------------
        # PANEL 1: SEÑAL TEMPORAL COMPLETA
        # -------------------------------------------------------------
        self.ax1.plot(tiempo, senal, color=PALETA["cyan"], linewidth=0.75, alpha=0.9, label="Amplitud s(t)")
        self.ax1.set_ylim(-1.18, 1.18)

        # Región de extracción para el Zoom
        duracion = tiempo[-1]
        t_centro = duracion * 0.5
        n_ciclos = 5
        dur_zoom = n_ciclos * T if (T and T > 0) else 0.02
        t_ini_z = max(0.0, t_centro - dur_zoom / 2.0)
        t_fin_z = min(duracion, t_centro + dur_zoom / 2.0)

        self.ax1.axvspan(t_ini_z, t_fin_z, color=PALETA["yellow"], alpha=0.25, label=f"Zona de Zoom ({n_ciclos} ciclos)")
        self.ax1.legend(loc="upper right", facecolor=PALETA["bg_card"], edgecolor=PALETA["border"], fontsize=8)

        # -------------------------------------------------------------
        # PANEL 2: ZOOM TEMPORAL (MEDICIÓN DIRECTA DE T)
        # -------------------------------------------------------------
        mascara_z = (tiempo >= t_ini_z) & (tiempo <= t_fin_z)
        t_z = tiempo[mascara_z]
        s_z = senal[mascara_z]
        t_z_ms = (t_z - t_z[0]) * 1000.0 if len(t_z) > 0 else np.array([])

        if len(t_z_ms) > 0:
            self.ax2.plot(
                t_z_ms, s_z,
                color=PALETA["red"],
                marker="o",
                markersize=3.0,
                linewidth=1.4,
                alpha=0.95,
                label="Onda acústica muestreada"
            )
            self.ax2.set_ylim(-1.18, 1.18)

            # Cota visual con doble flecha para el periodo T
            T_ms = T * 1000.0
            if T_ms > 0 and len(t_z_ms) > 1 and t_z_ms[-1] >= 2 * T_ms:
                # Dibujar cota en y = 0.82
                cota_x1 = 0.5 * T_ms
                cota_x2 = 1.5 * T_ms
                self.ax2.annotate(
                    "",
                    xy=(cota_x1, 0.82),
                    xytext=(cota_x2, 0.82),
                    arrowprops=dict(arrowstyle="<->", color=PALETA["green"], lw=2.2)
                )
                self.ax2.text(
                    (cota_x1 + cota_x2) / 2.0,
                    0.90,
                    f"T ≈ {T_ms:.3f} ms",
                    ha="center",
                    va="bottom",
                    color=PALETA["green"],
                    fontweight="bold",
                    fontsize=9.5
                )

            self.ax2.legend(loc="upper right", facecolor=PALETA["bg_card"], edgecolor=PALETA["border"], fontsize=8)

        # -------------------------------------------------------------
        # PANEL 3: ESPECTRO FFT Y PICOS ARMÓNICOS
        # -------------------------------------------------------------
        f_max_vista = 3000.0
        mascara_esp = (frecs >= 0) & (frecs <= f_max_vista)
        f_plot = frecs[mascara_esp]
        esp_plot = espectro[mascara_esp]

        self.ax3.plot(f_plot, esp_plot, color=PALETA["purple"], linewidth=1.3, label="Espectro FFT |X(f)|")
        self.ax3.fill_between(f_plot, esp_plot, color=PALETA["purple"], alpha=0.25)
        self.ax3.set_xlim(0, f_max_vista)

        # Identificar y marcar picos
        if picos_frecs is not None and len(picos_frecs) > 0:
            for idx, (pf, pa) in enumerate(zip(picos_frecs, picos_amps)):
                if pf <= f_max_vista:
                    es_fund = (idx == 0)
                    color_pico = PALETA["cyan"] if es_fund else PALETA["red"]
                    etiqueta = f"f₀={pf:.1f}Hz" if es_fund else f"f{idx+1}={pf:.1f}Hz"

                    self.ax3.plot(pf, pa, "o", color=color_pico, markersize=5.5)
                    self.ax3.annotate(
                        etiqueta,
                        xy=(pf, pa),
                        xytext=(pf, pa + (np.max(esp_plot) * 0.10)),
                        ha="center",
                        fontsize=8,
                        fontweight="bold",
                        color=PALETA["text_primary"],
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=PALETA["bg_card"], edgecolor=color_pico, alpha=0.9),
                        arrowprops=dict(arrowstyle="->", color=color_pico, lw=1.2, shrinkB=4)
                    )

        self.ax3.legend(loc="upper right", facecolor=PALETA["bg_card"], edgecolor=PALETA["border"], fontsize=8)

        # Redibujar en el Canvas
        self.canvas.draw_idle()

    # ==========================================================================
    # REPRODUCCIÓN DE AUDIO (PLAYBACK)
    # ==========================================================================
    def alternar_reproduccion_audio(self):
        if self.amplitud_senal is None or len(self.amplitud_senal) == 0:
            messagebox.showinfo("Sin Audio", "No hay ninguna señal cargada para reproducir.")
            return

        if self.esta_reproduciendo:
            sd.stop()
            self.esta_reproduciendo = False
            self.btn_reproducir.configure(text="🔊 Reproducir Señal Actual", fg_color=PALETA["bg_card"])
            self.lbl_estado.configure(text="● Reproducción detenida", text_color=PALETA["text_secondary"])
        else:
            try:
                sd.play(self.amplitud_senal, self.fs_actual)
                self.esta_reproduciendo = True
                self.btn_reproducir.configure(text="⏹ Detener Reproducción", fg_color=PALETA["yellow"], text_color="#12171f")
                self.lbl_estado.configure(text="🔊 Reproduciendo audio...", text_color=PALETA["yellow"])

                # Programar restablecimiento de botón al terminar duración
                dur_ms = int((len(self.amplitud_senal) / self.fs_actual) * 1000)
                self.after(dur_ms + 100, self._al_terminar_playback)
            except Exception as e:
                messagebox.showerror("Error al reproducir", f"No se pudo reproducir el audio:\n{e}")

    def _al_terminar_playback(self):
        if self.esta_reproduciendo:
            self.esta_reproduciendo = False
            self.btn_reproducir.configure(text="🔊 Reproducir Señal Actual", fg_color=PALETA["bg_card"], text_color=PALETA["text_primary"])
            self.lbl_estado.configure(text="● Reproducción finalizada", text_color=PALETA["green"])

    # ==========================================================================
    # EXPORTACIÓN DE FIGURA CIENTÍFICA (PNG / PDF 300 DPI)
    # ==========================================================================
    def exportar_grafica_dialogo(self):
        if self.amplitud_senal is None:
            messagebox.showwarning("Sin Datos", "Capture o cargue una señal antes de exportar.")
            return

        ruta = filedialog.asksaveasfilename(
            title="Guardar Gráfica Científica",
            defaultextension=".png",
            filetypes=[
                ("Imagen PNG de Alta Resolución (*.png)", "*.png"),
                ("Documento PDF Vectorial (*.pdf)", "*.pdf"),
                ("Gráfico SVG Vectorial (*.svg)", "*.svg")
            ],
            initialfile="analisis_espectral_utb.png"
        )
        if not ruta:
            return

        try:
            self.fig.savefig(ruta, dpi=300, facecolor=self.fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
            messagebox.showinfo("Exportación Exitosa", f"Gráfica exportada a 300 DPI correctamente en:\n{ruta}")
            self.lbl_estado.configure(text=f"● Gráfica guardada: {os.path.basename(ruta)}", text_color=PALETA["green"])
        except Exception as e:
            messagebox.showerror("Error al exportar", f"No se pudo guardar la figura:\n{e}")

    # ==========================================================================
    # RESET Y LIMPIEZA
    # ==========================================================================
    def limpiar_interfaz(self):
        self.tiempo_senal = None
        self.amplitud_senal = None

        self.card_f0["lbl_valor"].configure(text="-- Hz")
        self.card_f0["lbl_sub"].configure(text="Tono dominante")

        self.card_T["lbl_valor"].configure(text="-- ms")
        self.card_T["lbl_sub"].configure(text="T = 1 / f₀")

        self.card_omega["lbl_valor"].configure(text="-- rad/s")
        self.card_omega["lbl_sub"].configure(text="ω = 2π·f₀")

        self.card_nota["lbl_valor"].configure(text="--")
        self.card_nota["lbl_sub"].configure(text="Escala temperada A4")

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self._estilizar_eje(self.ax1, "1. Registro Temporal Completo s(t)", "Tiempo t (s)", "Amplitud [-1, 1]")
        self._estilizar_eje(self.ax2, "2. Zoom Temporal — Medición de Periodo T", "Tiempo relativo Δt (ms)", "Amplitud")
        self._estilizar_eje(self.ax3, "3. Espectro de Frecuencias (FFT)", "Frecuencia f (Hz)", "Magnitud |X(f)|")
        self.canvas.draw_idle()

        self.lbl_estado.configure(text="● Sistema Limpio", text_color=PALETA["green"])


def main():
    """Punto de entrada principal para la GUI."""
    app = InterfazAcusticaUTB()
    app.mainloop()


if __name__ == "__main__":
    main()
