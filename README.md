# 🎵 Registro de Ondas Sonoras y Análisis de su Espectro (FFT)
**Laboratorio de Física Ondulatoria y Procesamiento de Señales**  
*Universidad Tecnológica de Bolívar (UTB)*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Scipy](https://img.shields.io/badge/SciPy-FFT-orange.svg)](https://scipy.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

Este proyecto implementa una aplicación en Python para la captura de audio en tiempo real desde el micrófono, visualización de la señal en el dominio temporal y frecuencial, cálculo de la Transformada Rápida de Fourier (FFT), y extracción rigurosa de parámetros físicos ondulatorios ($T$, $f$, $\omega$ y componentes armónicos) contrastando el timbre acústico de una **flauta** frente a un **segundo instrumento (guitarra/piano)**.

---

## 📐 Fundamento Físico y Matemático

1. **Parámetros Ondulatorios Fundamentales:**
   - **Periodo ($T$):** Tiempo que tarda la perturbación de presión acústica en completar un ciclo oscilatorio ($s$ o $ms$).
   - **Frecuencia ($f$):** Número de oscilaciones por unidad de tiempo:
     $$f = \frac{1}{T} \quad [\text{Hz}]$$
   - **Frecuencia Angular ($\omega$):** Rapidez de cambio de fase:
     $$\omega = 2\pi f = \frac{2\pi}{T} \quad [\text{rad/s}]$$

2. **Transformada Rápida de Fourier (FFT):**
   Para una señal discreta $x[n]$ con $N$ muestras muestreada a una tasa $f_s$ ($44100\text{ Hz}$):
   $$X[k] = \sum_{n=0}^{N-1} x[n] e^{-i \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1$$
   La resolución espectral está gobernada estrictamente por la duración temporal de la ventana de registro $t_{rec}$:
   $$\Delta f = \frac{f_s}{N} = \frac{f_s}{f_s \cdot t_{rec}} = \frac{1}{t_{rec}} \quad [\text{Hz}]$$

---

## 📁 Estructura del Repositorio

```text
UTB-Acustica-Ondas/
│
├── gui_app.py              # Interfaz gráfica moderna (Dark Mode) con CustomTkinter y Matplotlib
├── main.py                 # Orquestador con menú de consola y lanzador de GUI (--gui)
├── audio_recorder.py       # Captura de micrófono con sounddevice y lectura/escritura WAV
├── signal_processing.py    # Algoritmos FFT unilateral, ventaneo Hann y parámetros físicos
├── visualizer.py           # Generador de figuras científicas compuestas (3 subplots)
├── analisis_ondas.py       # Script autónomo con las 4 funciones canónicas solicitadas
├── requirements.txt        # Dependencias de Python requeridas (customtkinter, scipy, etc.)
├── README.md               # Documentación integral del proyecto
│
├── audio_samples/          # Muestras de audio capturadas en formato WAV
│   ├── flauta_demo.wav
│   └── guitarra_demo.wav
│
└── figures/                # Gráficas generadas en alta resolución (300 DPI)
    ├── analisis_flauta.png
    └── analisis_guitarra.png
```

---

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/registro-ondas-espectro-fft.git
cd registro-ondas-espectro-fft
```

### 2. Crear y activar entorno virtual (Recomendado)
- **En Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **En Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instalar librerías requeridas
```bash
pip install -r requirements.txt
```

---

## 💻 Instrucciones de Uso

### Opción A: Interfaz Gráfica Moderna (GUI Dark Mode) — *Recomendada*
Ejecute directamente la interfaz visual:
```bash
python gui_app.py
```
o mediante la bandera del orquestador:
```bash
python main.py --gui
```
**Características de la GUI:**
- 🎛️ Selector dinámico de micrófonos de entrada con detección de hardware.
- 🔴 Grabación asíncrona no bloqueante (multithreading) con barra de progreso animada.
- 📊 Dashboard de métricas físicas en tiempo real ($f_0$, $T$, $\omega$, nota musical y cents).
- 📈 Renderizado de los 3 paneles interactivos en modo oscuro de alta fidelidad.
- 🔊 Reproducción de audio en altavoces con un solo clic.
- 💾 Exportación directa de figuras científicas a 300 DPI (PNG, PDF o SVG).

### Opción B: Menú Interactivo por Consola
Si desea utilizar la versión en terminal:
```bash
python main.py
```
Permite grabar, cargar archivos WAV o generar pruebas sintéticas de flauta y guitarra.

### Opción C: Script Canónico de Rúbrica
Para la ejecución directa del script unificado:
```bash
python analisis_ondas.py
```

---

## 📊 Visualizaciones Generadas

Cada prueba genera una figura compuesta con tres paneles científicos:
1. **Señal Temporal Completa:** Exhibe la envolvente de la señal (ataque, sostenimiento y caída).
2. **Zoom Temporal:** Muestra de 5 a 6 ciclos completos, permitiendo medir de cresta a cresta el periodo $T$.
3. **Espectro FFT:** Gráfica de magnitud con detección automática del pico fundamental $f_0$ y los sobretonos armónicos $n \cdot f_0$.

| Flauta Dulce (Tono Puro) | Guitarra Acústica (Rica en Armónicos) |
|:---:|:---:|
| Predominio absoluto de $f_0$, armónicos casi despreciables. | Múltiples armónicos equiespaciados característicos del timbre de cuerda tensa. |

---

## 👥 Autores
- **Estudiante(s):** [Nombres y Apellidos]
- **Docente:** [Nombre del Profesor de Física]
- **Asignatura:** Física Ondulatoria / Laboratorio de Física III
- **Institución:** Universidad Tecnológica de Bolívar (Cartagena de Indias, Colombia)
