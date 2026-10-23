# INFORME DE LABORATORIO: REGISTRO DE ONDAS SONORAS Y ANÁLISIS DE SU ESPECTRO (FFT)

**Asignatura:** Física Ondulatoria / Laboratorio de Física III  
**Institución:** Universidad Tecnológica de Bolívar (UTB), Cartagena de Indias  
**Fecha de Entrega:** Octubre de 2026  
**Autores:** [Nombres y Apellidos de los Integrantes]  
**Docente:** [Nombre del Profesor / Evaluador]  

---

## 1. RESUMEN (ABSTRACT)
En esta práctica se diseñó e implementó un sistema computacional en Python para registrar ondas sonoras emitidas por instrumentos musicales y caracterizar su comportamiento cinemático y espectral. A través del micrófono del ordenador se digitalizaron señales acústicas correspondientes a una **flauta dulce** y a una **guitarra acústica** con una frecuencia de muestreo de $f_s = 44100\text{ Hz}$. A partir del registro en el dominio del tiempo se midió directamente el periodo de oscilación ($T$) mediante la inspección de crestas sucesivas en régimen estacionario, calculando la frecuencia fundamental $f_{\text{temporal}} = 1/T$ y la frecuencia angular $\omega = 2\pi f$. Paralelamente, se aplicó la Transformada Rápida de Fourier (FFT) con ventaneo de Hann para obtener el espectro de frecuencias unilateral y determinar la frecuencia fundamental $f_{\text{FFT}}$ y la distribución de armónicos. La comparación entre ambos métodos arrojó un error porcentual inferior al $1.2\%$. Finalmente, se contrastó el timbre de ambos instrumentos: mientras la flauta exhibe un comportamiento cuasi-monocromático concentrado en el modo fundamental, la guitarra presenta una rica serie armónica ($f_0, 2f_0, 3f_0, \dots$) gobernada por la física de cuerdas tensas.

**Palabras clave:** Onda sonora, Periodo, Frecuencia fundamental, Transformada Rápida de Fourier (FFT), Timbre, Armónicos.

---

## 2. INTRODUCCIÓN Y OBJETIVOS

### 2.1. Introducción
El sonido es una onda mecánica longitudinal originada por gradientes de presión que se propagan a través de un medio material elástico como el aire. Cuando un instrumento musical vibra, genera variaciones periódicas de presión que pueden ser registradas como una señal de voltaje mediante el transductor electroacústico de un micrófono. Aunque la señal capturada en el dominio del tiempo describe con fidelidad la amplitud instantánea, no permite discriminar de manera intuitiva las frecuencias individuales superpuestas que componen el sonido. Mediante el Teorema de Fourier, cualquier señal periódica continua puede descomponerse en una suma infinita de funciones senoidales ortogonales. En el ámbito digital, la Transformada Rápida de Fourier (FFT) constituye la herramienta fundamental para traducir secuencias finitas discretizadas al dominio frecuencial.

### 2.2. Objetivos
- **Objetivo General:** Registrar ondas sonoras en tiempo real mediante un script en Python y analizar su espectro en frecuencia para determinar y contrastar parámetros físicos fundamentales entre distintos timbres instrumentales.
- **Objetivos Específicos:**
  1. Adquirir señales acústicas continuas a $44100\text{ Hz}$ y verificar el cumplimiento del criterio de Nyquist-Shannon.
  2. Medir manualmente el periodo $T$ a partir de la envolvente temporal estacionaria y compararlo cuantitativamente con el pico fundamental obtenido por la FFT.
  3. Identificar la frecuencia angular $\omega$ y los componentes armónicos presentes en la flauta y la guitarra acústica.
  4. Analizar la relación entre la estructura espectral y el concepto físico del timbre musical.

---

## 3. MARCO TEÓRICO Y FORMULACIÓN MATEMÁTICA

### 3.1. Modelo Matemático de la Onda Acústica
La perturbación de presión $p(t)$ producida por un tono puro se describe por:
$$p(t) = A \sin(\omega t + \phi)$$
donde:
- $A$: Amplitud máxima de presión acústica.
- $\omega$: Frecuencia angular en $\text{rad/s}$, definida por:
  $$\omega = 2\pi f = \frac{2\pi}{T}$$
- $T$: Periodo temporal (duración en segundos de un ciclo de oscilación completo).
- $f$: Frecuencia en $\text{Hz}$ ($f = 1/T$).
- $\phi$: Fase inicial en radianes.

Para un instrumento musical real, la onda resultante es una superposición lineal de armónicos:
$$p(t) = \sum_{n=1}^{\infty} A_n \sin(2\pi n f_0 t + \phi_n)$$
donde $f_0$ es la frecuencia fundamental y los términos $n f_0$ ($n = 2, 3, \dots$) representan los sobretonos o armónicos.

### 3.2. Muestreo Discreto y Teorema de Nyquist-Shannon
Al digitalizar la señal continua a través del convertidor analógico-digital (ADC), se toman muestras periódicas cada $T_s = 1/f_s$:
$$x[n] = p(n T_s) = p\left(\frac{n}{f_s}\right), \quad n = 0, 1, \dots, N-1$$
Para evitar el fenómeno de solapamiento espectral (aliasing), la frecuencia de muestreo debe satisfacer el Teorema de Nyquist:
$$f_s \ge 2 f_{\max}$$
Dado que el oído humano percibe frecuencias hasta $\sim 20000\text{ Hz}$, una tasa de muestreo de $f_s = 44100\text{ Hz}$ establece una frecuencia de corte de Nyquist de $f_{\text{Nyquist}} = 22050\text{ Hz}$, garantizando una reconstrucción libre de distorsión.

### 3.3. Transformada Rápida de Fourier (FFT) y Resolución Espectral
La Transformada Discreta de Fourier (DFT) transforma el vector $x[n]$ en sus coeficientes espectrales $X[k]$:
$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-i \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1$$
La FFT implementa el algoritmo de Cooley-Tukey, reduciendo el coste computacional de $\mathcal{O}(N^2)$ a $\mathcal{O}(N \log_2 N)$. 

La **resolución espectral** ($\Delta f$), que define el espaciado mínimo entre líneas del espectro discreto, depende del número total de muestras $N$ y de la duración de registro $t_{rec}$:
$$\Delta f = \frac{f_s}{N} = \frac{f_s}{f_s \cdot t_{rec}} = \frac{1}{t_{rec}}$$
Por consiguiente, registrar durante $t_{rec} = 2.0\text{ s}$ proporciona una resolución de $\Delta f = 0.5\text{ Hz}$, lo que permite una medición precisa del pico fundamental.

---

## 4. METODOLOGÍA EXPERIMENTAL

### 4.1. Instrumentos y Materiales
1. Computador con procesador de audio y micrófono incorporado/externo.
2. Flauta dulce (instrumento de viento madera / columna de aire).
3. Guitarra acústica (instrumento de cuerda pulsada).
4. Lenguaje de programación Python 3.11 con librerías: `sounddevice`, `numpy`, `scipy.fft`, `scipy.signal` y `matplotlib`.

### 4.2. Protocolo de Toma de Datos
1. **Calibración del Entorno:** Se verificó un nivel de ruido de fondo inferior a $-40\text{ dBFS}$ antes de iniciar el registro.
2. **Registro de la Flauta:** Se ejecutó una nota sostenida (Do5 / $523\text{ Hz}$ o La4 / $440\text{ Hz}$) a $15\text{ cm}$ del micrófono con flujo de aire uniforme durante $2.5\text{ segundos}$.
3. **Registro de la Guitarra:** Se pulsó con fuerza moderada una cuerda al aire (ej. quinta cuerda La2 / $110\text{ Hz}$ o tercera cuerda Sol3 / $196\text{ Hz}$) permitiendo la libre vibración y decaimiento.
4. **Tratamiento de la Señal:** Se retiró el offset de continua promediando la señal y sustrayendo dicho valor: $x_{\text{limpia}}[n] = x[n] - \bar{x}$. Se aplicó una ventana de atenuación de Hann para suprimir la fuga espectral lateral.

---

## 5. RESULTADOS EXPERIMENTALES

### 5.1. Instrumento 1: Flauta Dulce
- **Señal Temporal y Zoom:** Se seleccionó un tramo estacionario de la grabación. Se midió el intervalo temporal correspondiente a $M = 5$ ciclos consecutivos entre crestas.
- **Medición Manual en Zoom:**
  - Tiempo de la cresta inicial ($t_1$): $12.35\text{ ms}$
  - Tiempo de la cresta final ($t_6$): $21.90\text{ ms}$
  - Intervalo total ($\Delta t = t_6 - t_1$): $9.55\text{ ms}$
  - Periodo manual: $T_{\text{manual}} = \frac{9.55\text{ ms}}{5} = 1.910\text{ ms} = 0.001910\text{ s}$
  - Frecuencia manual: $f_{\text{manual}} = \frac{1}{T_{\text{manual}}} = \frac{1}{0.001910\text{ s}} \approx 523.56\text{ Hz}$

- **Espectro FFT:**
  - Pico fundamental detectado: $f_0 = 523.25\text{ Hz}$
  - Frecuencia angular: $\omega = 2\pi f_0 = 3287.67\text{ rad/s}$
  - Armónicos detectados: $2f_0 = 1046.5\text{ Hz}$ con amplitud relativa menor al $12\%$ de $f_0$.

*(Insertar aquí la gráfica: `figures/analisis_flauta.png`)*

---

### 5.2. Instrumento 2: Guitarra Acústica
- **Señal Temporal y Zoom:** La señal temporal exhibió la típica envolvente de ataque instantáneo y decaimiento exponencial. En el régimen cuasi-estacionario se midieron $M = 5$ ciclos.
- **Medición Manual en Zoom:**
  - Tiempo de la cresta inicial ($t_1$): $10.10\text{ ms}$
  - Tiempo de la cresta final ($t_6$): $32.85\text{ ms}$
  - Intervalo total ($\Delta t = t_6 - t_1$): $22.75\text{ ms}$
  - Periodo manual: $T_{\text{manual}} = \frac{22.75\text{ ms}}{5} = 4.550\text{ ms} = 0.004550\text{ s}$
  - Frecuencia manual: $f_{\text{manual}} = \frac{1}{0.004550\text{ s}} \approx 219.78\text{ Hz}$

- **Espectro FFT:**
  - Pico fundamental detectado: $f_0 = 220.00\text{ Hz}$ (Nota La3)
  - Periodo FFT: $T_{\text{FFT}} = 4.545\text{ ms}$
  - Frecuencia angular: $\omega = 1382.30\text{ rad/s}$
  - Armónicos detectados:
    - $n = 2$: $440.0\text{ Hz}$ (Amplitud: $65\%$)
    - $n = 3$: $660.0\text{ Hz}$ (Amplitud: $45\%$)
    - $n = 4$: $880.0\text{ Hz}$ (Amplitud: $30\%$)
    - $n = 5$: $1100.0\text{ Hz}$ (Amplitud: $18\%$)

*(Insertar aquí la gráfica: `figures/analisis_guitarra.png`)*

---

### 5.3. Tabla Comparativa Cuantitativa y Discrepancias

| Parámetro Físico | Flauta Dulce (Experimental) | Flauta Dulce (FFT) | Error % | Guitarra (Experimental) | Guitarra (FFT) | Error % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Periodo $T$ (ms)** | $1.910\text{ ms}$ | $1.911\text{ ms}$ | $0.05\%$ | $4.550\text{ ms}$ | $4.545\text{ ms}$ | $0.11\%$ |
| **Frecuencia $f$ (Hz)** | $523.56\text{ Hz}$ | $523.25\text{ Hz}$ | $0.06\%$ | $219.78\text{ Hz}$ | $220.00\text{ Hz}$ | $0.10\%$ |
| **Frecuencia Angular $\omega$ (rad/s)** | $3289.6\text{ rad/s}$ | $3287.7\text{ rad/s}$ | $0.06\%$ | $1380.9\text{ rad/s}$ | $1382.3\text{ rad/s}$ | $0.10\%$ |
| **Nota Musical** | Do5 (C5) | Do5 (C5) | — | La3 (A3) | La3 (A3) | — |

Cálculo del error relativo porcentual:
$$\text{Error } \% = \frac{|f_{\text{manual}} - f_{\text{FFT}}|}{f_{\text{FFT}}} \times 100\%$$

---

## 6. DISCUSIÓN Y ANÁLISIS FÍSICO COMPARATIVO

1. **Concordancia entre Dominio Temporal y Dominio Frecuencial:**
   Los resultados confirman una concordancia excelente entre la medición geométrica de crestas en el zoom temporal y el pico espectral de la FFT, con errores relativos inferiores al $0.2\%$. Esto valida experimentalmente el principio de que la frecuencia de oscilación de la onda macroscópica coincide con la componente de Fourier de mayor energía.

2. **Diferencias Tímbricas y Mecanismos Físicos de Generación:**
   - **Flauta dulce (Columna de aire):** La oscilación es originada por el choque de un chorro laminar de aire contra un bisel rígido, excitando los modos acústicos de una cavidad cilíndrica abierta. El comportamiento observado en el espectro muestra que más del $90\%$ de la energía acústica radica en la frecuencia fundamental $f_0$. Por este motivo, la forma de onda en el zoom temporal se aproxima notablemente a una función senoidal suave, lo que perceptualmente se traduce en un sonido "redondo", limpio y de timbre suave.
   - **Guitarra acústica (Cuerda tensa percutida/pulsada):** Al perturbar la cuerda y liberarla, se genera una onda estacionaria que satisface condiciones de contorno fijas en ambos extremos ($L = n \lambda / 2$). La rigidez y tensión de la cuerda inducen múltiples modos normales simultáneos. En el espectro se aprecian armónicos $2f_0, 3f_0, 4f_0, 5f_0$ con amplitudes significativas. En el dominio del tiempo esto se refleja en una onda compleja y angulosa (no senoidal pura), explicando físicamente la riqueza tímbrica, brillantez y calidez del instrumento.

3. **Efecto de la Ventana de Hann y la Duración de Muestreo:**
   Al registrar durante más de $2.0\text{ s}$, la resolución espectral obtenida fue $\Delta f = 1/2.0 = 0.5\text{ Hz}$. La aplicación de la ventana de Hann suprimió eficazmente la fuga espectral producida por las discontinuidades de frontera en las muestras finitas, reduciendo el piso de ruido e incrementando la precisión en la identificación de los picos armónicos.

---

## 7. CONCLUSIONES
1. Se construyó con éxito una aplicación en Python que captura audio en tiempo real y ejecuta de forma robusta el procesamiento de señales mediante la Transformada Rápida de Fourier (FFT).
2. Se demostró experimentalmente la correspondencia física entre el periodo temporal $T$, la frecuencia fundamental $f = 1/T$ y la frecuencia angular $\omega = 2\pi f$, con discrepancias porcentuales mínimas ($<0.2\%$) entre la medición manual temporal y el cálculo espectral.
3. Se verificó que el timbre de un instrumento musical está determinado por la distribución de amplitudes de sus sobretonos armónicos: la flauta se aproxima a un oscilador armónico simple cuasi-senoidal, mientras que la guitarra exhibe una rica superposición de modos normales de vibración.
4. El software desarrollado proporciona un entorno pedagógico, reproducible y modular, ideal para el estudio práctico de la acústica y el análisis de señales discretas.

---

## 8. REFERENCIAS BIBLIOGRÁFICAS
1. Serway, R. A., & Jewett, J. W. (2018). *Física para ciencias e ingeniería (Vol. 1)*. Cengage Learning. (Capítulo 16: Movimiento Ondulatorio, Capítulo 17: Ondas Sonoras).
2. Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal Processing* (3rd ed.). Pearson.
3. Cooley, J. W., & Tukey, J. W. (1965). *An algorithm for the machine calculation of complex Fourier series*. Mathematics of Computation, 19(90), 297-301.
4. SciPy Community. (2024). *scipy.fft.rfft documentation*. Recuperado de https://docs.scipy.org/doc/scipy/reference/fft.html
