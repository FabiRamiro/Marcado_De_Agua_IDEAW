# Documentación Completa del Proyecto IDEAW (Marcado de Agua de Audio)

## 1. Introducción y Conceptos Teóricos

El proyecto **IDEAW (Robust Neural Audio Watermarking with Invertible Dual-Embedding)** implementa un sistema de **marcado de agua neuronal para audio**. El objetivo principal es esconder un mensaje (marca de agua) dentro de un archivo de audio de forma que sea imperceptible para el oído humano, pero que pueda ser extraído posteriormente incluso si el audio sufre modificaciones o ataques (como compresión MP3, ruido, filtros, recortes, etc.).

### Conceptos Clave
1. **Redes Neuronales Invertibles (INN - Invertible Neural Networks):** Son arquitecturas diseñadas para que el proceso hacia adelante (embedding o incrustación) y el proceso hacia atrás (extracting o extracción) se realicen utilizando exactamente los mismos pesos en la red. Esto garantiza que no se pierda información en el proceso de incrustación y que la extracción sea teóricamente perfecta si no hay ruido.
2. **Incrustación Dual (Dual-Embedding):** El modelo no solo incrusta un "Mensaje" (la información real que se quiere ocultar), sino también un "Código de Localización" (lcode). En audios largos que pueden ser recortados, el código de localización ayuda a identificar dónde empieza y termina cada fragmento de la marca de agua.
3. **STFT (Short-Time Fourier Transform):** Antes de que el audio pase por la red neuronal, no se procesa como una simple onda de tiempo (amplitud sobre tiempo), sino que se transforma al dominio de la frecuencia mediante STFT. Esto permite manipular magnitudes de frecuencias específicas.
4. **Modificación de Bajas Frecuencias (Tu Aporte):** En lugar de alterar todas las frecuencias del audio, el código actual separa la magnitud de la STFT y **solamente incrusta la marca en las frecuencias bajas** (los primeros 100 bins de frecuencia). Las frecuencias altas y la fase (angle) original del audio se mantienen intactas. Esto permite un control más preciso y asegura que la fase original del audio no se corrompa, mejorando la imperceptibilidad.

---

## 2. Parámetros y Dimensiones del Sistema

El sistema procesa el audio en fragmentos (chunks). Para entender el flujo, es crucial conocer las dimensiones exactas definidas en `config.yaml`:

*   **Frecuencia de Muestreo (Sample Rate):** 16,000 Hz.
*   **Tamaño del Fragmento (Chunk Size):** 16,000 puntos (es decir, 1 segundo de audio por chunk).
*   **Mensaje (msg_bit):** 16 bits (unos y ceros).
*   **Código de Localización (num_lc_bit):** 10 bits.
*   **STFT Parameters:**
    *   `n_fft`: 1000. Esto produce 501 *bins* de frecuencia (de 0 a Nyquist).
    *   `hop_len`: 250. Cantidad de muestras que avanza la ventana. Para 16000 muestras, produce `16000 / 250 + 1 = 65` cuadros temporales (frames).
    *   `win_len`: 1000. Tamaño de la ventana de análisis.

### Flujo de Dimensiones (Las 4 Dimensiones de la Red Neuronal)

La red neuronal invertible (INN) que procesa la información en este código (`Mihnet_s1` y `Mihnet_s2`) espera estrictamente recibir datos en un formato de **4 dimensiones**. En PyTorch, este es el formato estándar para procesar "imágenes" (Lote, Canales, Alto, Ancho), pero aplicado a espectrogramas de audio se traduce de la siguiente manera:

El formato final que exige la red es **`[B, C, T, F]`**, y representan lo siguiente:

1. **`B` (Batch Size / Lote):**
   * Es la cantidad de audios (o fragmentos) procesándose al mismo tiempo. Durante la inferencia (uso normal) suele ser 1. Durante el entrenamiento puede ser 16 o 32, dependiendo de los recursos de la tarjeta gráfica.
2. **`C` (Channels / Canales):**
   * Representa la "profundidad" de la señal. En STFT suele haber 2 canales (Real e Imaginario), pero como tú extrajiste la **magnitud pura**, el audio se quedó temporalmente sin este canal (se volvió un mapa 2D de frecuencias y tiempo). Por eso en el código agregas un "canal falso" al final usando `.unsqueeze(-1)`, dándole un tamaño de 1.
3. **`T` (Time Frames / Cuadros Temporales):**
   * Es la longitud del audio en el dominio del tiempo tras aplicarle STFT. Al tener un fragmento de 16,000 muestras procesado con ventanas que saltan de 250 en 250 (`hop_len`), la división matemática da como resultado **65** columnas (o cuadros) de tiempo.
4. **`F` (Frequency Bins / Contenedores de Frecuencia):**
   * Es la cantidad de frecuencias distintas que se están analizando. Como pusiste el límite (`LIMITE_BAJOS = 100`), solo se pasan las primeras **100** frecuencias.

**¿Cómo evoluciona el tensor línea por línea en tu código?**
1. **Entrada de Audio Base:** 2 dimensiones `[Batch, 16000]`.
2. **Salida Compleja STFT:** 4 dimensiones `[B, 501, 65, 2]`. *(501 frecuencias, 65 tiempos, 2 para real/imaginario).*
3. **`torch.abs()` (Cálculo de Magnitud):** Colapsa a 3 dimensiones `[B, 501, 65]`.
4. **Corte de Bajos (`audio_mag[:, :100, :]`):** Se queda en 3 dimensiones `[B, 100, 65]`.
5. **Ajuste para la Red (`unsqueeze(-1)`):** Sube de nuevo a 4 dimensiones, quedando como `[B, F, T, C]`, es decir `[B, 100, 65, 1]`.
6. **Permutación (`permute(0, 3, 2, 1)`):** La función `enc_dec_1` voltea la matriz porque la red exige el formato `[B, C, T, F]`. Por lo tanto, el tensor final que entra al corazón de la red neuronal es:
   **`[B, 1, 65, 100]`**

### El rol matemático de `.unsqueeze(-1)` y `.squeeze(-1)`

En PyTorch, los tensores son como cajas dentro de cajas (matrices). La función `.unsqueeze()` sirve para **agregar una nueva dimensión de tamaño 1**, mientras que `.squeeze()` hace lo opuesto, **eliminando dimensiones que tengan tamaño 1**. El parámetro `-1` significa "en la última posición".

**¿Por qué lo necesitas en este proyecto?**
Cuando haces `audio_mag = torch.abs(audio_complex)`, pasas de tener un tensor complejo de 4 dimensiones `[B, F, T, 2]` (donde el '2' representaba la parte real e imaginaria) a tener un tensor de magnitudes puras de 3 dimensiones: `[B, F, T]`.

Sin embargo, la red neuronal (`Mihnet_s1`) **está programada para siempre recibir 4 dimensiones**, ya que matemáticamente realiza convoluciones 2D (como las que se usan en imágenes). Una imagen siempre tiene Alto, Ancho, y **Canales de color** (RGB = 3 canales, Escala de grises = 1 canal). 

Como tu espectrograma de magnitud pura ya no tiene canales reales/imaginarios, se convirtió en algo equivalente a una "imagen plana en escala de grises".
Para cumplir con el requisito de la red y decirle "aquí está mi dimensión de canales", usas:
`audio_bajos_in = audio_bajos_mag.unsqueeze(-1)`

Esto toma el tensor de tamaño `[Lote, 100, 65]` y lo convierte a `[Lote, 100, 65, 1]`. Acabas de inventar la cuarta dimensión (el canal falso) sin alterar los datos numéricos internos.

**El proceso inverso (`.squeeze(-1)`):**
Cuando la red neuronal termina de incrustar la marca de agua y devuelve el tensor resultante, lo entrega con esa cuarta dimensión extra que tú le diste: `[Lote, 100, 65, 1]`.
Pero para poder juntar (concatenar) estas frecuencias bajas ya marcadas con las frecuencias altas originales que dejaste intactas, ambas partes deben tener el mismo formato original de 3 dimensiones `[Lote, F, T]`. Por lo tanto, usas:
`wm_bajos_mag = wm_bajos_mag_out.squeeze(-1)`

Esto elimina ese "1" colgado en la última posición, regresando el tensor a su forma matemática de `[Lote, 100, 65]`, dejándolo listo para unirse a las frecuencias altas, multiplicar por la fase y reconstruir el audio.

---

## 3. Desglose del Código: `models/ideaw.py`

Esta es la clase central que orquesta la incrustación y extracción. Analizaremos las funciones clave en las que trabajaste.

### 3.1. `embed_msg(self, audio, msg)` - Incrustación del Mensaje

**Propósito:** Toma el fragmento de audio completo y el mensaje de 16 bits, y oculta el mensaje en las bajas frecuencias del audio.

**Paso a paso del código:**
1.  **Transformación a STFT:** 
    `audio_stft = self.stft(audio)`
    Convierte la onda de 16,000 puntos en un espectrograma.
2.  **Separación de Magnitud y Fase:**
    ```python
    audio_complex = torch.view_as_complex(audio_stft)
    audio_mag = torch.abs(audio_complex) # Magnitud (Amplitud de la frecuencia)
    audio_phase = torch.angle(audio_complex) # Fase (Desplazamiento de la onda)
    ```
    Es crucial guardar la `audio_phase` para poder reconstruir el audio después de modificar la magnitud sin introducir ruidos extraños.
3.  **Separación de Bajas y Altas Frecuencias:**
    ```python
    LIMITE_BAJOS = 100
    audio_bajos_mag = audio_mag[:, :LIMITE_BAJOS, :]
    audio_altos_mag = audio_mag[:, LIMITE_BAJOS:, :]
    ```
4.  **Preparación del Mensaje:**
    El mensaje de 16 bits pasa por una capa lineal (`msg_fc`) que lo expande al tamaño del audio (16,000 puntos), se le aplica STFT, y también se recorta a sus bajas frecuencias (`msg_bajos_mag`).
5.  **Paso por la Red Neuronal (INN):**
    ```python
    audio_bajos_in = audio_bajos_mag.unsqueeze(-1) # Agrega un canal falso para la red
    msg_bajos_in = msg_bajos_mag.unsqueeze(-1)
    wm_bajos_mag_out, _ = self.enc_dec_1(audio_bajos_in, msg_bajos_in, rev=False)
    ```
    Aquí, `enc_dec_1` es la red neuronal. `rev=False` significa que va en dirección "hacia adelante" (incrustando). La red fusiona la información de `msg` dentro de `audio_bajos_mag`.
6.  **Reconstrucción del Espectrograma y el Audio:**
    Se juntan los bajos marcados con los altos originales:
    `wm_mag_completa = torch.cat([wm_bajos_mag, audio_altos_mag], dim=1)`
    Se multiplica por la fase original conservada:
    `wm_audio_complex = wm_mag_completa * torch.exp(1j * audio_phase)`
    Finalmente se usa la Transformada Inversa (`istft`) para regresar a una onda de audio normal.

### 3.2. `extract_msg(self, wm_mid_stft)` - Extracción del Mensaje

**Propósito:** Hacer el proceso inverso. Tomar el espectrograma modificado y pasarlo en reversa por la red para obtener los 16 bits originales.

**Paso a paso del código:**
1.  **Obtención de Magnitudes:** Extrae la magnitud del audio marcado y corta las frecuencias bajas (de 0 a 100).
2.  **Paso Inverso por la Red (INN):**
    ```python
    _, extr_msg_bajos_mag_out = self.enc_dec_1(bajos_marcados_in, aux_signal_in, rev=True)
    ```
    Al pasar `rev=True`, la misma red que antes mezclaba ahora "desmezcla" y recupera la magnitud del mensaje.
3.  **Relleno (Padding) y Reconstrucción:**
    Para que el tensor pueda volver a pasar por el ISTFT (que espera 501 frecuencias), rellenamos las 401 frecuencias altas con ceros (`ceros_altos`). Se asume una fase de 0.
4.  **Decodificación:**
    Se transforma inversamente a onda de tiempo y pasa por la red neuronal lineal inversa (`msg_fc_back`) para reducirlo de vuelta a los 16 bits originales (`clamp(-1, 1)` para asegurar que sean valores binarios lógicos).

### 3.3. `embed_lcode` y `extract_lcode`

El "lcode" (código de localización) sigue un proceso casi idéntico al mensaje, con la diferencia de que el lcode no se incrusta en todo el fragmento de 16,000 puntos.
Según la variable `chunk_ratio = 4`:
El fragmento de audio se divide. El lcode solo se incrusta en la **primera cuarta parte** del audio (`int(self.num_point / self.chunk_ratio)` = 4000 puntos).
El resto de los 12,000 puntos se mantienen sin lcode. Esto permite a la red escanear fragmentos pequeños rápidamente para encontrar el inicio de una marca de agua. El proceso técnico de separación de bajas frecuencias y pase por `enc_dec_2` es análogo al del mensaje.

---

## 4. Archivos Auxiliares y Flujo de Procesamiento General

### `embed_extract.py`
Este script es la demostración en vivo de cómo usar el modelo ya entrenado. 
1. **Lectura y División:** Lee el archivo `.wav`, lo remuestrea a 16kHz si es necesario. Divide el audio en fragmentos de 16,000 puntos con intervalos de 8,000 puntos de espacio libre entre ellos (definido por `interval_size`).
2. **Incrustación en Bucle:** Para cada fragmento de 16,000 puntos:
   - Se llama a `ideaw.embed_msg()`.
   - El resultado pasa a `ideaw.embed_lcode()`.
3. **Guardado:** Se concatenan los pedazos y se guardan como el audio marcado (`_wmd.wav`).
4. **Extracción y Evaluación:** Se lee de nuevo el audio marcado, se pasa por `ideaw.extract_lcode()` y `ideaw.extract_msg()`, y finalmente se comparan los bits extraídos con los originales usando `calc_acc` para obtener el porcentaje de precisión (Accuracy).

### `models/attackLayer.py` (Resiliencia)
En el entrenamiento de este modelo (`train.py`), el audio marcado pasa por capas de ataque simuladas (ruido Gaussiano, recortes, filtros pasa banda) **antes** de intentar ser extraído. Esto fuerza a la red neuronal a aprender formas robustas de ocultar el mensaje de tal manera que sobreviva a estos ataques. Al colocar la marca de agua en las bajas frecuencias, se aprovecha que la audición humana y muchos compresores como MP3 preservan con mucha prioridad las frecuencias bajas (ya que ahí está la estructura de voces, bajos, tambores, etc.).

---

## 5. Guía de Replicación para Principiantes

Si alguien desea replicar tu código o entender cómo hacerlo funcionar desde cero, debe seguir estos pasos lógicos:

1. **Instalación de Entorno:**
   Crear un entorno virtual e instalar las librerías ubicadas en `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
2. **Entender el Config (`models/config.yaml`):**
   Aquí se ajusta si se quiere usar fragmentos más grandes (modificando `num_point` y los valores de STFT).
3. **El Corazón del Modelo (`ideaw.py`):**
   Cualquier programador notará que la magia ocurre al manipular tensores de PyTorch.
   - Si se quiere modificar qué frecuencias se marcan, solo se cambia la variable `LIMITE_BAJOS = 100` a, por ejemplo, `200`.
   - Se debe recordar siempre separar la `magnitud` de la `fase`, modificar **solo la magnitud** en la red neuronal, y luego multiplicar por `torch.exp(1j * fase_original)` para regresar al número complejo antes del ISTFT. 
4. **Ejecutar Pruebas Base (`embed_extract.py`):**
   Para probar que tu código funciona sin entrenarlo desde cero (usando pesos aleatorios o un checkpoint previamente entrenado), solo basta correr:
   ```bash
   python embed_extract.py --input "ruta_a_un_audio.wav"
   ```
   Esto imprimirá el "SNR" (Relación Señal/Ruido que mide la calidad del audio) y la precisión de la extracción.

## Resumen de tu Contribución (Fabian Ramiro)
El código de IDEAW original probablemente modificaba el espectrograma completo. Tus aportes en las regiones marcadas de `ideaw.py` optimizan el proceso logrando:
*   **Aislamiento en Frecuencias:** Solo la banda inferior (100 bins) es manipulada.
*   **Conservación Perfecta de Fase:** La fase se aísla antes del procesamiento y se reintegra al final. Esto reduce drásticamente artefactos audibles (el audio no suena "metálico" ni "robótico").
*   **Relleno Eficiente en Extracción:** Al extraer, simulas correctamente el espectro con ceros para las altas frecuencias de forma que la red neuronal inversa pueda procesar matemáticamente la matriz correcta.
