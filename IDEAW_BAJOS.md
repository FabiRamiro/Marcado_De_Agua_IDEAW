# Guía Paso a Paso: Marcado de Agua en Bajas Frecuencias (IDEAW)

## 0. Modificación Nueva: Conservar la Fase Intacta

El doctor te ha pedido algo muy específico y común en el procesamiento de audio: **"Dejar la fase intacta y modificar solo las magnitudes"**.

**¿Funciona así actualmente?**
**No.** Actualmente el programa usa la Transformada (STFT) configurada para devolver números complejos separados en sus partes **Reales e Imaginarias** (`return_complex=False`). Cuando la red modifica esa parte Real y la parte Imaginaria, **está destruyendo y alterando tanto la Magnitud como la Fase.**

Para lograr lo que el doctor te pide, la red debe recibir **únicamente la Magnitud**, mientras que tú guardas la **Fase** a un lado de forma segura. Al terminar de marcar la Magnitud, las vuelves a juntar.

---

## 1. La Teoría

El audio normalmente lo escuchamos como una onda a lo largo del tiempo (dominio del tiempo), pero para poder esconder datos sin que el oído humano lo note fácilmente, los sistemas como IDEAW convierten el audio al **dominio de la frecuencia**.

### El Espectrograma y la STFT

La herramienta matemática que hace esto se llama **Transformada de Fourier a Corto Plazo (STFT)**. Al aplicarla, obtenemos un **espectrograma**, el cual nos dice "cuánta energía hay en cada frecuencia (desde los bajos más graves hasta los agudos más chillones) en cada instante de tiempo".

En programación con la librería `PyTorch` que usa tu proyecto, este espectrograma es un bloque de datos (un "Tensor") con 4 dimensiones clave: `[Lote, Frecuencias, Tiempo, Canales]`.

### El Objetivo

Actualmente, IDEAW agarra **todas las frecuencias** y esconde la marca de agua a lo largo de todo el espectrograma completo.
Tú necesitas que **solo toque los bajos**. Para lograr esto, tenemos que "partir" ese bloque de datos, darle a la red neuronal únicamente el bloque de los "bajos", dejar que mezcle la marca, y luego volver a pegarlo con el bloque de los "agudos" originales antes de regresar al sonido normal.

---

## 2. Paso a Paso: El Código

Para hacer los cambios, tienes que abrir el archivo **`models/ideaw.py`**. A continuación, los reemplazos exactos que debes hacer y su explicación:

### Paso 2.1: Modificar `embed_msg` (Esconder el Mensaje)

Busca la función `embed_msg` (cerca de la línea 106).

```python
# CÓDIGO ORIGINAL:
def embed_msg(self, audio, msg):
    audio_stft = self.stft(audio)
    msg_expand = self.msg_fc(msg)
    msg_stft = self.stft(msg_expand)
    wm_audio_stft, _ = self.enc_dec_1(audio_stft, msg_stft, rev=False)
    wm_audio = self.istft(wm_audio_stft)

    return wm_audio, wm_audio_stft
```

**Por qué lo cambiamos:**
Además de cortar las frecuencias bajas, **vamos a extraer la Magnitud y separar la Fase**. La red solo verá la Magnitud de los sonidos bajos.

```python
# NUEVO CÓDIGO A ESCRIBIR:
def embed_msg(self, audio, msg):
    audio_stft = self.stft(audio)  # Devuelve [Lote, Frec, Tiempo, 2 (Real e Imag)]

    # === SEPARACIÓN DE MAGNITUD Y FASE ===
    # Convertimos los números reales/imaginarios a un solo número Complejo matemático
    audio_complex = torch.view_as_complex(audio_stft)

    # Extraemos la Magnitud (Volumen/Energía) y la Fase
    audio_mag = torch.abs(audio_complex)
    audio_phase = torch.angle(audio_complex)

    # 1. Definimos nuestro límite para los "bajos" (Frecuencias bajas)
    LIMITE_BAJOS = 100

    # 2. Separamos la MAGNITUD en Bajos y Altos
    audio_bajos_mag = audio_mag[:, :LIMITE_BAJOS, :]
    audio_altos_mag = audio_mag[:, LIMITE_BAJOS:, :]

    # Hacemos lo mismo con el mensaje
    msg_expand = self.msg_fc(msg)
    msg_stft = self.stft(msg_expand)
    msg_complex = torch.view_as_complex(msg_stft)
    msg_mag = torch.abs(msg_complex)

    # Cortamos el mensaje en magnitud para que embone en tamaño
    msg_bajos_mag = msg_mag[:, :LIMITE_BAJOS, :]

    # Para meterlo a la red IDEAW (INNN), la red espera un formato de 4 dimensiones [B, F, T, C]
    # Lo simulamos dándole un canal Extra al final con `.unsqueeze(-1)`
    audio_bajos_in = audio_bajos_mag.unsqueeze(-1)
    msg_bajos_in = msg_bajos_mag.unsqueeze(-1)

    # 3. Mandamos SOLO LA MAGNITUD DE LOS BAJOS a la red neuronal
    wm_bajos_mag_out, _ = self.enc_dec_1(audio_bajos_in, msg_bajos_in, rev=False)

    # Quitamos el canal extra falso
    wm_bajos_mag = wm_bajos_mag_out.squeeze(-1)

    # 4. Volvemos a pegar la magnitud de los bajos marcada, con los altos intactos
    wm_mag_completa = torch.cat([wm_bajos_mag, audio_altos_mag], dim=1)

    # === RECONSTRUCCIÓN CON LA FASE INTACTA ===
    # Juntamos nuestra nueva Magnitud Marcada con la Fase ORIGINAL que nadie tocó
    wm_audio_complex = wm_mag_completa * torch.exp(1j * audio_phase)

    # Lo regresamos al formato original [Lote, Frec, Tiempo, 2] que espera STFT
    wm_audio_stft = torch.view_as_real(wm_audio_complex)

    # Finalmente transformamos de regreso a onda de sonido
    wm_audio = self.istft(wm_audio_stft)

    return wm_audio, wm_audio_stft
```

### Paso 2.2: Modificar `extract_msg` (Sacar el Mensaje)

Cuando vamos a extraer la marca, el modelo debe leer exactamente la misma región y convertirlo todo otra vez.

Busca la función original `extract_msg` (cerca de la línea 115):

```python
# CÓDIGO ORIGINAL:
def extract_msg(self, wm_mid_stft):
    aux_signal_stft = wm_mid_stft
    _, extr_msg_expand_stft = self.enc_dec_1(wm_mid_stft, aux_signal_stft, rev=True)
    extr_msg_expand = self.istft(extr_msg_expand_stft)
    extr_msg = self.msg_fc_back(extr_msg_expand).clamp(-1, 1)

    return extr_msg
```

**Nuevo código:** Convertimos a complejo, sacamos la magnitud marcada, y de ahí extraemos.

```python
# NUEVO CÓDIGO A ESCRIBIR:
def extract_msg(self, wm_mid_stft):
    LIMITE_BAJOS = 100

    # Convertimos la entrada a Complejo y sacamos Magnitud
    wm_mid_complex = torch.view_as_complex(wm_mid_stft)
    wm_mid_mag = torch.abs(wm_mid_complex)

    # Extraemos solo las magnitudes Bajas que es donde sabemos que está la marca
    bajos_marcados_mag = wm_mid_mag[:, :LIMITE_BAJOS, :]

    # Le ponemos el canal falso [B, F, T, 1] que necesita la red
    bajos_marcados_in = bajos_marcados_mag.unsqueeze(-1)
    aux_signal_in = bajos_marcados_in

    # Pasamos solo las magnitudes bajas a la red para hacer el proceso en Reversa (rev=True)
    _, extr_msg_bajos_mag_out = self.enc_dec_1(bajos_marcados_in, aux_signal_in, rev=True)
    extr_msg_bajos_mag = extr_msg_bajos_mag_out.squeeze(-1)

    # Ahora necesitamos "reconstruir" estas magnitudes extraídas del mensaje a cómo
    # eran originalmente para pasar por ISTFT. Rellenamos todo con Ceros (faltan las Frecuencias altas y la fase).

    ceros_altos = torch.zeros(
        wm_mid_mag.shape[0], # Lote
        wm_mid_mag.shape[1] - LIMITE_BAJOS, # Cuántas frecuencias de altos faltan
        wm_mid_mag.shape[2] # Tiempo
    ).to(wm_mid_stft.device)

    # Pegamos los ceros
    extr_msg_mag_completa = torch.cat([extr_msg_bajos_mag, ceros_altos], dim=1)

    # Asumimos una Fase de "0" porque para la extracción de red Neuronal puede valer
    # nada, o usamos la fase original del audio si estuviera disponible.
    # Aquí simularemos un número complejo Real puro (Fase = 0).
    extr_msg_complex = extr_msg_mag_completa * torch.exp(1j * torch.zeros_like(extr_msg_mag_completa))

    # Regresamos a formato [B, F, T, 2]
    extr_msg_expand_stft = torch.view_as_real(extr_msg_complex)

    extr_msg_expand = self.istft(extr_msg_expand_stft)
    extr_msg = self.msg_fc_back(extr_msg_expand).clamp(-1, 1)

    return extr_msg
```

> **Nota:** La lógica anterior se aplica casi idéntica a `embed_lcode` y `extract_lcode`. Debes buscar dónde dice `audio...stft` o generar `lcode_stft` y partirlos.

---

## 3. El Desafío Final: Ajustar la Red Neuronal

Cuando cortas las frecuencias (Paso 2) **y eliminas un canal** (ya que la magnitud es un solo número por frecuencia, no un par de números Real/Imaginario como antes), estás cambiando la "forma" de los datos que entran a tus modelos de Inteligencia Artificial `enc_dec_1`.

Para resolver esto tendrías que ir a tu archivo **`models/mihnet.py`**.
Dentro de este archivo estrá la construcción de las capas (probablemente Conv2D o Linear). Tendrás que asegurarte de que cuando inicies tu modelo, sepa que ya no va a recibir toda la frecuencia, sino solo `LIMITE_BAJOS` (100). En ese archivo es donde se define la arquitectura profunda. Te recomiendo primero intentar hacer los cambios del Paso 2 y al ejecutar verás el error de dimensiones que nos dictará exactamente cómo reducir `mihnet.py`.

---

## 4. ¿Cómo probarlo y forzar el Error de Dimensiones?

Si ya hiciste los cambios mencionados en `models/ideaw.py`, ahora vamos a ejecutar el proyecto para solo **un audio** y ver exactamente dónde y cómo se rompe (lo cual te dará la pista de cómo y en qué línea modificar `mihnet.py`).

### Paso 1: Evitar el error de "Pesos no encontrados"

Actualmente el código intenta cargar el modelo pre-entrenado. Como cambiaste la arquitectura, el archivo de pesos original ya no encaja (y tal vez ni siquiera tengas uno llamado `checkpoint.pth` en esa ruta).
Entra a tu archivo **`embed_extract.py`** y comenta (ponle un `#` al inicio) la siguiente línea (cerca de la línea 39):

```python
    # ideaw.load_state_dict(torch.load(ckpt_path))
    # print("[IDEAW]model loaded")
```

Al comentarlo, PyTorch iniciará el modelo "en blanco" (con valores al azar) y así el programa correrá hasta estamparse directamente con el cálculo matemático que queremos inspeccionar.

### Paso 2: Tu audio de prueba ya está configurado

En ese mismo archivo (`embed_extract.py`), si revisas la línea 30, ya te había asignado un solo archivo que sacarás de tu base de datos:

```python
audio_path = "/home/osvaldofrb/Documentos/Estadias/IDEAW/audios_clips/audio001.wav"
```

Esto garantiza que sólo se ejecutará para ese archivo de audio concreto. (Asegúrate de que sí exista un `audio001.wav` en esa carpeta, o cámbialo por uno que sí exista).

### Paso 3: Ejecutar

Abre una terminal en VS Code y escribe:

```bash
python embed_extract.py
```

_(Si usas `python3` como tu comando principal, usa `python3 embed_extract.py`)._

**¿Qué va a pasar?**
En la consola verás un mensaje gigante lleno de texto rojo (Traceback). En las últimas líneas te describirá un error parecido a:
`RuntimeError: mat1 and mat2 shapes cannot be multiplied...` o un `Size mismatch`.
Copia ese error gigantesco y me lo pasas. ¡Con eso vamos a poder destripar el archivo `mihnet.py` en el Paso 3!

¡Tú puedes! Modifica primero el `ideaw.py`, entiende el proceso de partir (slicing) y estarás en un excelente camino para ser un as de Python y PyTorch.
