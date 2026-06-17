import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# Configuracion inicial
informacion = "informacion.txt"
ruta_origen = Path("audios")
ruta_salida = Path("audios_clips")
duracion_objetivo = 5

# Crear carpeta de salida si no existe
if not os.path.exists(ruta_salida):
    os.mkdir(ruta_salida)
    print(f"Carpeta creada en: {ruta_salida}")

# Listar audios originales
archivos = list(ruta_origen.glob("*.wav"))

with open(informacion, "a") as f:
    for archivo in archivos:
        audio, sr = librosa.load(archivo, sr=None, mono=False)
        print(f"Procesando original: {archivo.name}")

        muestras_por_clip = sr * duracion_objetivo
        contador_clip = 1
        es_estereo = audio.ndim == 2

        if es_estereo:
            muestras_totales = audio.shape[1]
        else:
            muestras_totales = audio.shape[0]

        # Bucle para segmentar el audio
        for inicio in range(0, muestras_totales, muestras_por_clip):
            fin = inicio + muestras_por_clip

            # Aqui hacemos que si el ultimo fragmento no completa los 5 segundos se ignora
            if fin > muestras_totales:
                break

            if es_estereo:
                clip_actual = audio[:, inicio:fin]
                max_amp_izq = np.max(clip_actual[0, :])
                min_amp_izq = np.min(clip_actual[0, :])
                max_amp_der = np.max(clip_actual[1, :])
                min_amp_der = np.min(clip_actual[1, :])
                clip_guardar = clip_actual.T
            else:
                clip_actual = audio[inicio:fin]
                max_amp_izq = np.max(clip_actual)
                min_amp_izq = np.min(clip_actual)
                max_amp_der = "N/A"
                min_amp_der = "N/A"
                clip_guardar = clip_actual

            # Guardar el clip de 5 segundos
            nombre_clip = ruta_salida / f"{archivo.stem}_clip_{contador_clip}.wav"
            sf.write(nombre_clip, clip_guardar, sr)

            print(f"  -> Guardado {nombre_clip.name}")
            contador_clip += 1
