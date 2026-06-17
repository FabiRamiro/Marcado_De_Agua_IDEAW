import librosa
import numpy as np
from pathlib import Path

# Configuracion inicial
ruta_clips = Path("audios_clips")
tamaño_bloque = 1024
salto = 512  # Traslape del 50% (1024 / 2)

archivos_clips = list(ruta_clips.glob("*.wav"))
print(f"Se encontraron {len(archivos_clips)} clips para cargar.\n")

for ruta_clip in archivos_clips:
    audio_clip, sr = librosa.load(ruta_clip, sr=None, mono=False)

    print(
        f"Cargado: {ruta_clip.name} | Canales: {audio_clip.ndim} | Muestras: {audio_clip.shape}"
    )

    # =================================================================
    # PROCESAMIENTO DEL ESPECTROGRAMA
    # =================================================================

    # 1. Convertir a mono si es estereo
    if audio_clip.ndim > 1:
        audio_mono = librosa.to_mono(audio_clip)
    else:
        audio_mono = audio_clip

    # 2. Calcular la STFT con los nuevos parametros
    espectrograma = librosa.stft(audio_mono, n_fft=tamaño_bloque, hop_length=salto)

    # 3. Separar la Magnitud de la Fase
    # Nos quedamos con la fase guardada porque se necesitara intacta
    # para reconstruir el audio al final usando ISTFT.
    magnitud, fase = librosa.magphase(espectrograma)

    # 4. Calcular el indice del bin hasta donde se cortara
    frecuencia_corte_hz = 1000
    bin_corte = int((frecuencia_corte_hz * tamaño_bloque) / sr)

    # 5. Extraer solo los bajos (ignorando la fila 0, hasta el bin de corte)
    magnitud_bajos = magnitud[1:bin_corte, :]

    print(f"  -> Magnitudes extraídas. Rango de bins: 1 al {bin_corte}")
    print(f"  -> Forma matriz de bajos: {magnitud_bajos.shape}\n")
