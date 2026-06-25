"""Extract watermark from audio files (Decoder)"""

import numpy
import time
import torch
import tqdm
import warnings
import os
import argparse

from models.ideaw import IDEAW
from data.process import read_resample

warnings.filterwarnings("ignore")


def process_file_extract(ideaw, audio_path, output_txt_path, device):
    with torch.no_grad():
        ideaw.eval()

        try:
            audio_extr, _, _ = read_resample(
                audio_path=audio_path, sr=16000, audio_limit_len=None
            )
        except Exception as e:
            print(f"[IDEAW] Error reading {audio_path}: {e}")
            return

        audio_length = len(audio_extr)
        audio_extr = torch.tensor(audio_extr).to(torch.float32).unsqueeze(0).to(device)

        chunk_size = 16000
        interval_size = 8000
        chunk_num = int(audio_length / (chunk_size + interval_size))

        if chunk_num == 0:
            print(f"File: {os.path.basename(audio_path)} is too short for chunks.")
            return

        extracted_msgs = []
        extracted_lcodes = []

        for i in range(chunk_num):
            start_pos = i * (chunk_size + interval_size)
            chunk = audio_extr[:, start_pos : start_pos + chunk_size]

            mid_stft, extract_lcode = ideaw.extract_lcode(chunk)
            extract_msg = ideaw.extract_msg(mid_stft)

            msg_binary = (extract_msg >= 0.5).int().cpu().numpy().squeeze().tolist()
            lcode_binary = (extract_lcode >= 0.5).int().cpu().numpy().squeeze().tolist()

            if not isinstance(msg_binary, list):
                msg_binary = [msg_binary]
            if not isinstance(lcode_binary, list):
                lcode_binary = [lcode_binary]

            extracted_msgs.append(msg_binary)
            extracted_lcodes.append(lcode_binary)

        print(f"File: {os.path.basename(audio_path)} processed.")

        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Archivo: {os.path.abspath(audio_path)}\n")
            f.write(f"Total de fragmentos (chunks) procesados: {chunk_num}\n\n")

            for i in range(chunk_num):
                msg_str = "".join(map(str, extracted_msgs[i]))
                lcode_str = "".join(map(str, extracted_lcodes[i]))
                f.write(f"--- Fragmento {i+1} ---\n")
                f.write(f"  Codigo de Localizacion (Locate Code) : {lcode_str}\n")
                f.write(f"  Mensaje Extraido (Message)           : {msg_str}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDEAW Extraction (Decoder)")
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta al archivo .wav individual que deseas decodificar",
    )
    parser.add_argument(
        "--output_txt",
        required=True,
        help="Ruta para guardar el archivo .txt con el mensaje extraido",
    )
    parser.add_argument(
        "--config",
        default="models/config.yaml",
        help="Ruta al archivo config del modelo",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Ruta al archivo checkpoint del modelo (.pt) - opcional",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ideaw = IDEAW(args.config, device)
    print("[IDEAW] modelo construido")

    if args.checkpoint and os.path.exists(args.checkpoint):
        ideaw.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"[IDEAW] modelo cargado desde {args.checkpoint}")
    else:
        print("[IDEAW] usando pesos aleatorios (no se cargo checkpoint)")

    print(
        """
#############################################
#####          IDEAW DECODER            #####
#############################################
        """
    )

    process_file_extract(ideaw, args.input, args.output_txt, device)
    print(
        f"[IDEAW] Decodificacion completada. Resultados guardados en {os.path.abspath(args.output_txt)}"
    )
