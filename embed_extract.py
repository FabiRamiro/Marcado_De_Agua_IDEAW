""" Test the trained IDEAW (hard coding, only for specific test)
    * get the marked audio
    * extract the msg and compute ACC
    * ...
"""

import numpy
import time
import torch
import tqdm
import warnings

from scipy.io.wavfile import write

from models.ideaw import IDEAW
from data.process import read_resample
from metrics import calc_acc, signal_noise_ratio

warnings.filterwarnings("ignore")

import numpy
import time
import torch
import tqdm
import warnings
import os
import argparse

from scipy.io.wavfile import write

from models.ideaw import IDEAW
from data.process import read_resample
from metrics import calc_acc, signal_noise_ratio

warnings.filterwarnings("ignore")

def process_file(ideaw, audio_path, output_path, msg_bit, lcode_bit, device):
    # generate msg and lcode
    watermark_msg = torch.randint(0, 2, (1, msg_bit), dtype=torch.float32).to(device)
    locate_code = torch.randint(0, 2, (1, lcode_bit), dtype=torch.float32).to(device)

    chunk_wmd_list = []
    # meta data
    embed_time_cost = 0
    audio_length = 0

    with torch.no_grad():
        ideaw.eval()

        try:
            audio, _, _ = read_resample(
                audio_path=audio_path, sr=16000, audio_limit_len=None
            )
        except Exception as e:
            print(f"[IDEAW] Error reading {audio_path}: {e}")
            return

        audio_length = len(audio)
        audio = torch.tensor(audio).to(torch.float32).unsqueeze(0).to(device)

        start_time = time.time()
        chunk_size = 16000
        interval_size = 8000
        chunk_num = int(audio_length / (chunk_size + interval_size))
        
        end_pos = 0
        if chunk_num > 0:
            for i in range(chunk_num):
                start_pos = i * (chunk_size + interval_size)
                wm_end_pos = start_pos + chunk_size
                end_pos = start_pos + chunk_size + interval_size
                chunk = audio[:, start_pos:wm_end_pos]
                chunk_rest = audio[:, wm_end_pos:end_pos]

                # embed msg/lcode
                audio_wmd1, audio_wmd1_stft = ideaw.embed_msg(chunk, watermark_msg)
                audio_wmd2, _ = ideaw.embed_lcode(audio_wmd1, locate_code)

                # concat watermarked chunk
                chunk_wmd = audio_wmd2.squeeze().cpu().numpy()
                chunk_rest = chunk_rest.squeeze().cpu().numpy()

                chunk_wmd_list.append(chunk_wmd)
                chunk_wmd_list.append(chunk_rest)

        audio_rest = audio[:, end_pos:]
        audio_rest = audio_rest.squeeze().cpu().numpy()
        chunk_wmd_list.append(audio_rest)

        audio_wmd = numpy.concatenate(chunk_wmd_list)
        write(output_path, 16000, audio_wmd)

        end_time = time.time()
        embed_time_cost = end_time - start_time

        # calculate SNR
        SNR = signal_noise_ratio(audio.squeeze().cpu().numpy(), audio_wmd)

        # EXTRACTION
        acc_msg_list = []
        acc_lcode_list = []
        
        audio_extr, _, _ = read_resample(
            audio_path=output_path, sr=16000, audio_limit_len=None
        )
        audio_extr = torch.tensor(audio_extr).to(torch.float32).unsqueeze(0).to(device)

        start_time_extr = time.time()
        for i in range(chunk_num):
            start_pos = i * (chunk_size + interval_size)
            chunk = audio_extr[:, start_pos : start_pos + chunk_size]

            # extract lcode/msg
            mid_stft, extract_lcode = ideaw.extract_lcode(chunk)
            extract_msg = ideaw.extract_msg(mid_stft)

            # compute acc
            acc_lcode = calc_acc(extract_lcode, locate_code, 0.5)
            acc_msg = calc_acc(extract_msg, watermark_msg, 0.5)

            acc_lcode_list.append(acc_lcode.cpu())
            acc_msg_list.append(acc_msg.cpu())

        end_time_extr = time.time()
        extract_time_cost = end_time_extr - start_time_extr

        acc_lcode_all = numpy.array(acc_lcode_list)
        acc_msg_all = numpy.array(acc_msg_list)

        print(f"File: {os.path.basename(audio_path)}")
        print(f"  SNR: {SNR:4f}")
        if chunk_num > 0:
            print(f"  lcode/msg acc: {acc_lcode_all.mean():4f}/{acc_msg_all.mean():4f}")
        else:
            print(f"  Audio too short for chunks.")
        print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDEAW Embedding and Extraction")
    parser.add_argument("--input", default="./audios_clips/audio001.wav", help="Path to a single wav file or a directory of wav files")
    parser.add_argument("--output_dir", default="./watermark", help="Directory to save watermarked files")
    parser.add_argument("--config", default="./models/config.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (optional)")
    parser.add_argument("--msg_bit", type=int, default=16)
    parser.add_argument("--lcode_bit", type=int, default=10)
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model
    ideaw = IDEAW(args.config, device)
    print("[IDEAW] model built")

    if args.checkpoint and os.path.exists(args.checkpoint):
        ideaw.load_state_dict(torch.load(args.checkpoint))
        print(f"[IDEAW] model loaded from {args.checkpoint}")
    else:
        print("[IDEAW] using random weights (no checkpoint loaded)")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(
        """
#############################################
#####               IDEAW               #####
#############################################
        """
    )

    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".wav")]
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        files = []

    for f in tqdm.tqdm(files, desc="Processing files"):
        file_name = os.path.basename(f)
        output_path = os.path.join(args.output_dir, file_name.replace(".wav", "_wmd.wav"))
        process_file(ideaw, f, output_path, args.msg_bit, args.lcode_bit, device)
