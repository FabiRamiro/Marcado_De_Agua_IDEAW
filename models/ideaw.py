""" IDEAW
    * Embed & Extract functions
"""

import random
import torch
import torch.nn as nn
import yaml

from models.mihnet import Mihnet_s1, Mihnet_s2
from models.componentNet import Discriminator, BalanceBlock
from models.attackLayer import AttackLayer


class IDEAW(nn.Module):
    def __init__(self, config_path, device):
        super(IDEAW, self).__init__()
        self.load_config(config_path)
        self.hinet_1 = Mihnet_s1(config_path, self.num_inn_1, in_channels=1)  # for embedding msg
        self.hinet_2 = Mihnet_s2(config_path, self.num_inn_2, in_channels=1)  # for embedding lcode
        self.msg_fc = nn.Linear(self.num_bit, self.num_point)
        self.msg_fc_back = nn.Linear(self.num_point, self.num_bit)
        self.lcode_fc = nn.Linear(
            self.num_lc_bit, int(self.num_point / self.chunk_ratio)
        )
        self.lcode_fc_back = nn.Linear(
            int(self.num_point / self.chunk_ratio), self.num_lc_bit
        )
        self.discriminator = Discriminator(config_path)
        self.attack_layer = AttackLayer(config_path, device)
        self.balance_block = BalanceBlock(config_path)

    def forward(self, audio, msg, lcode, robustness, shift):
        audio_wmd1, audio_wmd1_stft = self.embed_msg(audio, msg)
        msg_extr1 = self.extract_msg(audio_wmd1_stft)
        audio_wmd2, audio_wmd2_stft = self.embed_lcode(audio_wmd1, lcode)

        if shift == True:
            host_audio_stft = self.stft(audio)
            audio_wmd2_stft = self.shift(
                host_audio_stft, audio_wmd2_stft, self.extract_stripe
            )
            audio_wmd2 = self.istft(audio_wmd2_stft)

        if robustness == False:
            mid_stft, lcode_extr = self.extract_lcode(audio_wmd2)
            msg_extr2 = self.extract_msg(mid_stft)

        else:  # robustness == True
            # robustness training
            audio_att = self.attack_layer(audio_wmd2, audio)
            audio_att_stft = self.stft(audio_att)
            audio_att_stft = self.balance_block(audio_att_stft)
            mid_stft, lcode_extr = self.extract_lcode(audio_att)
            msg_extr2 = self.extract_msg(mid_stft)

        orig_output = self.discriminator(audio)
        wmd_output = self.discriminator(audio_wmd2)

        return (
            audio_wmd1,
            audio_wmd1_stft,
            audio_wmd2,
            audio_wmd2_stft,
            msg_extr1,
            msg_extr2,
            lcode_extr,
            orig_output,
            wmd_output,
        )

    def load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.win_len = config["IDEAW"]["win_len"]
            self.n_fft = config["IDEAW"]["n_fft"]
            self.hop_len = config["IDEAW"]["hop_len"]
            self.num_inn_1 = config["IDEAW"]["num_inn_1"]
            self.num_inn_2 = config["IDEAW"]["num_inn_2"]
            self.num_bit = config["IDEAW"]["num_bit"]
            self.num_lc_bit = config["IDEAW"]["num_lc_bit"]
            self.num_point = config["IDEAW"]["num_point"]
            self.chunk_ratio = config["IDEAW"]["chunk_ratio"]
            self.extract_stripe = config["IDEAW"]["extract_stripe"]

    def stft(self, data):
        window = torch.hann_window(self.win_len).to(data.device)
        ret = torch.stft(
            input=data,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            window=window,
            return_complex=False,
        )
        return ret  # [B, F, T, C]

    def istft(self, data):
        window = torch.hann_window(self.win_len).to(data.device)
        ret = torch.istft(
            input=data,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            window=window,
            return_complex=False,
        )
        return ret

    #========= EN LO QUE TRABAJE YO (FABIAN RAMIRO) =========
    # INN#1 Embedding & Extracting watermark message
    def embed_msg(self, audio, msg):
        audio_stft = self.stft(audio) #Devuelve: Lote, Frec, Tiempo y real e imaginario

        #Separacion de la magnitud y fase
        #Convertir los numeros a un solo numero complejo
        audio_complex = torch.view_as_complex(audio_stft)

        #Extraer la magnitud y la fase
        audio_mag = torch.abs(audio_complex)
        audio_phase = torch.angle(audio_complex)

        #Definir el limite para los bajos
        LIMITE_BAJOS = 100

        #Separar la magnitud en bajos y altos
        audio_bajos_mag = audio_mag[:,:LIMITE_BAJOS, :]
        audio_altos_mag = audio_mag[:, LIMITE_BAJOS:, :]

        # Construccion del mensaje
        msg_expand = self.msg_fc(msg)
        msg_stft = self.stft(msg_expand)
        msg_complex = torch.view_as_complex(msg_stft)
        msg_mag = torch.abs(msg_complex)

        # Cortar el mensaje en magnitud para que embone en tamaño
        msg_bajos_mag = msg_mag[:, :LIMITE_BAJOS, :]

        # Ya que IDEAW espera un formato de 4 dimensiones [B, F, T, C]
        # Se simulara dando un canal extra al final que para eso servira el .unsqueeze(-1)
        audio_bajos_in = audio_bajos_mag.unsqueeze(-1)
        msg_bajos_in = msg_bajos_mag.unsqueeze(-1)

        # Se manda solo la magnitud de los bajos a la red neuronal
        wm_bajos_mag_out, _ = self.enc_dec_1(audio_bajos_in, msg_bajos_in, rev=False)

        # Quitamos el canal extra que era falso xd
        wm_bajos_mag = wm_bajos_mag_out.squeeze(-1)

        # Se pega la magnitud de los bajos marcada, con los altos intactos
        wm_mag_completa = torch.cat([wm_bajos_mag, audio_altos_mag], dim=1)

        # RECONSTRUIMOS CON LA FASE INTACTA
        # Se juntara la nueva magnitud marcada con la fase original
        wm_audio_complex = wm_mag_completa * torch.exp(1j * audio_phase)

        # Se regresa al formato original que espera STFT
        wm_audio_stft = torch.view_as_real(wm_audio_complex)

        # Transfromar de regreso a onda de sonido
        wm_audio = self.istft(wm_audio_stft)

        return wm_audio, wm_audio_stft
    #========================================================


    #========= EN LO QUE TRABAJE YO (FABIAN RAMIRO) =========
    def extract_msg(self, wm_mid_stft):
        LIMITE_BAJOS = 100

        # Convertir la entrada a complejo y sacar la magnitud
        wm_mid_complex = torch.view_as_complex(wm_mid_stft)
        wm_mid_mag = torch.abs(wm_mid_complex)

        # Extraer solo las magnitudes bajas que es donde esta la marca
        bajos_marcados_mag = wm_mid_mag[:, :LIMITE_BAJOS, :]

        # Poner el canal falso [B, F, T, 1] que necesita la red
        bajos_marcados_in = bajos_marcados_mag.unsqueeze(-1)
        aux_signal_in = bajos_marcados_in

        # Pasar solo las magnitudes bajas a la red para hacer el proceso en Reversa (rev=True)
        _, extr_msg_bajos_mag_out = self.enc_dec_1(bajos_marcados_in, aux_signal_in, rev=True)
        extr_msg_bajos_mag = extr_msg_bajos_mag_out.squeeze(-1)

        # Ahora se va a reconstruir estas magnitudes extraidas del mensaje a como
        # eran originalmente para pasar por ISTFT. Se rellenara todo con ceros (faltan las frecuencias altas y la fase).

        ceros_altos = torch.zeros(
            wm_mid_mag.shape[0], # Lote
            wm_mid_mag.shape[1] - LIMITE_BAJOS, # Cuantas frecuencias de altos faltan
            wm_mid_mag.shape[2] # Tiempo
        ).to(wm_mid_stft.device)

        # Pegamos los ceros
        extr_msg_mag_completa = torch.cat([extr_msg_bajos_mag, ceros_altos], dim=1)

        # Se asume una Fase de 0 porque para la extracción de red Neuronal puede valer
        # nada, o usamos la fase original del audio si estuviera disponible
        # Se simulara un numero complejo Real puro (Fase = 0.
        extr_msg_complex = extr_msg_mag_completa * torch.exp(1j * torch.zeros_like(extr_msg_mag_completa))

        # Regresar a formato [B, F, T, 2]
        extr_msg_expand_stft = torch.view_as_real(extr_msg_complex)

        extr_msg_expand = self.istft(extr_msg_expand_stft)
        extr_msg = self.msg_fc_back(extr_msg_expand).clamp(-1, 1)

        return extr_msg
    #========================================================


    def enc_dec_1(self, audio_stft, msg_stft, rev):
        audio_stft = audio_stft.permute(0, 3, 2, 1)  # [B, C, T, F]
        msg_stft = msg_stft.permute(0, 3, 2, 1)

        audio_stft_, msg_stft_ = self.hinet_1(audio_stft, msg_stft, rev)

        return audio_stft_.permute(0, 3, 2, 1), msg_stft_.permute(0, 3, 2, 1)

    def shift(self, host_audio_stft, wmd_audio_stft, step_size):
        X = random.randint(0, step_size)
        for i in range(X):
            wmd_audio_stft[:, :, i, :] = host_audio_stft[:, :, i, :]
        return wmd_audio_stft

    # INN#2 Embedding & Extracting watermark locating code
    def embed_lcode(self, audio, lcode):
        LIMITE_BAJOS = 100

        lcode_expand = self.lcode_fc(lcode)
        lcode_stft = self.stft(lcode_expand)
        lcode_complex = torch.view_as_complex(lcode_stft)
        lcode_mag = torch.abs(lcode_complex)
        lcode_bajos_mag = lcode_mag[:, :LIMITE_BAJOS, :]
        lcode_bajos_in = lcode_bajos_mag.unsqueeze(-1)

        # l_code will be embedded into the head of the audio
        audio_1_raw = audio[:, : int(self.num_point / self.chunk_ratio)]
        audio_2 = audio[:, int(self.num_point / self.chunk_ratio) :]

        audio_1_stft = self.stft(audio_1_raw)
        audio_1_complex = torch.view_as_complex(audio_1_stft)
        audio_1_mag = torch.abs(audio_1_complex)
        audio_1_phase = torch.angle(audio_1_complex)

        audio_1_bajos_mag = audio_1_mag[:, :LIMITE_BAJOS, :]
        audio_1_altos_mag = audio_1_mag[:, LIMITE_BAJOS:, :]
        audio_1_bajos_in = audio_1_bajos_mag.unsqueeze(-1)

        wm_audio_1_bajos_mag_out, _ = self.enc_dec_2(audio_1_bajos_in, lcode_bajos_in, rev=False)
        wm_audio_1_bajos_mag = wm_audio_1_bajos_mag_out.squeeze(-1)

        wm_audio_1_mag_completa = torch.cat([wm_audio_1_bajos_mag, audio_1_altos_mag], dim=1)
        wm_audio_1_complex = wm_audio_1_mag_completa * torch.exp(1j * audio_1_phase)
        wm_audio_1_stft = torch.view_as_real(wm_audio_1_complex)

        wm_audio_1 = self.istft(wm_audio_1_stft)
        wm_audio = torch.concat([wm_audio_1, audio_2], dim=1)
        wm_audio_stft = self.stft(wm_audio)

        return wm_audio, wm_audio_stft

    def extract_lcode(self, wm_audio):
        LIMITE_BAJOS = 100

        wm_audio_1_raw = wm_audio[:, : int(self.num_point / self.chunk_ratio)]
        wm_audio_2 = wm_audio[:, int(self.num_point / self.chunk_ratio) :]

        wm_audio_1_stft = self.stft(wm_audio_1_raw)
        wm_audio_1_complex = torch.view_as_complex(wm_audio_1_stft)
        wm_audio_1_mag = torch.abs(wm_audio_1_complex)

        bajos_marcados_mag = wm_audio_1_mag[:, :LIMITE_BAJOS, :]
        bajos_marcados_in = bajos_marcados_mag.unsqueeze(-1)
        aux_signal_in = bajos_marcados_in

        mid_bajos_mag_out, extr_lcode_bajos_mag_out = self.enc_dec_2(
            bajos_marcados_in, aux_signal_in, rev=True
        )

        mid_bajos_mag = mid_bajos_mag_out.squeeze(-1)
        extr_lcode_bajos_mag = extr_lcode_bajos_mag_out.squeeze(-1)

    #========= EN LO QUE TRABAJE YO (FABIAN RAMIRO) =========
        # Reconstrucción de mid
        audio_1_altos_mag = wm_audio_1_mag[:, LIMITE_BAJOS:, :]
        mid_mag_completa = torch.cat([mid_bajos_mag, audio_1_altos_mag], dim=1)
        wm_audio_1_phase = torch.angle(wm_audio_1_complex)
        mid_1_complex = mid_mag_completa * torch.exp(1j * wm_audio_1_phase)
        mid_1_stft = torch.view_as_real(mid_1_complex)
        mid_1 = self.istft(mid_1_stft)
        mid = torch.concat([mid_1, wm_audio_2], dim=1)
        mid_stft = self.stft(mid)

        # Reconstrucción de lcode
        ceros_altos = torch.zeros(
            wm_audio_1_mag.shape[0],
            wm_audio_1_mag.shape[1] - LIMITE_BAJOS,
            wm_audio_1_mag.shape[2]
        ).to(wm_audio.device)

        extr_lcode_mag_completa = torch.cat([extr_lcode_bajos_mag, ceros_altos], dim=1)
        extr_lcode_complex = extr_lcode_mag_completa * torch.exp(1j * torch.zeros_like(extr_lcode_mag_completa))
        extr_lcode_expand_stft = torch.view_as_real(extr_lcode_complex)
        extr_lcode_expand = self.istft(extr_lcode_expand_stft)
        extr_lcode = self.lcode_fc_back(extr_lcode_expand).clamp(-1, 1)

        return mid_stft, extr_lcode
    #========================================================

    def enc_dec_2(self, audio_stft, lcode_stft, rev):
        audio_stft = audio_stft.permute(0, 3, 2, 1)  # [B, C, T, F]
        lcode_stft = lcode_stft.permute(0, 3, 2, 1)

        audio_stft_, lcode_stft_ = self.hinet_2(audio_stft, lcode_stft, rev)

        return audio_stft_.permute(0, 3, 2, 1), lcode_stft_.permute(0, 3, 2, 1)
