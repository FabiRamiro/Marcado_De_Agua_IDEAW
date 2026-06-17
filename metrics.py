""" metrics, optimizers, etc.
"""

import torch
import numpy


""" metric functions
    * BER / ACC
    * SNR
"""


# Bit Error Ratio (BER)
def calc_ber(watermark_decoded_tensor, watermark_tensor, threshold=0.5):
    watermark_decoded_binary = watermark_decoded_tensor >= threshold
    watermark_binary = watermark_tensor >= threshold
    ber_tensor = (
        1 - (watermark_decoded_binary == watermark_binary).to(torch.float32).mean()
    )
    return ber_tensor


# Accuracy (ACC)
def calc_acc(watermark_decoded_tensor, watermark_tensor, threshold=0.5):
    ber = calc_ber(watermark_decoded_tensor, watermark_tensor, threshold)
    return 1 - ber


def to_equal_length(original, audio_watermarked):
    if original.shape != audio_watermarked.shape:
        min_length = min(len(original), len(audio_watermarked))
        original = original[0:min_length]
        audio_watermarked = audio_watermarked[0:min_length]
    assert original.shape == audio_watermarked.shape
    return original, audio_watermarked


def signal_noise_ratio(original, audio_watermarked):
    if torch.is_tensor(original):
        # handle torch tensor
        if original.shape != audio_watermarked.shape:
            min_length = min(original.shape[-1], audio_watermarked.shape[-1])
            original = original[..., :min_length]
            audio_watermarked = audio_watermarked[..., :min_length]
        
        noise_strength = torch.sum((original - audio_watermarked) ** 2)
        if noise_strength == 0:
            return float('inf')
        signal_strength = torch.sum(original**2)
        ratio = signal_strength / (noise_strength + 1e-10)
        return (10 * torch.log10(ratio)).item()
    else:
        # handle numpy array
        original, audio_watermarked = to_equal_length(original, audio_watermarked)
        noise_strength = numpy.sum((original - audio_watermarked) ** 2)
        if noise_strength == 0:
            return numpy.inf
        signal_strength = numpy.sum(original**2)
        ratio = signal_strength / (noise_strength + 1e-10)
        return 10 * numpy.log10(ratio)


# SNR
def batch_signal_noise_ratio(original, audio_watermarked):
    signal = original.detach().cpu().numpy()
    audio_watermarked = audio_watermarked.detach().cpu().numpy()
    tmp_list = []
    for s, awm in zip(signal, audio_watermarked):
        out = signal_noise_ratio(s, awm)
        tmp_list.append(out)
    return numpy.mean(tmp_list)
