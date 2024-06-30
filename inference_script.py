import os
import time
import random
import yaml
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
from munch import Munch
import phonemizer
import sys

# Path to the custom espeak directory
espeak_path = os.path.join(os.path.dirname(__file__), 'espeak NG')

# Set the environment variable
os.environ['ESPEAK_DATA_PATH'] = espeak_path

if os.path.exists("runtime"):
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Add this directory to sys.path
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

from models import *
from utils import *
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

def load_configurations(config_path):
    return yaml.safe_load(open(config_path))

def load_models(config, device):
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    BERT_path = config.get('PLBERT_dir', False)
    from Utils.PLBERT.util import load_plbert
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]
    
    return model, model_params

def load_pretrained_model(model, model_path):
    params_whole = torch.load(model_path, map_location='cpu')
    params = params_whole['net']
    for key in model:
        if key in params:
            print(f'{key} loaded')
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)
    _ = [model[key].eval() for key in model]

def create_sampler(model):
    return DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask

def preprocess(wave, to_mel, mean, std):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path, model, to_mel, mean, std, device):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio, to_mel, mean, std).to(device)
    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)

def inference(text, ref_s, model, sampler, textclenaer, to_mel, device, model_params, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                         embedding=bert_dur,
                         embedding_scale=embedding_scale,
                         features=ref_s,
                         num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-50]

def main():
    set_seeds()

    config = load_configurations(r"C:\Users\jarod\OneDrive\Desktop\code\MY_GITHUB_UPLOADS\PACKAGES\StyleTTS2\Models\mel\config_ft.yml")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_1 = r"C:\Users\jarod\OneDrive\Desktop\code\MY_GITHUB_UPLOADS\PACKAGES\StyleTTS2\Models\mel\epoch_2nd_00029.pth"
    model_2 = "Models/ray_1_1000_0_0/epoch_2nd_00007.pth"
    comparison = False
    
    model, model_params = load_models(config, device)

    global global_phonemizer
    global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

    sampler = create_sampler(model)
    textclenaer = TextCleaner()
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mean, std = -4, 4

    text = '''My name, is melunuh.  Have you heard of the finger maidens?'''
    reference_dicts = {'696_92939': r"C:\Users\jarod\OneDrive\Desktop\code\MY_GITHUB_UPLOADS\PACKAGES\StyleTTS2\Data\mel\audio\file___1_file___1_segment_10.wav"}
    # willa 28

    noise = torch.randn(1, 1, 256).to(device)
    start = time.time()
    print(device)
    for k, path in reference_dicts.items():
        load_pretrained_model(model, model_path=model_1)
        ref_s = compute_style(path, model, to_mel, mean, std, device)
        
        wav1 = inference(text, ref_s, model, sampler, textclenaer, to_mel, device, model_params, alpha=0.3, beta=0.7, diffusion_steps=30, embedding_scale=1.0)
        rtf = (time.time() - start)
        print(f"RTF = {rtf:5f}")
        print(f"{k} Synthesized:")
        from scipy.io.wavfile import write
        write(f"{k}_synthesized.wav", 24000, wav1)
        
        if comparison:
            load_pretrained_model(model, model_path=model_2)
            ref_s = compute_style(path, model, to_mel, mean, std, device)
            wav2 = inference(text, ref_s, model, sampler, textclenaer, to_mel, device, model_params, alpha=0.3, beta=0.7, diffusion_steps=30, embedding_scale=1.0)
            write(f"{k}_synthesized3.wav", 24000, wav2)

if __name__ == "__main__":
    main()
