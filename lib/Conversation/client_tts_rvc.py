import asyncio
import datetime
import logging
import os
import time

import edge_tts
import librosa
import torch
from fairseq import checkpoint_utils

from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.Conversation.rmvpe import RMVPE
from lib.Conversation.vc_infer_pipeline import VC
from scipy.io.wavfile import write

logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

limitation = os.getenv("SYSTEM") == "spaces"

config = Config()

edge_output_filename = "audio_cache/edge_output.mp3"

# todo to get voice list
# tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
# tts_voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]
ln = 'jp'
voice_gen = 'female'
base_voice = "ja-JP-NanamiNeural-Female"

model_root = "weights"
models = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
if len(models) == 0:
    raise ValueError("No model found in `weights` folder")
models.sort()


def model_data(model_name):
    # global n_spk, tgt_sr, net_g, vc, cpt, version, index_file
    pth_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".pth")
    ]
    if len(pth_files) == 0:
        raise ValueError(f"No pth file found in {model_root}/{model_name}")
    pth_path = pth_files[0]
    print(f"Loading {pth_path}")
    cpt = torch.load(pth_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError("Unknown version")
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    print("Model loaded")
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    # n_spk = cpt["config"][-3]

    index_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".index")
    ]
    if len(index_files) == 0:
        print("No index file found")
        index_file = ""
    else:
        index_file = index_files[0]
        print(f"Index file found: {index_file}")

    return tgt_sr, net_g, vc, version, index_file, if_f0


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["pretrain/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


print("Loading hubert model...")
hubert_model = load_hubert()
print("Hubert model loaded.")

print("Loading rmvpe model...")
rmvpe_model = RMVPE("pretrain/rmvpe.pt", config.is_half, config.device)
print("rmvpe model loaded.")


class tts_pipeline:
    def __init__(self, model_name,
                 speed,
                 tts_voice,
                 f0_up_key,
                 f0_method,
                 index_rate,
                 protect,
                 filter_radius=3,
                 resample_sr=0,
                 rms_mix_rate=0.25):
        print('Loading Waifu Vocal Pipeline...')

        self.model_name = model_name
        self.speed = speed
        self.tts_voice = tts_voice
        self.f0_up_key = f0_up_key
        self.f0_method = f0_method
        self.index_rate = index_rate
        self.protect = protect
        self.filter_radius = filter_radius
        self.resample_sr = resample_sr
        self.rms_mix_rate = rms_mix_rate

        print("------------------")
        print(datetime.datetime.now())
        print(f"tts_voice: {self.tts_voice}")
        print(f"Model name: {self.model_name}")
        print(f"F0: {self.f0_method}, Key: {self.f0_up_key}, Index: {self.index_rate}, Protect: {self.protect}")

        self.tgt_sr, self.net_g, self.vc, self.version, self.index_file, self.if_f0 = model_data(self.model_name)
        if self.speed >= 0:
            self.speed_str = f"+{self.speed}%"
        else:
            self.speed_str = f"{self.speed}%"

    async def tts(self, text, save_path):
        t0 = time.time()

        await edge_tts.Communicate(
            text, "-".join(self.tts_voice.split("-")[:-1]), rate=self.speed_str
        ).save(edge_output_filename)

        t1 = time.time()
        edge_time = t1 - t0
        audio, sr = librosa.load(edge_output_filename, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")
        print(3)
        if limitation and duration >= 20:
            print("Error: Audio too long")
            return (
                f"Audio should be less than 20 seconds in this huggingface space, but got {duration}s.",
                edge_output_filename,
                None,
            )

        self.f0_up_key = int(self.f0_up_key)

        if not hubert_model:
            load_hubert()
        if self.f0_method == "rmvpe":
            self.vc.model_rmvpe = rmvpe_model
        times = [0, 0, 0]
        audio_opt = self.vc.pipeline(
            hubert_model,
            self.net_g,
            0,
            audio,
            edge_output_filename,
            times,
            self.f0_up_key,
            self.f0_method,
            self.index_file,
            # file_big_npy,
            self.index_rate,
            self.if_f0,
            self.filter_radius,
            self.tgt_sr,
            self.resample_sr,
            self.rms_mix_rate,
            self.version,
            self.protect,
            None,
        )
        if self.tgt_sr != self.resample_sr >= 16000:
            self.tgt_sr = self.resample_sr
        info = f"Success. Time: edge-tts: {edge_time}s, npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        print(info)
        print(type(audio_opt))
        print(audio_opt)
        print(audio)
        print(self.tgt_sr, self.resample_sr)
        if save_path:
            write(save_path, self.tgt_sr, audio_opt)


if __name__ == "__main__":
    # loop = asyncio.get_event_loop_policy().get_event_loop()
    main = tts_pipeline(
        model_name="lisa",
        speed=0,
        tts_voice=base_voice,
        f0_up_key=0.0,
        f0_method="rmvpe",
        index_rate=1,
        protect=0.33,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.25
    )
    while True:
        tts_text = input("start: ")
        asyncio.run(main.tts(tts_text, save_path=f'./audio_cache/dialog_cache.wav'))
    # try:
    #     loop.run_until_complete(main)
    # finally:
    #     loop.close()
