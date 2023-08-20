from AIVoifu.tts import tts
from AIVoifu.voice_conversion import vc_inference as vc


class tts_pipeline:
    def __init__(self) -> None:
        print('Loading Waifu Vocal Pipeline...')
        self.cache_root = './audio_cache'
        self.model = tts.auto_tts()
        self.vc_model = vc.vits_vc_inference(force_load_model=False)
        print('Loaded Waifu Vocal Pipeline')

    def tts(self, text, save_path):
        # text to speech
        if not save_path:
            save_path = f'{self.cache_root}/dialog_cache.wav'
        self.model.tts(text, save_path)
        self.vc_model.convert(save_path, 22050, from_file=True, save_path=save_path)
