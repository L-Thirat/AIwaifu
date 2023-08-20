from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from lib.Conversation.conversation import character_msg_constructor
import romajitable  # temporary use this since It'll blow up our ram if we use Machine Translation Model
import torch
from lib.translation.translate import translate_google
import json
from lib.Conversation.client_tts_rvc import tts_pipeline as rvc_tts_pipeline
import asyncio
from lib.translation.katakana import *

# ---------- Config ----------
f = open('config/init.json', encoding='utf-8')
cfg_init = json.load(f)
src_lang = cfg_init["src_lang"]
tgt_lang = cfg_init["tgt_lang"]
voice_gender = cfg_init["voice_gender"]
if tgt_lang == "ja" and voice_gender == "female":
    base_lang_voice = "ja-JP-NanamiNeural-Female"
tts_trained = cfg_init["tts_trained"]
translation = True if src_lang != tgt_lang else False
base_casual_model = cfg_init["base_casual_model"]
base_casual_full = cfg_init["base_casual_full"]
char_name = cfg_init["char_name"]
user_name = cfg_init["user_name"]
translation_model = cfg_init["translation_model"]

# ---------- load Conversation model ----------
print("Loading casual language model...")
tokenizer = AutoTokenizer.from_pretrained(base_casual_model, use_fast=True)
config = AutoConfig.from_pretrained(base_casual_model, is_decoder=True)
model = AutoModelForCausalLM.from_pretrained(base_casual_model, config=config, )


use_gpu = torch.cuda.is_available()
print("Detecting GPU...")
if use_gpu:
    print("GPU detected!")
    device = torch.device('cuda')
else:
    print("Using CPU...")
    use_gpu = False
    device = torch.device('cpu')


model = model.to(device)
if base_casual_full:
    print("Loading model at full precision...")
else:
    print("Loading model at half precision...")
    model.half()


print('--------Finished!----------')
# --------------------------------------------------
# TTS
voice_model='RVC'
if voice_model == 'RVC':
    main = rvc_tts_pipeline(
        model_name="lisa",
        speed=0,
        tts_voice=base_lang_voice,
        f0_up_key=0.0,
        f0_method="rmvpe",
        index_rate=1,
        protect=0.33,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=0.25
    )
# else:
#     from AIVoifu.client_pipeline import tts_pipeline
#     vocal_pipeline = tts_pipeline()
#     vocal_pipeline.tts(text, save_path=f'./audio_cache/dialog_cache.wav')


# --------- Define Waifu personality ----------
talk = character_msg_constructor(char_name, f"""Species("Elf")
Mind("sexy" + "cute" + "Loving" + "Based as Fuck")
Personality("sexy" + "cute"+ "kind + "Loving" + "Based as Fuck")
Body("160cm tall" + "5 foot 2 inches tall" + "small breasts" + "white" + "slim")
Description("{char_name} is 18 years old girl" + "she love pancake")
Loves("Cats" + "Birds" + "Waterfalls")
Sexual Orientation("Straight" + "Hetero" + "Heterosexual")""")
# ---------------------------------------------


### --- websocket server setup
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# use fast api instead
app = FastAPI()


# do a http server instead
@app.get("/waifuapi")
async def get_waifuapi(command: str, data: str, hist_cache: bool = False):
    if command == "chat":
        msg = data
        # ----------- Create Response --------------------------
        msg = talk.construct_msg(msg, talk.history_loop_cache)  # construct message input and cache History model

        ## ----------- Will move this to server later -------- (16GB ram needed at least)
        inputs = tokenizer(msg, return_tensors='pt')
        if use_gpu:
            inputs = inputs.to(device)
        print("generate output ..\n")
        out = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 80,  # todo must < 280
                             pad_token_id=tokenizer.eos_token_id)
        conversation = tokenizer.decode(out[0])
        print("conversation .. \n" + conversation)
        ## --------------------------------------------------

        ## get conversation in proper format and create history from [last_idx: last_idx+2] conversation
        # talk.split_counter += 0
        print("get_current_converse ..\n")
        current_converse = talk.get_current_converse(conversation)
        print("answer ..\n")  # only print waifu answer since input already show
        print(current_converse)
        if hist_cache:
            talk.history_loop_cache = '\n'.join(current_converse[:2])  # update history for next input message

        # -------------- use machine translation model to translate to japanese and submit to client --------------
        print("cleaning ..\n")
        cleaned_text = talk.clean_emotion_action_text_for_speech(current_converse[1])  # clean text for speech
        cleaned_text = cleaned_text.replace(f"{char_name}: ", "")
        cleaned_text = cleaned_text.replace(f"{char_name} : ", "")
        cleaned_text = cleaned_text.replace("<USER>", user_name)
        cleaned_text = cleaned_text.replace("\"", "")
        emotion_to_express = 'netural'

        if len(cleaned_text) > 2:
            print("cleaned_text\n" + cleaned_text)

            if translation:
                # cleaned_text = katakana_converter(cleaned_text) # todo to fix
                if translation_model == "google":
                    cleaned_text = translate_google(cleaned_text, src_lang, tgt_lang)
                elif translation_model == "lm":
                    from lib.translation.pipeline import Translate
                    tgt_lang_mapping = {
                        "ja": "jpn_Jpan"
                    }
                    translator = Translate(device, tgt_lang_mapping[tgt_lang])
                    cleaned_text = translator.translate(cleaned_text)  # translate to [language] if translation is enabled
                print(cleaned_text)

            # ----------- Waifu Expressing ----------------------- (emotion expressed)
            emotion = talk.emotion_analyze(current_converse[1])  # get emotion from waifu answer (last line)
            print(f'Emotion Log: {emotion}')
            if 'joy' in emotion:
                emotion_to_express = 'happy'

            elif 'anger' in emotion:
                emotion_to_express = 'angry'

            print(f'Emotion to express: {emotion_to_express}')
        await main.tts(cleaned_text, save_path=f'./audio_cache/dialog_cache.wav')

        return JSONResponse(content=f'{emotion_to_express}<split_token>{cleaned_text}')


# @app.get("/waifuapi_story")
# async def get_waifuapi_story(command: str, data: str, hist_cache: bool = False):
#     if command == "story":
#         msg = data
#         # ----------- Create Response --------------------------
#         msg = talk.construct_msg(msg, talk.history_loop_cache)  # construct message input and cache History model
#         ## ----------- Will move this to server later -------- (16GB ram needed at least)
#         inputs = tokenizer(msg, return_tensors='pt')
#         if use_gpu:
#             inputs = inputs.to(device)
#         out = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 100,
#                              pad_token_id=tokenizer.eos_token_id)
#         conversation = tokenizer.decode(out[0])
#         ## --------------------------------------------------
#
#         ## get conversation in proper format and create history from [last_idx: last_idx+2] conversation
#         talk.split_counter += 2
#         current_converse = talk.get_current_converse(conversation)[:talk.split_counter][
#                            talk.split_counter - 2:talk.split_counter]
#         print(conversation)  # only print waifu answer since input already show
#         talk.history_loop_cache = '\n'.join(current_converse)  # update history for next input message
#
#         # -------------- use machine translation model to translate to japanese and submit to client --------------
#         cleaned_text = talk.clean_emotion_action_text_for_speech(current_converse[-1])  # clean text for speech
#
#         translated = ''  # initialize translated text as empty by default
#         if translation:
#             if translation_model == "google":
#                 translated = translate_google(cleaned_text, src_lang, tgt_lang)
#             elif translation_model == "lm":
#                 from lib.translation.pipeline import Translate
#                 tgt_lang_mapping = {
#                     "ja": "jpn_Jpan"
#                 }
#                 translator = Translate(device, tgt_lang_mapping[tgt_lang])
#                 translated = translator.translate(cleaned_text)  # translate to [language] if translation is enabled # todo opensource
#             print(translated)
#
#         return JSONResponse(content=f'{current_converse[-1]}<split_token>{translated}')
#
#     if command == "reset":
#         talk.conversation_history = ''
#         talk.history_loop_cache = ''
#         talk.split_counter = 0
#         return JSONResponse(content='Story reseted...')


if __name__ == "__main__":
    import uvicorn
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 8267
    try:
        s.bind(("localhost", port))
        s.close()
    except socket.error as e:
        print(f"Port {port} is already in use")
        exit()
    uvicorn.run(app, host="0.0.0.0", port=port)
