print('Initializing... Dependencies')
from lib.Conversation.conversation import character_msg_constructor
from lib.vtube.vtube_studio import Char_control
import romajitable  # temporary use this since It'll blow up our ram if we use Machine Translation Model
import pyaudio
import soundfile as sf
import requests
import logging
import socket
import re
import time
import pytchat
import json
from emoji import demojize

DEBUG = True  # todo clean debug mode
# ---------- Config ----------
f = open('config/twitch_setup.json', encoding='utf-8')
cfg_init = json.load(f)
server = cfg_init["server"]
port = cfg_init["port"]
nickname = cfg_init["nickname"]
token = cfg_init["token"]
user = cfg_init["user"]
channel = cfg_init["channel"]
blacklist = ["Nightbot", "streamelements"]

con = ""
con_now = ""
con_prev = ""
conversation = []


logging.getLogger("requests").setLevel(logging.WARNING)  # make requests logging only important stuff
logging.getLogger("urllib3").setLevel(logging.WARNING)  # make requests logging only important stuff

talk = character_msg_constructor("Lilia", None)  # initialize character_msg_constructor



# ----------- Waifu Vocal Pipeline -----------------------


# initialize Vstudio Waifu Controller
if not DEBUG:
    print('Initializing... Vtube Studio')
    waifu = Char_control(port=8001, plugin_name='MyIsAI', plugin_developer='HRNPH')
    print('Initialized')


# chat api
def chat(msg, reset=False):
    command = 'chat'
    if reset:
        command = 'reset'
    params = {
        'command': f'{command}',
        'data': msg,
    }
    try:
        r = requests.get('http://localhost:8267/waifuapi', params=params)
        return r.text
    except requests.exceptions.ConnectionError as e:
        print('--------- Exception Occured ---------')
        print(
            'if you have run the server on different device, please specify the ip address of the server with the port')
        print('Example: http://192.168.1.112:8267 or leave it blank to use localhost')
        print('***please specify the ip address of the server with the port*** at:')
        print(f'*Line {e.__traceback__.tb_lineno}: {e}')
        print('-------------------------------------')
        exit()


def yt_livechat(video_id):
    global chat

    live = pytchat.create(video_id=video_id)
    while live.is_alive():
        # while True:
        try:
            for c in live.get().sync_items():
                # Ignore chat from the streamer and Nightbot, change this if you want to include the streamer's chat
                if c.author.name in blacklist:
                    continue
                # if not c.message.startswith("!") and c.message.startswith('#'):
                if not c.message.startswith("!"):
                    # Remove emojis from the chat
                    chat_raw = re.sub(r':[^\s]+:', '', c.message)
                    chat_raw = chat_raw.replace('#', '')
                    # chat_author makes the chat look like this: "Nightbot: Hello". So the assistant can respond to the user's name
                    chat = c.author.name + ' berkata ' + chat_raw
                    print(chat)

                time.sleep(1)
        except Exception as e:
            print("Error receiving chat: {0}".format(e))


def twitch_livechat():
    global con
    sock = socket.socket()

    sock.connect((server, port))

    sock.send(f"PASS {token}\n".encode('utf-8'))
    sock.send(f"NICK {nickname}\n".encode('utf-8'))
    sock.send(f"JOIN {channel}\n".encode('utf-8'))

    regex = r":(\w+)!\w+@\w+\.tmi\.twitch\.tv PRIVMSG #\w+ :(.+)"

    while True:
        try:
            resp = sock.recv(2048).decode('utf-8')

            if resp.startswith('PING'):
                sock.send("PONG\n".encode('utf-8'))

            elif user in resp:
                resp = demojize(resp)
                match = re.match(regex, resp)

                username = match.group(1)
                message = match.group(2)

                if username in blacklist:
                    continue
                print("username:", username)
                # con = username + ' said ' + message #todo if username needed
                con = message
                print(con)

        except Exception as e:
            print("Error receiving chat: {0}".format(e))


def preparation():
    global conversation, con_now, con, con_prev

    while True:
        if DEBUG:
            con = str(input("You: "))
        if con:
            print("question:" + con)

            if con.lower() == 'exit':
                print('Stopping...')
                break  # exit prototype

            if con.lower() == 'reset':
                print('Resetting...')
                print(chat('None', reset=True))
                continue  # reset story skip to next loop

            # ----------- Create Response --------------------------
            emo_answer = chat(con).replace("\"", "")  # send message to api
            emo, answer, base_answer = emo_answer.split("<split_token>")
            f = open("text_cache/question.txt", "w")
            f.write(con)
            f.close()
            f = open("text_cache/answer.txt", "w")
            f.write(base_answer)
            f.close()
            print(emo)
            print(answer)
            if len(answer) > 2:

                # ------------------------------------------------------
                print(f'Answer: {answer}')
                if answer.strip().endswith(f'{talk.name}:') or answer.strip() == '':
                    continue  # skip audio processing if the answer is just the name (no talking)

                # ----------- Waifu Talking -----------------------
                # play audio directly from cache
                p = pyaudio.PyAudio()
                data, samplerate = sf.read('./audio_cache/dialog_cache.wav', dtype='float32')
                stream = p.open(format=pyaudio.paFloat32,
                                channels=1,
                                rate=samplerate,
                                output=True)
                stream.write(data.tobytes())
                stream.stop_stream()
                stream.close()

                # --------------------------------------------------
                if emo and not DEBUG:  ## express emotion
                    waifu.express(emo)  # express emotion in Vtube Studio
                # --------------------------------------------------
                # conversation.append(con) # todo for hist mode
                # con_prev = con # todo for hist mode
                con = ''


if DEBUG:
    preparation()
else:
    import threading

    # Threading is used to capture livechat and answer the chat at the same time
    print("To use this mode, make sure to change utils/twitch_config.py to your own config")
    t = threading.Thread(target=preparation)
    t.start()
    twitch_livechat()
    # yt_livechat()