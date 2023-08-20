import requests
import json


def translate_deeplx(text, source, target):
    """
    You can use DeepL or Google Translate to translate the text
    DeepL can translate more casual text in Japanese
    DeepLx is a free and open-source DeepL API, i use this because DeepL Pro is not available in my country
    but sometimes i'm facing request limit, so i use Google Translate as a backup
    """
    # import sys
    # sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

    url = "http://localhost:1188/translate"
    headers = {"Content-Type": "application/json"}

    # define the parameters for the translation request
    params = {
        "text": text,
        "source_lang": source,
        "target_lang": target
    }

    # convert the parameters to a JSON string
    payload = json.dumps(params)

    # send the POST request with the JSON payload
    response = requests.post(url, headers=headers, data=payload)

    # get the response data as a JSON object
    data = response.json()

    # extract the translated text from the response
    translated_text = data['data']

    return translated_text


def translate_google(text, source, target):
    import googletrans
    translator = googletrans.Translator()
    result = translator.translate(text, src=source, dest=target)
    return result.text
    # except Exception as e:
    #     print(e)
    #     return


def detect_google(text):
    import googletrans
    try:
        translator = googletrans.Translator()
        result = translator.detect(text)
        return result.lang.upper()
    except:
        print("Error detect")
        return


def translate_lm(text, device):
    from Conversation.translation.pipeline import Translate
    # todo to use open source
    print("Translation enabled!")
    print("Loading machine translation model...")
    translator = Translate(device,
                           language="jpn_Jpan")  # initialize translator #todo **tt fix translation
    print("Translation disabled!")
    print("Proceeding... wtih pure english conversation")
    txt = translator.translate(text)  # translate to [language] if translation is enabled
    print("translated\n" + txt)


if __name__ == "__main__":
    text = "I'm Pina"
    # source = translate_deeplx(text, "EN", "JA")
    source = translate_google(text, "en", "ja")
    print(source)
