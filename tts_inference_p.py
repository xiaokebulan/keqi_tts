import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import pydub
import argparse
import commons
import scipy
# from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
# import gradio as gr
import librosa
import soundfile as sf
import time
from io import BytesIO
import traceback
from datetime import datetime
from flask import request, Flask, Response, render_template,json as json_flask
from flask_cors import CORS
# from flask_sockets import Sockets
from gevent import pywsgi
# from geventwebsocket.handler import WebSocketHandler

from text import text_to_sequence, _clean_text
from zh_normalization import TextNormalizer
import logging
import re
logger = logging.getLogger(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
language_marks = {
    "ja": "[JA]",
    "zh": "[ZH]",
    "en": "[EN]",
    "mix": "",
}
lang = ['ja', 'zh', 'en', 'Mix']

def split_chinese_english_japanese(text):
    # 正则表达式匹配中文、英文单词和日语字符
    pattern = re.compile(r'[\u4e00-\u9fff。，、？！；：“”‘’（）《》【】\[\]『』]+|[a-zA-Z0-9\s\.,!?;:"\'\(\)\[\]\<\>]+|[\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\U00020000-\U0002A6DF]+')
    
    matches = pattern.finditer(text)
    sentences = []
    for match in matches:
        sentences.append(match.group())
    
    return sentences

def matches_only_digits_and_special_chars(s):
    # 构建正则表达式，匹配只包含数字和特殊字符的字符串
    # 特殊字符包括所有非字母字符
    pattern = r'^[\W\d]+$'
    # 使用正则表达式匹配字符串
    return bool(re.match(pattern, s))

def is_chinese(text):
    """判断文本是否为中文"""
    # 正则表达式匹配中文字符
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    # 计算中文字符的数量
    chinese_count = len(chinese_pattern.findall(text))
    return chinese_count > 0

def is_english(text):
    """判断文本是否为英文"""
    # 正则表达式匹配英文字符
    english_pattern = re.compile(r'[a-zA-Z]')
    # 计算英文字符的数量
    english_count = len(english_pattern.findall(text))

    return english_count > 0

def is_japanese(text):
    """判断文本是否为日语"""
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FA5]')
    japanese_count = len(japanese_pattern.findall(text))

    return japanese_count > 0

def detect_language(sentence):
    """检测句子是中文还是英文"""
    if is_chinese(sentence):
        return "zh"
    elif is_english(sentence):
        return "en"
    elif is_japanese(sentence):
        return "ja"
    else:
        return "mix"

def wav_to_mp3(wav, mp3):

    sourcefile = pydub.AudioSegment.from_wav(wav)
    sourcefile.export(mp3, format="mp3")


def save_wav(wav, path, original_fs: int,
                    rate=24000, 
                    volume: float = 1.0,
                    speed: float = 1.0,
                    target_fs: int=0  ):

        if target_fs == 0 or target_fs > original_fs:
            target_fs = original_fs
            wav_tar_fs = wav
     
        else:
            wav_tar_fs = librosa.resample(
                np.squeeze(wav), original_fs, target_fs)
        wav_vol = wav_tar_fs * volume
        # wav_speed = change_speed(wav_vol, speed, target_fs)

        wav_norm = wav_vol * (32767 / max(0.001,np.max(np.abs(wav_vol))))
        scipy.io.wavfile.write(path, target_fs, wav_norm.astype(np.int16))

def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, language, speed, volume, file_path):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        
        speaker_id = 0
        stn_tst = get_text(text, hps, False)
        # print(stn_tst)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        # sf.write(file_path, audio, hps.data.sampling_rate)
        save_wav(audio, file_path, hps.data.sampling_rate)
  
        
    return tts_fn

        


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="./model/G_latest.pth", help="directory to your fine-tuned model")
parser.add_argument("--config_dir", default="./configs/modified_finetune_speaker.json", help="directory to your model config file")
parser.add_argument("--share", default=False, help="make link public (used in colab)")

args = parser.parse_args()
hps = utils.get_hparams_from_file(args.config_dir)


net_g = SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint(args.model_dir, net_g, None)
speaker_ids = hps.speakers
speakers = list(hps.speakers.keys())
# speaker_ids = dict((v, k) for k, v in enumerate(hps.speakers))
# speakers = hps.speakers
tts_fn = create_tts_fn(net_g, hps, speaker_ids)
# vc_fn = create_vc_fn(net_g, hps, speaker_ids)

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/")
# 允许跨越访问

# sockets = Sockets(app)
CORS(app)


@app.route("/ttswav", methods=['GET'])
def ttswav():
   
    text = request.args.get("text", "")
    speed = request.args.get("speed",1.1)
    lang = request.args.get("lang",'mix')
    volume = request.args.get("volume",2)
    try:
        split_sentences = split_chinese_english_japanese(text)
        final = ''
        tm = TextNormalizer()
        for i, sentence in enumerate(split_sentences):
            if(is_chinese(sentence) or matches_only_digits_and_special_chars(sentence)):
                sentence = tm.normalize_sentence(sentence)
          
            final += language_marks[detect_language(sentence)] + sentence + language_marks[detect_language(sentence)]

        start = time.time()
        out = BytesIO()
        tts_fn(final, lang, speed, volume, out)
        
        end = time.time()
        logger.info(f"{text}合成时间：{(round((end - start) * 1000))}ms" )
        return Response(out.getvalue(), mimetype="audio/wav")
    except Exception as e:
        logger.info(f'[{datetime.now()}] tts 错误：{e}')
        traceback.print_exc()
        return json_flask.jsonify(error=6,msg="tts wrong!") 

@app.route("/tts", methods=['GET'])
def tts():
    
    text = request.args.get("text", "")
    speed = request.args.get("speed",1.1)
    lang = request.args.get("lang",'mix')
    volume = request.args.get("volume",2)
    try:
        split_sentences = split_chinese_english_japanese(text)
        final = ''
        tm = TextNormalizer()
    
        for i, sentence in enumerate(split_sentences):
            if(is_chinese(sentence) or matches_only_digits_and_special_chars(sentence)):
                sentence = tm.normalize_sentence(sentence)
           
            final += language_marks[detect_language(sentence)] + sentence + language_marks[detect_language(sentence)]

   
        out = BytesIO()
        out1 = BytesIO()
        start = time.time()
        tts_fn(final, lang, speed, volume, out1)
        wav_to_mp3(out1,out)
        end = time.time()
        logger.info(f"{text}合成时间：{(round((end - start) * 1000))}ms" )

        
        return Response(out.getvalue(), mimetype="audio/mp3")
    except Exception as e:
        logger.info(f'[{datetime.now()}] tts 错误：{e}')
        traceback.print_exc()
        return json_flask.jsonify(error=6,msg="tts wrong!") 

@app.route('/')
def ttshome():
    return render_template("tts.html")
    
if __name__ == '__main__':
    server = pywsgi.WSGIServer(("0.0.0.0",5000),app)
    server.serve_forever()



