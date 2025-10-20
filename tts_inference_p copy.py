import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
# import gradio as gr
import librosa
import soundfile as sf

from text import text_to_sequence, _clean_text
device = "cuda:0" if torch.cuda.is_available() else "cpu"
language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}
lang = ['日本語', '简体中文', 'English', 'Mix']
def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed,file_path):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        
        speaker_id = 0
        stn_tst = get_text(text, hps, False)
        print(stn_tst)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        sf.write(file_path, audio, hps.data.sampling_rate)

    return tts_fn

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
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
    vc_fn = create_vc_fn(net_g, hps, speaker_ids)

    tts_fn("非常抱歉让您有这样的感受，小奇会尽力改进自己。你可以告诉我哪里可以改进吗？","",'简体中文',1,"out10.wav")
    # app = gr.Blocks()
    # with app:
    #     with gr.Tab("Text-to-Speech"):
    #         with gr.Row():
    #             with gr.Column():
    #                 textbox = gr.TextArea(label="Text",
    #                                       placeholder="Type your sentence here",
    #                                       value="こんにちわ。", elem_id=f"tts-input")
    #                 # select character
    #                 char_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label='character')
    #                 language_dropdown = gr.Dropdown(choices=lang, value=lang[0], label='language')
    #                 duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1,
    #                                             label='速度 Speed')
    #             with gr.Column():
    #                 text_output = gr.Textbox(label="Message")
    #                 audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
    #                 btn = gr.Button("Generate!")
    #                 btn.click(tts_fn,
    #                           inputs=[textbox, char_dropdown, language_dropdown, duration_slider,],
    #                           outputs=[text_output, audio_output])
    #     with gr.Tab("Voice Conversion"):
    #         gr.Markdown("""
    #                         录制或上传声音，并选择要转换的音色。User代表的音色是你自己。
    #         """)
    #         with gr.Column():
    #             record_audio = gr.Audio(label="record your voice", source="microphone")
    #             upload_audio = gr.Audio(label="or upload audio here", source="upload")
    #             source_speaker = gr.Dropdown(choices=speakers, value="User", label="source speaker")
    #             target_speaker = gr.Dropdown(choices=speakers, value=speakers[0], label="target speaker")
    #         with gr.Column():
    #             message_box = gr.Textbox(label="Message")
    #             converted_audio = gr.Audio(label='converted audio')
    #         btn = gr.Button("Convert!")
    #         btn.click(vc_fn, inputs=[source_speaker, target_speaker, record_audio, upload_audio],
    #                   outputs=[message_box, converted_audio])
    # # webbrowser.open("http://0.0.0.0:5000")
    # app.launch(share=False, ip_address="0.0.0.0", port=5002)

