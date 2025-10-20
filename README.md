# KEQI TTS语音合成项目

本项目是基于VITS（一种融合变分自编码器（VAE）、生成对抗网络（GAN）和标准化流（Normalizing Flows）的端到端TTS模型）开发的语音合成服务端。

**本项目使用的环境：**

 - CUDA 11.3
 - torch 1.9.0+cu111
 - python3.8
 - linux/amd64

## 快速部署

docker pull keshiyong/keqi_tts:2.0.2

docker run -d --name='keqi_tts' --net='bridge' --privileged=true -e 'NVIDIA_DRIVER_CAPABILITIES'='all' -e 'NVIDIA_VISIBLE_DEVICES'='GPU-xxx' -p '5000:5000/tcp' --runtime=nvidia 'keshiyong/keqi_tts:2.0.2'

GPU-xxx替换成自己的

## 欢迎加入知识星球或者QQ群讨论，知识星球里面提供项目的模型文件。

qq群811427872

<img src="static/xing.jpg" width="500">


## 打赏作者

<img src="static/coffee.jpg" width="500">


