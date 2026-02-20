# speech-to-text-to-speech-AI
Assistente de voz em Python no Google Colab que grava áudio no navegador, transcreve com Whisper, gera respostas com modelos da Hugging Face e converte texto em fala em português. Projeto que integra reconhecimento de fala, NLP (Processamento de Linguagem Natura) e síntese de voz em um fluxo completo. Esse é o desafio final do Bootcampo Bradesco - GenAI & Dados, sendo alterado para que pudesse ser utilizado com ferramentas gratuitas, incluindo atualizações de código para que seja executável em 2026.

# 🎙️ Assistente de Voz com Whisper + Hugging Face no Google Colab

Este projeto cria um **assistente de voz em português** que:
1. Grava áudio no navegador.
2. Transcreve fala em texto usando **Whisper**.
3. Gera respostas com modelos gratuitos da **Hugging Face**.
4. Converte texto em fala (TTS) para responder em áudio.
---

📌 Observações

• 	Projeto atualizado para ser 100% gratuito, sem necessidade de chave da OpenAI.<br>
• 	Ideal para iniciantes em IA aplicada à voz com base em Python e JS.<br>
• 	Pode ser expandido com outros modelos de linguagem ou TTS.

---
## Teste no Google Colab

Você pode abrir e executar este projeto diretamente no Google Colab clicando no botão abaixo:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_j2vNKUs0WP3W0bGgs_wEddEyfrOT8oO#scrollTo=ZXpYi2m8NncD)

---

## 🚀 Configuração Inicial
Definimos o idioma global:

```python
language = "pt"

````
🎤 Etapa 1: Gravação de Áudio
Captura de áudio via navegador usando JavaScript integrado ao Colab com código atualizado.

```from IPython.display import display, Javascript
from google.colab import output
from base64 import b64decode

Record = """
async function Record(ms) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  let data = [];
  recorder.ondataavailable = event => data.push(event.data);
  recorder.start();
  await new Promise(resolve => setTimeout(resolve, ms));
  recorder.stop();
  await new Promise(resolve => recorder.onstop = resolve);
  const audioBlob = new Blob(data);
  const reader = new FileReader();
  reader.readAsDataURL(audioBlob);
  return new Promise(resolve => {
    reader.onloadend = () => resolve(reader.result);
  });
}
"""

def recorder(sec=5):
    display(Javascript(Record))
    js_result = output.eval_js(f'Record({sec * 1000})')
    audio = b64decode(js_result.split(',')[1])
    file_name = 'request_audio.wav'
    with open(file_name, "wb") as f:
        f.write(audio)
    return f'/content/{file_name}'

record_file = recorder(5)  #grava 5 segundos

```
📝 Etapa 2: Reconhecimento de fala (Transcrição do Áudio) usando o modelo Whisper.

```
!pip install -q openai-whisper
import whisper

model = whisper.load_model("small")
result = model.transcribe(record_file, fp16=False, language=language)
transcription = result["text"]

print("Transcrição:", transcription)

```
🤖 Etapa 3: Resposta com Hugging Face, substituindo o ChatGPT (gpt-3.5-turbo ou superior), sem necessidade de uma api_key e gratuito.<br>
Por o Hugging Face usar um modelo antigo (GPT2), podem ocorrer alucinações da IA.

```
!pip install -q transformers torch
from transformers import pipeline

chatbot = pipeline("text-generation", model="pierreguillou/gpt2-small-portuguese")
chat_response = chatbot(transcription, max_length=100, do_sample=True, top_p=0.95, temperature=0.7)[0]["generated_text"]

print("Resposta:", chat_response)

```
🔊 Etapa 4: Conversão em Áudio com Hugging Face TTS. A bibliotega gTTS performa melhor nessa etapa e também é gratuita.
Utilizando o hfTTS apenas para mostrar alternativas.

```
!pip install -q transformers soundfile
from transformers import pipeline
import soundfile as sf
import numpy as np
from IPython.display import Audio

tts = pipeline("text-to-speech", model="facebook/mms-tts-por")
speech = tts(chat_response)

# Ajustar velocidade (ex.: 1.25x mais rápido)
speed_factor = 1.25
new_audio = np.interp(
    np.arange(0, len(speech["audio"]), speed_factor),
    np.arange(0, len(speech["audio"])),
    speech["audio"]
)

response_audio_path = "/content/response_audio_fast.wav"
sf.write(response_audio_path, new_audio, speech["sampling_rate"])

Audio(response_audio_path, autoplay=True)

```
✅ Fluxo Completo
1. 	🎤 Você fala → áudio é gravado.
2. 	📝 Whisper transcreve para texto.
3. 	🤖 Hugging Face gera resposta.
4. 	🔊 Resposta é falada de volta para você.
