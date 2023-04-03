import whisper
import openai
from decouple import config

model = whisper.load_model("medium")

audio = whisper.load_audio("doctorpolochocolate.mp4")

result = model.transcribe(audio, fp16=False)

print('real transcription -> ', result['text'], '\n')

openai.api_key = config("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[{
    "role": "user",
    "content": 'Me podrías decir porque el chocolate es bueno según este texto: ' + result['text']
  }],
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response['choices'][0]['message']['content'])
