---
title: "감정 분류 예측 모델"
output: html_document
---

## 감정 분류 예측 모델을 위한 Python 코드

### 필요한 라이브러리 불러오기



```{python}
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json

토크나이저 설정 불러오기
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    json_data = f.read()
    loaded_tokenizer = tokenizer_from_json(json_data)

미리 학습된 모델 불러오기
model = load_model('my_model.h5')

감정 분류 매핑
emotion_mapping = {'분노': 0, '기쁨': 1, '불안': 2, '당황': 3, '슬픔': 4, '상처': 5}
reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

예측하고 싶은 문장
new_text = ['관객의 몸과 마음을 뒤흔드는 굉음, 그 뒤에 찾아오는 서늘함.']

텍스트를 시퀀스로 변환
new_sequences = loaded_tokenizer.texts_to_sequences(new_text)

if not new_sequences or any(None in sublist for sublist in new_sequences):
    print("토크나이저가 새로운 텍스트에 대한 단어를 알지 못합니다.")
else:
    new_padded = pad_sequences(new_sequences, maxlen=10, padding='post', truncating='post')
    predictions = model.predict(new_padded)

예측 결과 출력
predicted_label = np.argmax(predictions[0])
emotion_str = reverse_emotion_mapping[predicted_label]
print(f"'{new_text[0]}'의 예측 감정: {emotion_str}")
