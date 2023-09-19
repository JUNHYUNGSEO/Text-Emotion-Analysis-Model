from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.models import load_model
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('tokenizer.json', 'r', encoding='utf-8') as f:
    json_data = f.read()
    loaded_tokenizer = tokenizer_from_json(json_data)

model = load_model('my_model.h5')

emotion_mapping = {'분노': 0, '기쁨': 1, '불안': 2, '당황': 3, '슬픔': 4, '상처': 5}
reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

new_text = ['진짜 악질적이다…해명글도 진짜..현장에서 숨어있는 가해자 찾아내시고.. 경찰분들 고생 많으셨어요ㅠㅠㅜ']

new_sequences = loaded_tokenizer.texts_to_sequences(new_text)

if not new_sequences or any(None in sublist for sublist in new_sequences):
    print("토크나이저가 새로운 텍스트에 대한 단어를 알지 못합니다.")
else:
    new_padded = pad_sequences(new_sequences, maxlen=10, padding='post', truncating='post')
   

    
    predictions = model.predict(new_padded)

    predicted_label = np.argmax(predictions[0])
    emotion_str = reverse_emotion_mapping[predicted_label]
    print(f"'{new_text[0]}'의 예측 감정: {emotion_str}")
