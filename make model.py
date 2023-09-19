import numpy as np
import pandas as pd
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

df = pd.read_csv('Training.csv')

emotion_mapping = {'분노': 0, '기쁨': 1, '불안': 2, '당황': 3, '슬픔': 4, '상처': 5}
df['감정_대분류'] = df['감정_대분류'].map(emotion_mapping)

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['사람문장1'])

tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer.to_json())

sequences = tokenizer.texts_to_sequences(df['사람문장1'])
padded = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')

model = Sequential([
    Embedding(10000, 16, input_length=10),
    LSTM(32),
    Dense(24, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(padded, df['감정_대분류'], epochs=100)


model.save('my_model.h5')


reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}


new_texts = ['일은 왜 해도 해도 끝이 없을까?', '이번 달에 또 급여가 깎였어!', '회사에 신입이 들어왔는데 말투가 거슬려']
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=10, padding='post', truncating='post')
predictions = model.predict(new_padded)

for i, pred in enumerate(predictions):
    predicted_label = np.argmax(pred)
    emotion_str = reverse_emotion_mapping[predicted_label]
    print(f"'{new_texts[i]}'의 예측 감정: {emotion_str}")