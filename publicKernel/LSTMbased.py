#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: LSTMbased.py
@time: 18-12-22 上午10:24
@desc:
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split


# --------------------------Embedding part-----------------------------
train_df = pd.read_csv("../../Qura_data/train.csv")
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=28)

embeddings_index = {}
f = open('../../Qura_data/embeddings/glove.840B.300d/glove.840B.300d.txt')

# 粗暴将embedding结果解析到字典中,解析时间大约两分钟
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
#-----------------------------------------------------------------------


# -------------------------定义解析方法,取前30个词-------------------------
# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    # 这里仅仅使用了每一个问题的前30个单词!
    text = text[:-1].split()[:30]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (30 - len(embeds))
    return np.array(embeds)
#------------------------------------------------------------------------

val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])])
val_y = np.array(val_df["target"][:3000])

# Data providers
batch_size = 128

# -------------------------定义批次函数--------------------------------------
def batch_gen(train_df):
    n_batches = math.ceil(len(train_df) / batch_size)
    while True:
        train_df = train_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            # 每一个批次解析一次数据
            text_arr = np.array([text_to_array(text) for text in texts])
            yield text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])
#---------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout

# --------------------------metrics part------------------------------------
from keras import backend as K
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# -------------------------------------------------------------------------

# ---------------------------------模型部分---------------------------------
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True),
                        input_shape=(30, 300)))
model.add(Bidirectional(LSTM(64)))

# model.add(Dropout(0.1))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f1])
# --------------------------------------------------------------------------

# ----------------------------training part---------------------------------
from keras.callbacks import EarlyStopping, ModelCheckpoint
check_point = ModelCheckpoint('model.hdf5', monitor="val_f1", mode="max",
                              verbose=True, save_best_only=True)
early_stop = EarlyStopping(monitor="val_f1", mode="max", patience=8,verbose=True)
mg = batch_gen(train_df)
model.fit_generator(mg, epochs=30,
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y),
                    verbose=True,
                    callbacks=[early_stop,check_point])
# --------------------------------------------------------------------------

# ----------------------------prediction part-------------------------------
model.load_weights('model.hdf5')

batch_size = 256
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        # 将所有的样本一次性全部解析
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr

test_df = pd.read_csv("../../Qura_data/test.csv")

all_preds = []
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(model.predict(x).flatten())

y_te = (np.array(all_preds) > 0.5).astype(np.int)

submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)
