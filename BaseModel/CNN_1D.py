#!/usr/bin/env python
#encoding=utf-8
"""
@author: TianMao
@contact: tianmao1994@yahoo.com
@file: CNN_1D.py
@time: 18-12-22 下午3:25
@desc:
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from keras.regularizers import l1_l2
import pickle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# --------------------------Embedding part-----------------------------
train_df = pd.read_csv("../../Qura_data/train_clean.csv")
test_df = pd.read_csv("../../Qura_data/test_clean.csv")

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

embeddings_index=None
with open("../../Qura_data/glove.pickle",'rb') as f:
    embeddings_index = pickle.load(f)
#-----------------------------------------------------------------------


# -------------------------定义解析方法,取前30个词-------------------------
# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    # 这里仅仅使用了每一个问题的前30个单词!
    text = text[:-1].split()[:72]

    # 没有在字典中的词语直接使用了300维的[0,0...,0]向量代替
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    # 长度不够72个单词的也使用[0,0...,0]向量代替
    embeds+= [empyt_emb] * (72 - len(embeds))
    return np.array(embeds)
#------------------------------------------------------------------------

val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])])
val_y = np.array(val_df["target"][:3000])

# Data providers
batch_size = 128
input_shape=(72,300)
nb_filters=64
kernel_size=3

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

from keras.models import Sequential
from keras.layers import Conv1D,Dense,Flatten,MaxPooling1D
model=Sequential()
model.add(Conv1D(filters=nb_filters,kernel_size=kernel_size,padding='same',strides=1,input_shape=input_shape))
model.add(MaxPooling1D())
model.add(Conv1D(filters=nb_filters,kernel_size=kernel_size,padding='same',strides=1))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(128,activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy',f1])
model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint
check_point = ModelCheckpoint('model_CNN.hdf5', monitor="val_f1", mode="max",
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
model.load_weights('model_CNN.hdf5')

batch_size = 256
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        # 将所有的样本一次性全部解析
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr



all_preds = []
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(model.predict(x).flatten())

y_te = (np.array(all_preds) > 0.5).astype(np.int)

submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission_cnn.csv", index=False)