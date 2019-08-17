import numpy as np
from functools import partial
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import optimizers
from keras.utils import np_utils
import matplotlib.pyplot as plt

#カテゴリ
charactors = ['tokinosora','fubuki','roboco','matsuri','aki','choco','meru','aqua','subaru','ayame','shion','ha-to','mio','miko']
nb_classes = len(charactors)
image_size = 64

#モデル構築
def build_model(in_shape):
    model = Sequential()
    model.add(Conv2D(32,(3,3),padding='same',input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),data_format='channels_first'))

    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation='softmax'))

    #モデル構成の確認
    adam = optimizers.Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy',optimizer=adam ,metrics=['accuracy'])
    return model

#モデルを訓練する
def model_train(X,y,X_t,y_t):
    model = build_model(X.shape[1:])
    history = model.fit(X,y,batch_size=32,epochs=10,validation_data=(X_t,y_t))
    #モデルを保存する
    hdf5_file = './model/hololive.hdf5'
    model.save(hdf5_file)
    return model

#モデルの評価
def model_eval(model):
    X = np.load('./data/X_test.npy')
    y_test = np.load('./data/y_test.npy')
    #データの正規化
    X_test = X.astype('float') /256
    y_test = np_utils.to_categorical(y_test,nb_classes)
    score = model.evaluate(x=X_test,y=y_test)
    print('loss : ',score[0])
    print('accuracy : ',score[1])

    #学習開始
def main():
    np.load = partial(np.load, allow_pickle=True)
    X_train,X_test,y_train,y_test = np.load('./data/train.npy')
    np.load = partial(np.load, allow_pickle=False)
    print('X_train shape : ',X_train.shape)
    print('X_test shape : ',X_test.shape)
    print('y_train shape : ',y_train.shape)
    print('y_test shape : ',y_test.shape)
    #データの正規化
    X_train = X_train.astype('float') /255
    X_test  = X_test.astype('float') /255
    y_train = np_utils.to_categorical(y_train,nb_classes)
    y_test = np_utils.to_categorical(y_test,nb_classes)
    #モデルを訓練し評価する
    model = model_train(X_train,y_train,X_test,y_test)
    model_eval(model)

if __name__ == '__main__':
    main()