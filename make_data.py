from sklearn.model_selection import train_test_split
from PIL import Image
import os,glob
import numpy as np
import shutil
import math
import random


#分類のカテゴリーを選ぶ
root_dir = './data/'
train_dir = './data/train/'
test_dir = './data/test/'
charactors = ['tokinosora','fubuki','roboco','matsuri','aki','choco','meru','aqua','subaru','ayame','shion','ha-to','mio','miko']
nb_classes = len(charactors)
image_size = 64


if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)
#顔をtrain,testに分けるコード
for idx,charactor in enumerate(charactors):
    print('----{}を処理中----'.format(charactor))
    image_dir = root_dir + charactor
    move_train_dir = train_dir + charactor
    if not os.path.isdir(move_train_dir):
        os.mkdir(move_train_dir)
    move_test_dir = test_dir + charactor
    if not os.path.isdir(move_test_dir):
        os.mkdir(move_test_dir)
    files = glob.glob(image_dir+'/*.jpg')
    print(len(files))
    th = math.floor(len(files)*0.2)
    random.shuffle(files)
    #データの20%をtestディレクトリに移動させる
    for i in range(th):
        shutil.copy(files[i],move_test_dir)
    #残りすべてをtrainディレクトリに移動させる
    files = glob.glob(image_dir+'/*.jpg')
    for file in files:
        shutil.copy(file,move_train_dir)


"""画像データをNumpy形式に変換"""
root_dir = './data/train/'
#訓練データ
#フォルダごとの画像データを読み込む
X = []
Y = []
for idx,charactor in enumerate(charactors):
    image_dir = root_dir + charactor
    files = glob.glob(image_dir+'/*.jpg')
    print('----{}を処理中----'.format(charactor))
    for i,f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB') #カラーモードの変更
        img = img.resize((image_size,image_size))#画像サイズの変更
        data = np.asarray(img)
        X.append(data)#画像をベクトルにしたもの
        Y.append(idx)#二値化問題
X = np.array(X)
Y = np.array(Y)

#学習用データと検証用データに分ける
X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=0)
xy = (X_train,X_test,y_train,y_test)
np.save('./data/train.npy',xy)
print(X_train.shape[1:])
print('ok',len(Y))

#テストデータ
#フォルダごとの画像データを読み込む
X = []
Y = []
for idx,charactor in enumerate(charactors):
    image_dir = root_dir + charactor
    files = glob.glob(image_dir+'/*.jpg')
    print('----{}を処理中----'.format(charactor))
    for i,f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB') #カラーモードの変更
        img = img.resize((image_size,image_size))#画像サイズの変更
        data = np.asarray(img)
        X.append(data)#画像をベクトルにしたもの
        Y.append(idx)#二値化問題
X = np.array(X)
Y = np.array(Y)

#テストデータ保存
np.save('./data/X_test.npy',X)
np.save('./data/y_test.npy',Y)
print(X_train.shape[1:])
print('ok',len(Y))
