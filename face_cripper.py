# -*- coding: utf-8 -*-
import os
import cv2
import argparse
from PIL import Image


def filter_extension(img_dir, extensions):
    # """ある拡張子のファイルだけを返す"""

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.split('.')[-1] in extensions:
                yield os.path.join(root, file), root, file


def resize_img(img_dir, extensions, size):
    # """ディレクトリの画像を、リサイズし上書きする"""

    for file_path, _, _ in filter_extension(img_dir, extensions):
        img = Image.open(file_path)
        new_img = img.resize(size)
        new_img.save(file_path)


SPINS = {
    'flipTB': Image.FLIP_TOP_BOTTOM,
    'spin90': Image.ROTATE_90,
    'spin270': Image.ROTATE_270,
    'flipLR': Image.FLIP_LEFT_RIGHT,
}

def spin_img(img_dir, extensions):
    # """ディレクトリの画像を、4方向にスピンしたものを追加する"""

    for file_path, root, file in filter_extension(img_dir, extensions):
        img = Image.open(file_path)
        for name, kind in SPINS.items():
            new_img = img.transpose(kind)
            new_file_name = '{0}_{1}'.format(name, file)
            new_file_path = os.path.join(root, new_file_name)
            new_img.save(new_file_path)


def detect_face(image, cascade_path):
    # """顔の検出を行う"""

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)

    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale( image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(64, 64))

    return facerect


def collect_img(cap, fps, extension, save_dir, cascade_path):
    # """動画から、顔を検出し保存する"""

    frame_number = 0
    img_number = 0

    while(cap.isOpened()):
        frame_number += 1
        ret, image = cap.read()
        if not ret:
            break

        # ループfps回毎に処理に入る。少なくすると画像が増える（細かくキャプチャされる）
        if (frame_number % fps) == 0:
            facerect = detect_face(image, cascade_path)

            # 認識結果がnullだったら次のframeへ
            if len(facerect) == 0:
                continue

            for rect in facerect:
                h,w,c = image.shape
                croped = image[ max([rect[1]-int(0.5*rect[3]),0]):min([rect[1]+int(1.1*rect[3]),h]), max([rect[0]-int(0.2*rect[3]),0]):min([rect[0]+int(1.2*rect[2]),w])]
                file_name = '{0}.{1}'.format(img_number, extension)
                path = os.path.join(save_dir, file_name)
                cv2.imwrite(path, croped)
                img_number += 1
    return img_number


def start(video_path, cascade_path, save_dir, extension,
          resize_size, fps,  resize, pad):
    # """動画から、画像の取得を開始する"""

    # 保存先のディレクトリが存在しなければ作る
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 動画から、画像を抜き出す
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('not find video file')
        exit

    img_numbers = collect_img(cap, fps, extension, save_dir, cascade_path)
    cap.release()
    print('画像のキャプチャを終了', img_numbers)

    # 画像を、リサイズし統一する
    if resize:
        resize_img(save_dir, (extension,), resize_size)
        print('リサイズの終了')
 
    # スピン画像を追加し、データを水増しする
    if pad:
        spin_img(save_dir, (extension,))
        print('スピン画像の追加を終了')
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='anime face cripper')
    parser.add_argument('video_path',type=str,help='PATH')
    parser.add_argument('-s','--save_directory',type=str,default='imgs')
    parser.add_argument('-c','--classifier',type=str,default='lbpcascade_animeface.xml')
    parser.add_argument('--extension',type=str,default='jpg')
    parser.add_argument('-f','--framerate', type=int,default='100')
    parser.add_argument('-r','--resize_img_size', type=int,default='256')
    parser.add_argument('--resize',type=bool,default=False)
    parser.add_argument('--pad',type=bool,default=False)
    args = parser.parse_args()
    start(  args.video_path, 
            args.classifier ,
            args.save_directory , 
            args.extension ,
            (args.resize_img_size ,args.resize_img_size) ,
            args.framerate ,
            args.resize ,
            args.pad)