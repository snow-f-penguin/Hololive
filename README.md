Hololiver Face Recognition
====

Overview

ホロライブ所属Vtuverの3Dモデルの顔を識別するモデルの構築

## Description

顔画像は各キャラの3Dモデル初公開動画からface_cripper.pyを用いて切り抜きました．
そのため3Dモデルが公開されていないキャラクターは今回対象としていません．

3Dモデルの顔画像のみを学習データとして用いているたファンアートのキャラの識別はほぼうまくいきません．
学習に全身画像を使うことでファンアートの判別の精度が向上するのではないかと予想しています．

## Reference
構築に当たり次に挙げるサイトを参考にさせていただきました．

[画像認識で坂道グループの判別AIの作成](https://qiita.com/tigerz17/items/e4d1d5b8e00f7a771177)

[KerasでCNNを簡単に構築](https://qiita.com/sasayabaku/items/9e376ba8e38efe3bcf79)

[美女を見分けられない機械はただの機械だ：Pythonによる機械学習用データセット生成](https://qiita.com/Tatsuro64/items/821e9c0b3539baf0fd45)

## Requirement

- OS : Ubuntu 18.04
- numpy : 1.16.4
- tensorflow : 1.12.0
- keras : 2.2.4
- cudatoolkit : 9.0
- CUDNN : 7.6.0
- matplotlib : 2.2.2

## Licence

[MIT](https://github.com/snow-f-penguin/Hololive/blob/master/LICENCE)

## Author

[snow-f-penguin](https://github.com/snow-f-penguin)