# dsc2021

atheleticsで開催されたバンダイナムコ様の  
Data Science Challenge 2021の課題の回答プログラムです

# DEMO

45フレーム補間のデモGIFです

![2022-01-10 11-46-18](https://user-images.githubusercontent.com/54616067/148713970-d5b8964f-122d-4d2c-b730-f0a2a1026857.gif)

# 用いた手法

以下の論文を参考にモデルを構築した

*Robust Motion In-betweening*  
https://static-wordpress.akamaized.net/montreal.ubisoft.com/wp-content/uploads/2020/07/09155337/RobustMotionInbetweening.pdf
(Félix G. Harvey and Mike Yurick and Derek Nowrouzezahrai and Christopher Pal, ACM Transactions on Graphics,39,4,2020)

*Recurrent Transition Networks for Character Locomotion*  
https://arxiv.org/ftp/arxiv/papers/1810/1810.02363.pdf
(Harvey et al. 2018)

## モデルの全体図　　

![models-1](https://user-images.githubusercontent.com/54616067/148716821-62630fdb-fb4e-42bf-8cef-ad6015cd3601.jpg)

基本的構造は*Robust Motion In-betweening*を参考にしている  
しかし、こちらの論文ではQuatanionデータも使用して、Forward Kinematicsを用いるなどして精度を向上させていた  
今回の課題では、データがポジションのみなので*Recurrent Transition Networks for Character Locomotion*改良前のこちらの論文を参考にしてモデル構築を行った

# Requirement

実行環境 Windows
Anaconda

pytorch
open-cv
numpy

* conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
* conda install -c anaconda numpy
* conda install -c conda-forge opencv

# Usage
実行方法
```bash
git clone https://github.com/Kerorin9899/dcs2021.git
cd dcs2021
```

train
```bash
python main.py -name "Test"
```

Train 引数の設定
* --train_path : "input_data/train/train.csv"
* -name        : 重みを保存する名前 str
* -lr          : 学習率 float
* -len         : 学習する補間の長さ int
* -batch       : バッチ数 int
* -e           : エポック数 int
* -load        : 学習した重みを転移学習する際に使用 str
* -full        : 学習するデータすべてで学習するかどうか bool

predict
```bash
python predict.py -name "Test" -weights "weights/Test.pth" -len 5 --test_path "input_data/test/test_easy.csv"
```

Predict 引数の設定
* --test_path  : どのファイルをPredictするか
* -name        : Csvファイルを保存する名前 str
* -weights     : 学習した重みのパス str
* -len         : 学習した補間の長さ int

**test_pathは必ず補完するlenの長さを一致したものを使用すること**
