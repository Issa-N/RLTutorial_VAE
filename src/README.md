# Source Code Description
author: Issa Nakamura

# 1.Basic Usage
 初めに，"main.py"を実行する．
```
python3 main.py
```
　"main.py"を実行すると，AEとVAEを用いてMNISTのクラスタリング学習を行い，srcフォルダ内に学習モデルと学習結果をファイル形式で保存する．

  保存されるファイル形式の詳細は以下に記載する，

  【学習モデル】
  - model_AE.pth
  - model_VAE.pth
    
  【学習結果】
  - history_AE.pkl
  - history_VAE.pkl


 次に，"create_fig.py"を実行する．
```
python3 create_fig.py
```
　"create_fig.py"を実行すると，"history_AE.pkl"と"history_VAE.pkl"を読み込み、学習損失と潜在空間のグラフ描画を行い、グラフをpngファイル形式で保存する．

  最後に，jupyter notebookで"Figure.ipynb"を実行することで，"create_fig.py"で作図したグラフを確認することができる．

# 2. About Source Code
　本章では、srcファイル内にある各ソースコードの概要を説明する．

## 2.1 main.py
### 概要)
  すべての基本的なソースコードを動かす中数プログラム．AEおよびVAEを用いてMNISTのクラスタリングを行う．
### クラス&関数)
　なし
### 特記事項)
　AEとVAEの学習結果を"history_AE.pkl"および"history_VAE.pkl"、学習モデルを"model_AE.pth"および"model_VAE.pth"としてファイル形式で保存する．

## 2.2 create_fig.py
### 概要)
  "history_AE.pkl"および"history_VAE.pkl"を読み込み、学習損失と潜在空間のグラフ描画を行う．
### クラス&関数)
**def draw_fig()**

  　概要:　 "history_AE.pkl"および"history_VAE.pkl"から辞書型変数を読み込む
  
  　引数:　なし
  
  　返り値:　history_AE(辞書型),history_VAE(辞書型) AEとVAEの学習結果
   
**plot_lossFig(history_AE,history_VAE)**

  　概要:　 AEとVAEの学習損失をグラフ描画する
  
  　引数:　history_AE(辞書型),history_VAE(辞書型) ->AEとVAEの学習結果
  
  　返り値:　なし

  
**Create_latentFig(history_AE,history_VAE)**

  　概要:　 AEとVAEの潜在空間をグラフ描画する
  
  　引数:　history_AE(辞書型),history_VAE(辞書型) ->AEとVAEの学習結果
  
  　返り値:　なし
 
### 特記事項)
　学習ロスのグラフは、".png", 潜在空間のブラフは".png"と".png"で保存される．

## 2.3 dataset.py
### 概要)
  MNISTのデータセット生成を行う．
### クラス&関数)
 【関数】
 
 　**Create_dataset(BATCH_SIZE, train_rate)**
 
  　概要:　MNISTのデータセットを生成する
  
  　引数:　BATCH_SIZE　-> バッチサイズ(整数)
   
   　　　　train_rate　-> 訓練データの比率(浮動小数)
  
  　返り値:　なし
　

## 2.4 AE.py
### 概要)
  AEのネットワークと損失関数を定義するプログラム
### クラス&関数)
【クラス】

 　**Encoder_AE**
 
 　　⇒AE用のエンコーダネットワーク
 
 　**Decoder_AE**
 
 　　⇒AE用のデコーダネットワーク
 
 　**AE**
 
 　　⇒AEのネットワーク
 
 【関数】
 
 　**criterion_AE**
 
　　 ⇒AE用の損失関数

 ## 2.5 VAE.py
### 概要)
  VAEのネットワークと損失関数を定義するプログラム
### クラス&関数)
【クラス】

 　**Encoder_VAE**
 
 　　⇒VAE用のエンコーダネットワーク
 
 　**Decoder_VAE**
 
 　　⇒AE用のデコーダネットワーク
 
 　**VAE**
 
 　　⇒VAEのネットワーク
 
 【関数】
 
 　**criterion_VAE**
 
　　 ⇒VAE用の損失関数
