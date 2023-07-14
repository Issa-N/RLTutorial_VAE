# Source Code Description
author: Issa Nakamura

# 1.Basic Usage
 初めに，"main.py"を実行する．
 (実行時、オプションとしてデータセットのミニバッチ数やエポック数を指定する必要がある．オプションの詳細については表1を参照)
```
python3 main.py --patch_size 100 --train_rate 0.8 --EPOCH 20 --z_dim 2 --input_size 28 --array_number 300 100
```
表1　main.pyのパラメータ詳細
|  パラメータ名  |  意味  |  備考  |
| ---- | ---- | ---- |
|  patch_size  |  パッチ数  |  　正の整数を入力　  |
|  train_rate  |  訓練データの比率  |  0以上1以下の少数で入力  |
|  EPOCH  |  エポック数  |  正の整数を入力  |
|  z_dim  |  潜在空間の次元数  |  今回は課題の設定上、2を指定  |
|  input_size  |  入力画像の縦幅  |  今回はMNISTを使うので28を指定  |
|  array_number  |  畳み込み層による削減次元数  |  第1要素>第2要素となるように3以上の正の整数を入力  |

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
### 関数)
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
### 関数)
 
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

 ## 2.6 learning_AE.py
### 概要)
  AEの学習を行う．
### 関数)
 
 　**Learning_VAE(z_dim, num_epochs, train_loader,val_loader)**
  
  　引数:　z_dim　-> 潜在空間の次元数(整数)
   
   　　　　num_epochs　->　エポック数(整数)

   　　　　train_loader　-> 訓練データセット

   　　　　val_loader　-> テストデータセット

  
  　返り値:　history_AE -> 学習結果(辞書型)

 ## 2.7 learning_VAE.py
### 概要)
  VAEの学習を行う．
### 関数)
 
 　**Learning_VAE(z_dim, num_epochs, train_loader,val_loader)**
  
  　引数:　z_dim　-> 潜在空間の次元数(整数)
   
   　　　　num_epochs　->　エポック数(整数)

   　　　　train_loader　-> 訓練データセット

   　　　　val_loader　-> テストデータセット

  
  　返り値:　history_VAE -> 学習結果(辞書型)
