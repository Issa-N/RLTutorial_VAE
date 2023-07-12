# torch_tutorial_y_i-3
The repository for M1 tutorial. 

The administers are YANO and NAKAMURA. 

このリポジトリは「課題3 オートエンコーダVAE」に関するリポジトリです.

# 課題
以下の1.-7.を実施する．

1. MNISTをダウンロードする．
2. オートエンコーダを実装する．(ただし，各種パラメータhあ変更できるようにすること)
3. 潜在空間の次元数を2としてオートエンコーダを訓練する．
4. VAEを実装する．
5. 潜在空間の次元数を2としてVAEを訓練する．
6. 潜在空間を可視化する．
7. AEとVAEの潜在空間を比較し，考察する．


# Quick Start
## SSH接続(SSH接続でリモートPCへ接続する場合のみlocal terminalにて実行が必要)
```
 ssh -L 63322:localhost:63322 -L 6006:localhost:6006 -L 6007:localhost:6007  <username>@<remotePC IP>
```
接続完了後， 環境構築へ
## nativeの場合
環境構築へ
## 環境設定
### 初回
任意のディレクトリにて， githubからリポジトリのクローンを作成する．
```
git clone https://github.com/Issa-N/torch_tutorial_y-i-3.git
```
次に，Dockerで仮想環境を作る．
```
cd torch_tutorial_y-i-3/docker

bash build.sh
bash run.sh
cd
```
### 2回目以降
以下のコマンドで，dockerのコンテナへの接続を行う．
```
docker exec -it torch_tutorial_y-i　bash
```
一時的に抜ける時は 「controll + P + Q」

## Jupiter notebookを開く時
```
cd
jupyter-notebook --ip 0.0.0.0 --port 63322 --allow-root
```


# notebookデータについて
本課題において一連のVAE, AEの構築及び評価に関しては ~/src/run_AE_VAE.ipynb にて作業を行なった. 

まず[Jupiter notebookを開く時]に記載済みのコマンドを実行し, jupyter notebookが起動することを確認する. 

~/src に移動し 表示されたファイル群の中のrun_AE_VAE.ipynbを選択することで今回の課題成果物を確認できる. 

詳細については当notebookを参照されたい. 

# ハイパラメータ指定

NNを組んだ後にネットワーク構造を変更し, その挙動を確認したいという場面は往々にして起こりうる事象であるため, プログラム自体も

それに適応した形にするべきである. そこで今回は~/src/main.pyの実行時にAE ,VAE共にハイパラメータやデータセットの分割割合を指定可能なプログラムを作成した. 

下記コマンドを実行することで任意パラメータでの実行が可能となっている. 
```
python3 main.py --patch_size 100 --train_rate 0.8 --EPOCH 20 --z_dim 2 --input_size 28 --array_number 300 100
```

そして下記コマンドで"create_fig.py"を実行すると，"history_AE.pkl"と"history_VAE.pkl"を読み込み、学習損失と潜在空間のグラフ描画を行い、グラフをpngファイル形式で保存する．
```
python3 create_fig.py
```
最後に，jupyter notebookに戻り"Figure.ipynb"に記載されているセルを実行することで，"create_fig.py"で作図したグラフを確認することができる．
