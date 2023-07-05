# torch_tutorial_y_-3
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
## 環境設定
### 初回
初めに，githubからリポジトリのクローンを作成する．
```
git clone https://github.com/Issa-N/torch_tutorial_y-i-3.git
```
次に，Dockerで仮想環境を作る．
```
cd torch_tutorial_y-i-3
cd docker

bash build.sh
bash run.sh
```
### 2回目以降
以下のコマンドで，dockerのコンテナへの接続を行う．
```
docker exec -it torch_tutorial_y-i　bash
```
抜ける時は 「controll + P + Q」


