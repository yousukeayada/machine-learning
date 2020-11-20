---
marp: true
theme: gaia
paginate: true
style: |
  section {
    padding-top: 20px;
  }
---

## 深層学習
- 人間の脳のニューロンの構造と機能を模倣した「ニューラルネットワーク（NN）」を多層にした機械学習手法の一つ。
- 2012年に画像認識コンペで大きな成果を出し、それが人工知能ブームにつながる。
- https://tmytokai.github.io/open-ed/activity/dlearning/text02/page02.html

---
## 深層学習
### パーセプトロン
- 線形分離可能な問題を正しく表現できる。
$$y = step(x^Tw)=step(\sum_{i=1}^{N}x_iw_i) \\
step(v) = \left\{
\begin{array}{ll}
1 & (c \lt v) \\
0 & (otherwise)
\end{array}
\right.$$
- 多層にすることで非線形な分離もできる。

---
## 深層学習
### 順伝播型ニューラルネットワーク（多層パーセプトロン、MLP：Multi Layer Perceptron）
- 勾配法により学習する。
  - 目的関数：$E(w) = \frac{1}{2}\sum_{n=1}^N||f(x,w)-y||^2$
  - 更新式：$w = w_{old}-lr\nabla E(w)$
  - $x$：入力、$w$：重みパラメータ、$f(x,w)$：出力
  - $y$：正解
  - $lr$：学習率

---
## 深層学習
### 活性化関数
- step
  - 0 or 1
- sigmoid
  - 0 から 1 の範囲を出力する。
- tanh
  - 原点を通り -1 から 1 を出力する。
- ReLU
  - $ReLU(x)=\max(0,x)$ により勾配焼失問題を回避する。

---
## 畳み込みニューラルネットワーク（CNN）
- 例えば画像認識の場合、MLP では入力画像の全ての点を同等に扱っているが、CNN はデータの形状を保持しながら処理する。
- 入力サイズが大きくなっても重みパラメータの数は増えない。
- (入力)→(畳み込み層)→(畳み込み層)→(プーリング層)→(全結合層)→(出力)といった使われ方をする。
- https://www.imagazine.co.jp/%E7%95%B3%E3%81%BF%E8%BE%BC%E3%81%BF%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%E3%81%AE%E3%80%8C%E5%9F%BA%E7%A4%8E%E3%81%AE%E5%9F%BA%E7%A4%8E%E3%80%8D%E3%82%92%E7%90%86%E8%A7%A3%E3%81%99/

---
## 畳み込みニューラルネットワーク（CNN）
### 畳み込み層
- 画像にカーネル（フィルタ）を適用して特徴量を抽出する。
- パラメータ数はフィルタサイズに依存する。

### プーリング層
- 画像の小さな位置変化に対して頑健なモデルを構築するため、前の層から代表値を抽出する。

---
## 畳み込みニューラルネットワーク（CNN）
### CNN の応用
- 特徴抽出器としての役割（層を重ねるごとに、より高次元の特徴を抽出している）。
- 物体検出：「何がどこにあるか」を検出する。
- セグメンテーション：物体検出のように矩形領域ではなく、ピクセル単位で推定する。
- 強化学習：ゲーム画面から特徴量を抽出する（実際に人間がゲームをするときも、画面を見てキャラクターの位置などの隠れた状態変数を得て、行動を選択している）。

---
## 再帰型ニューラルネットワーク（RNN）
### SimpleRNN
- 時系列データに対して有用。長い時系列データに対してはうまくいかない。
- 中間状態：$h_t = tanh(Wx_t+Rh_{t-1})$（※バイアス項は省略）
  - $x$：入力
  - $W,R$：重みパラメータ。どのタイムステップでも同じものを使う。

---
## 再帰型ニューラルネットワーク（RNN）
### LSTM（Long-Short Term Memory）
- 長期に記憶を保存する $c_{t-1}$ を保持し、次のセルへの入力とする。

---
## 再帰型ニューラルネットワーク（RNN）
### RNN の応用
- 対話文生成：発話・応答のペアを学習させることで、発話から応答を生成する。
- 機械翻訳
- 文章生成：様々な応用が考えられる（イメージキャプショニング、ポエム生成、Actor-Critic を用いた巡回セールスマン問題の探索など）

---
### Colab
- MLP で手書き文字分類
https://colab.research.google.com/drive/1DBxK2k4PgsJr_x-6B7D-j7O6lZZDBWMu?usp=sharing
- CNN で手書き文字分類
https://colab.research.google.com/drive/1jGZIGyZxUdkBj2K68kRnpDes29OxgD91?usp=sharing
- SimpleRNN（と LSTM）で sin　波の予測
https://colab.research.google.com/drive/1iloLOgeL0ejsU88J5sekObbaPAuG8A5R?usp=sharing
