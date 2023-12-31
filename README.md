# Physiris

深層学習・DL基礎2023 最終課題

## 概要

![機器の設置](20230728_2215.png) 

Physirisは身体運動で操作するテトリス風ゲームです。

## 必要システム環境

動画撮影機器（例　Webカメラ）

Python 3（バージョン3.10.4 Win32／amd64版にて動作確認を行いました）

Python用パッケージopencv-python（バージョン4.8.0.74にて動作確認を行いました）

Python用パッケージpygame（バージョン2.1.2にて動作確認を行いました）

YOLOv8（Python用パッケージultralytics）

## 導入（インストール）

動画撮影機器（例　Webカメラ）をコンピューターに接続し、ビデオ入力用として使用できるようにしておきます。

Python 3を導入します。

Python用パッケージopencv-python、pygame、ultralyticsを導入します（例　コマンドラインシェルにて命令「pip install (Python用パッケージの名前)」を発行します）。但し、パッケージultralyticsに
はYOLOv8以外にも様々なライブラリーが含まれますが、今回必要なものはYOLOv8のみですので、パッケージultralyticsを導入する代わりに、YOLOv8に関するGitHubのページ（ https://github.com/ultralytics/ultralytics ）に掲載されているファイルを、Physirisの本体であるファイル「physiris.py」と同じディレクトリーに配置しても構いません。その場合には、YOLOv8に関するGitHubのページから入手したファイル「requirements.txt」に記載されているPython用パッケージを導入する必要があります（例　コマンドラインシェルにて命令「pip install -r requirements.txt」を発行します）。

YOLOv8用のモデルyolov8n（ファイル「yolov8n.pt」）が別途必要になります。通常は最初にPhysiris（あるいはYOLOv8）を実行するときに自動的にインターネット経由にてダウンロードされますが、代わりに事前にインターネット経由にて https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt からダウンロードし、それによって得られたファイル「yolov8n.pt」をPhysirisの本体であるファイル「physiris.py」と同じディレクトリーに配置しても構いません。

## 使用方法

「現在のディレクトリー」（the current directory / the working directory）をPhysirisの本体であるファイル「physiris.py」が存在するディレクトリーに設定し、コマンドラインシェルにて命令「python physiris.py」を発行します。

ゲーム画面を映しているウィンドウ「Physiris」が表示されたら操作が可能です。Physirisの特徴は身体運動によって操作できるところにありますが、Physirisの起動時には身体運動による操作は無効になっています。身体運動による操作（YOLOv8）を有効にするには、キーボードのキーYを押します。キーUを押すと、身体運動による操作（YOLOv8）は無効になります。

操作方法としての身体運動としては左右への移動、跳躍やしゃがみが含まれますが、これらが正しく認識されるために、Physirisの起動時に位置合わせが必要です（現時点では設定をファイルなどに保存しませんので、Physirisを起動するたびに位置合わせを行う必要があります）。そこで、まず、操作者（ゲームプレイヤー）の身長が動画撮影機器によって撮影された結果得られる画像の高さ方向の50～75％程度を占めるように、動画撮影機器から離れます。また、操作者の全身がその画像に収まるように、動画撮影機器の（上下方向の）向きを調節します。そして、操作者が「左端」（好きに決めていただいて構いません）へ移動したところでキーZを押し、操作者が「右端」（好きに決めていただいて構いません）へ移動したところでキーXを押します。そして、操作者が「（水平方向の）中央」にて直立している状態でキーCを押し、操作者が「（水平方向の）中央」にて跳躍して頂点に到達する瞬間を見計らってキーVを押し、操作者が「（水平方向の）中央」にてしゃがんでいる状態でキーBを押します。このような位置合わせは何回でも行えますが、操作者と動画撮影機器の距離が明らかに変化したり、動画撮影機器の向きを変更したりする場合には、再度位置合わせを行うことをお勧めいたします。

キーEnterを押すとゲームが開始します。操作者が左右に移動すると、キャラクターも左右に移動します。操作者が跳躍すると、キャラクターは反時計回りに回転します（但し、そのキャラクターが回転の結果既に配置されているキャラクターや壁に衝突する場合には回転しません）。キャラクターは時間経過とともに落下しますが、操作者がしゃがむと、キャラクターの落下が早まります。キャラクターが落下して下にあるものにぶつかると、そのキャラクターは固定され、新しいキャラクターが出現します。

キャラクターを構成するブロックが行の全てを埋めると、その行のブロックは消去され、それより上の行が下へ詰めてきます。また、このときに消去された行の数に応じて得点が加算されます。

20行を消去すると、ゲームクリアとなり、そのままゲーム終了となります。行の消去に掛かる時間も表示されますので、ゲームクリアの所要時間で競いましょう。

キャラクターが上まで積みあがって新しいキャラクターが出現する位置に重なると、やはりゲーム終了となります。

キーEscを押すとPhysirisは終了します。

Physirisではあまり意味がありませんが、身体運動の代わりにキーボードのキーを使用してキャラクターを操作することも可能です。

## キー一覧

操作者が左へ移動する （YOLOv8有効時のみ）キャラクターを左へ移動させる

操作者が右へ移動する （YOLOv8有効時のみ）キャラクターを右へ移動させる

操作者が跳躍する （YOLOv8有効時のみ）キャラクターを反時計回りに回転させる

操作者がしゃがむ （YOLOv8有効時のみ）キャラクターの落下を早める

キーAを押す YOLOv8での「操作者が左へ移動する」の端（水平方向）を初期設定に戻す

キーBを押す YOLOv8での「操作者がしゃがむ」の基準（鉛直方向）を設定する

キーCを押す YOLOv8での鉛直方向中立状態の基準を設定する

キーDを押す YOLOv8での鉛直方向中立状態の基準を初期設定に戻す

キーFを押す YOLOv8での「操作者が跳躍する」の基準（鉛直方向）を初期設定に戻す

キーGを押す YOLOv8での「操作者がしゃがむ」の基準（鉛直方向）を初期設定に戻す

キーEnterを押す ゲームを開始する

キーEscを押す Physirisを終了する

キーSを押す YOLOv8での「操作者が右へ移動する」の端（水平方向）を初期設定に戻す

キーUを押す YOLOv8を無効にする

キーVを押す YOLOv8での「操作者が跳躍する」の基準（鉛直方向）を設定する

キーXを押す YOLOv8での「操作者が右へ移動する」の端（水平方向）を設定する

キーYを押す YOLOv8を有効にする

キーZを押す YOLOv8での「操作者が左へ移動する」の端（水平方向）を設定する
