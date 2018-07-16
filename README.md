# speech_emotion
Detect human emotion from audio.

Refer to some code in [Speech_emotion_recognition_BLSTM](https://github.com/RayanWang/Speech_emotion_recognition_BLSTM), thanks a lot.

### Get started

environment: Python 3

#### main dependencies

- tensorflow- `pip install tensorflow`
- keras- `pip install keras`: build and train the B-LSTM model
- librosa- `pip install librosa`: audio resampling

#### dataset

[Berlin Database of Emotional Speech](http://emodb.bilderbar.info/download/), you can download and unzip it in data/ folder.

### How use it

1. `python train.py`

Train the model. You can skip this because the trained model named "weights_blstm_hyperas_1.h5" has been uploaded. If you want to retrain the model, you will need to extract features from [berlin](http://emodb.bilderbar.info/download/) dataset when you first run it. For saving time, the audio feature file named "berlin_db.p" and "berlin_features.p" has uploaded.

2. `python predict.py`

Predict emotion from audio. You shoud specify the file path of audio to be predicted. For good performance, the audio should be less than 5 second. You will get the result such as 

"the top 2 emotion is: ('happiness', 0.20501734)
the top 2 emotion is: ('neutral', 0.29067296)"

### File structure

- utility folder
  - audioFeatureExtraction.py: extract feature from audio. Modify from [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)
  - functions.py: some utility about audio.
  - globalvars.py: global variable.
- berlin_db.p & berlin_features.p: [berlin](http://emodb.bilderbar.info/download/) feature  file.
- predict.py: predict emotion from audio file.
- train.py: train the model by keras.
- weights_blstm_hyperas_1.h5: trained model.

### Connect

[cnmengnan@gmail.com](mailto:cnmengnan@gmail.com)

blog: [WinterColor blog](http://www.cnblogs.com/mengnan/)

enjoy it