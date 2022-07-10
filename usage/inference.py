import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import soundfile as sf


SAMPLE_RATE = 16000
STEP = 1600 # 100ms
COMMANDS = ['ass', 'backward', 'bed', 'bird', 'bitch', 'booty', 'cat', 'cocaine', 'cock',
 'cum', 'damn', 'dick', 'dog', 'down', 'eight', 'faggot', 'fellation', 'five',
 'follow', 'forward', 'four', 'fuck', 'go', 'goddamn', 'happy', 'hoe', 'house',
 'learn', 'left', 'marvin', 'nigga', 'nine', 'no', 'off', 'on', 'one', 'pussy',
 'right', 'seven', 'sex', 'sheila', 'shit', 'six', 'stop', 'suck', 'three',
 'titties', 'tree', 'two', 'up', 'visual', 'weed', 'wow', 'yes', 'zero',
 '_background_noise_']


def get_spectrogram(waveform):
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  waveform = tf.cast(waveform, dtype=tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


if __name__ == '__main__':
    vocals_filename = "vocals.wav"
    no_vocals_filename = "no_vocals.wav"
    model = tf.keras.models.load_model('cnn/')
    x, sr = sf.read(vocals_filename)
    print("Search...")
    for i in range(0, len(x), STEP):
        chunk = x[i:i+SAMPLE_RATE]
        chunk = tf.convert_to_tensor(chunk, dtype=tf.float32)
        spec = get_spectrogram(chunk)
        spec = tf.reshape(spec, [1, *spec.shape])
        prediction = model.predict(spec, verbose=0)
        if np.argmax(tf.nn.softmax(prediction[0])) == 21 and np.max(tf.nn.softmax(prediction[0])) > 0.8:
            print(i / SAMPLE_RATE, COMMANDS[np.argmax(tf.nn.softmax(prediction[0]))], np.max(tf.nn.softmax(prediction[0])))
            x[i:i+SAMPLE_RATE] = 0
    print("done.")
    sf.write(vocals_filename.split(".")[0] + "_vocals_edit.wav", x, sr)
    print("Vocals result saved to", vocals_filename.split(".")[0] + "_vocals_edit.wav")
    y, sr = sf.read(no_vocals_filename)
    result = x + y
    sf.write(vocals_filename.split(".")[0] + "_mixed_edit.wav", result, sr)
    print("Mixed result saved to", vocals_filename.split(".")[0] + "_mixed_edit.wav")