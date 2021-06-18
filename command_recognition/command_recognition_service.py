import librosa
import tensorflow as tf
import numpy as np
from command_recognition.build_features import COMMAND_LENGTH, THIS_DIR, DATASET_SAMPLE_RATE
from command_recognition.train_model import MODEL_PATH
import os


class _CommandRecognitionService:
    """Singleton class for command recognition."""

    model = None
    _mapping = [
        "down",
        "off",
        "on",
        "no",
        "yes",
        "stop",
        "up",
        "right",
        "left",
        "go"
    ]
    _instance = None

    def predict(self, file):
        """
        :param file: audio file or path to audio file to predict
        :return predicted_command(str): Command predicted by the model
        """

        # extract MFCC
        MFCCs = self.preprocess(file)

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_command = self._mapping[predicted_index]
        return predicted_command

    def preprocess(self, file, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.
        :param file: audio file or path to audio file to predict
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Length of window for STFT computation (in samples)
        :param hop_length (int): Window step for STFT (in samples)
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (time, coefficients)
        """

        # load and resample audio file
        signal, sample_rate = librosa.load(file, sr=DATASET_SAMPLE_RATE)

        # check length of signal
        if len(signal) >= COMMAND_LENGTH:
            # truncate signal to COMMAND_LENGTH
            signal = signal[:COMMAND_LENGTH]

        elif len(signal) < COMMAND_LENGTH:
            # zero pad signal to COMMAND_LENGTH
            padding = np.zeros(COMMAND_LENGTH - len(signal))
            signal = np.append(signal, padding)

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

        # input data for the model should be 4 dimensional array: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        return MFCCs


def CommandRecognitionService():
    """Factory function for _CommandRecognitionService class.
    :return _CommandRecognitionService._instance
    """

    # ensure an instance is created only the first time the factory function is called
    if _CommandRecognitionService._instance is None:
        _CommandRecognitionService._instance = _CommandRecognitionService()
        _CommandRecognitionService.model = tf.keras.models.load_model(MODEL_PATH)
    return _CommandRecognitionService._instance


if __name__ == "__main__":
    # Test prediction
    service = CommandRecognitionService()
    keyword = service.predict(os.path.join(THIS_DIR, "down.wav"))
    print(keyword)
