import librosa
import numpy as np
import os
import json

# Project Paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(THIS_DIR, os.pardir, "data")
DATASET_PATH = os.path.join(THIS_DIR, "dataset.json")  # MFCCs output file


DATASET_SAMPLE_RATE = 16000
# set max audio length to 1.2 seconds and convert to samples (input data for CNN must have equal shape)
COMMAND_LENGTH = int(DATASET_SAMPLE_RATE*1.2)


def preprocess_dataset(data_path, dataset_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from dataset and saves them into a json file.
    :param data_path (str): Path to data
    :param dataset_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): FFT length in samples
    :param hop_length (int): Sliding window for FFT in samples
    :return:
    """

    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path)):

        if dirpath is not data_path:
            # save label (i.e., sub-folder name) in the mapping
            label = os.path.split(dirpath)[1]
            data["mapping"].append(label)
            print(f"\nProcessing: {label}")

            # process all audio files in sub-dir and store features
            for f in filenames:
                file_path = os.path.join(dirpath, f)

            #%% extract features: MFCC
                # load audio file
                signal, sample_rate = librosa.load(file_path, sr=None)

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

                #%% store data for analysed audio file
                data["MFCCs"].append(MFCCs.tolist())
                data["labels"].append(i-1)
                data["files"].append(file_path)
                print("{}: {}".format(file_path, i-1))

    # save dataset in json file
    with open(dataset_path, "w") as fp:
        json.dump(data, fp, indent=4)
    return data


if __name__ == "__main__":
    preprocess_dataset(DATA_PATH, DATASET_PATH)