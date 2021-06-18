import requests
import os
from command_recognition.build_features import DATA_PATH

# endpoint for command prediction
URL = "http://127.0.0.1:5000/predict-command"

# load test audio file for prediction
file_path = os.listdir(os.path.join(DATA_PATH, "down"))[0]

if __name__ == "__main__":

    # open file
    file = open(file_path, "rb")

    # make POST request
    values = {"file": (file_path, file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print("Predicted command: {}".format(data["command"]))
