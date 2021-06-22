# Command recognition API
Speech recognition system that can recognize predefined spoken words: down, off, on, no, yes, stop, up, right, left, go.

The system was build using CNN (convolutional neural network) and MFCCs (mel-frequency cepstral coefficients) in Tensorflow and Keras. 
The CNN was trained on the [Speech command dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html).

The API was implemented with the Flask framework and deployed with uWSGI, NGINX and Docker.
 
