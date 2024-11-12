# Honk Honk

Car Horn segmentation using Yolo. This is a work in progress...

## Getting Started
`main.py` is used for training the network. `client.py` is for testing with real data

1. Install dependencies
2. Run `dataset.download()` in `main.py`
3. Download [this](https://www.youtube.com/watch?v=unfmzc0opk4) video as a mp3 and save it in the root directory
3. Run `prepare_data`
4. Run `train`
5. Update the `YOLO` in `client.py` to the path of your trained model
6. Run `client.py`