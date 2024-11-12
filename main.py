import random
import time
from pathlib import Path

from ultralytics import YOLO
import soundata
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
import numpy as np
import yaml
import pyaudio

# https://github.com/GorillaBus/urban-audio-classifier/blob/master/3-cnn-model-mfcc.ipynb
# https://github.com/mpolinowski/yolo-listen/blob/master/02_YOLO_Classifier.ipynb
def prepare_data(dataset):
    clipid_to_clip = dataset.load_clips()

    label_map = [
        'air_conditioner',
        'car_horn',
        'children_playing',
        'dog_bark',
        'drilling',
        'engine_idling',
        'gun_shot',
        'jackhammer',
        'siren',
        'street_music'
    ]

    # Set up the training config to work with YOLO
    # https://roboflow.com/formats/yolov5-pytorch-txt?ref=ultralytics
    training_config = {
        "train": "train/images",
        "val": "validate/images",
        "test": "test/images",
        "nc": len(label_map),
        "names" : label_map
    }

    with open("dataset/data.yaml", "w") as f:
        yaml.dump(training_config, f)

    # Now prepare the actual data
    for clip_id, clip in clipid_to_clip.items():
        if("bimodal" not in clip_id): # For now bimodal seems too hard, let's skip that
            batch = clip.split # The dataset provides recommended test/train splits
            im = preprocess(clip.audio_path)
            labels = get_labels(clip.txt_path, label_map)
            print(clip_id)
            Path(f"datasets/{batch}/images").mkdir(parents=True, exist_ok=True)
            Path(f"datasets/{batch}/labels").mkdir(parents=True, exist_ok=True)
            im.save(f"datasets/{batch}/images/{clip_id}.png")
            with open(f"datasets/{batch}/labels/{clip_id}.txt", "w") as f:
                for label in labels:
                    # TODO: the ultraanalytics library automatically skips labels outside of bounds but maybe we should filter them anyway
                    if(label[1] < 320 - 50): # we want to filter out any labels that are fully cropped out
                        # label format is provided by ultralytics library: x_center y_center width height
                        # https://github.com/ultralytics/yolov5/issues/2293
                        f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")

def preprocess(input_path):
    y, sr = librosa.load(input_path)
    normalized_y = librosa.util.normalize(y[:220500]) # Crop clip since some are 1 frame longer

    normalized_y = add_background_sound(normalized_y)
    # play_audio(normalized_y)

    stft = librosa.core.stft(normalized_y, n_fft=2048, hop_length=512)
    mel = librosa.feature.melspectrogram(S=stft, n_mels=256)

    # Convert sound intensity to log amplitude:
    mel_db = librosa.amplitude_to_db(abs(mel))

    # Normalize between 0 and 255
    normalized_mel = (librosa.util.normalize(mel_db) + 1) * 255

    # YOLO likes multiples of 160x160 so let's make sure we have a constant size
    # Pad any heights that are too short
    padded = np.pad(normalized_mel, ((0,320-normalized_mel.shape[0]),(0,0)), 'constant', constant_values=-1)

    # We're also going to crop our audio to make things easy. We will revisit this if things don't work
    padded = padded[:,:320]

    # Convert to a png for writing
    return ImageOps.flip(Image.fromarray(padded).convert('L'))

def add_background_sound(normalized_y):
    # The training dataset includes some background noise, but it seems too clean to me.
    # I downloaded the following video and we will add random snippets from it to our audio.
    # https://www.youtube.com/watch?v=unfmzc0opk4
    y, sr = librosa.load("city-background.mp3", offset=random.randint(0,21600), duration=10) # TODO: should we make this deterministic?
    normalized_background = librosa.util.normalize(y)

    # return np.minimum(normalized_y * 1 + normalized_background * 0.0,1)
    return np.minimum(normalized_y * 0.60 + normalized_background * 0.40,1)

def get_labels(input_path, label_map):
    clip_length_seconds = 10
    clip_length_columns = 431
    output_size = 320
    result = []
    with open(input_path, "r") as f:
        for line in f:
            # Yolo expects us to provide bounding boxes of the form x_center y_center width height (normalized)
            #  https://github.com/ultralytics/yolov5/issues/2293
            label_line = line.strip().split('\t')
            start = float(label_line[0]) / clip_length_seconds * clip_length_columns
            end = float(label_line[1]) / clip_length_seconds * clip_length_columns
            width = end - start
            height = 256.0
            x_center = start + (width/2)
            y_center = output_size - (height / 2)
            label = label_map .index(label_line[2])
            result.append((label, x_center/output_size, y_center/output_size, width/output_size, height/output_size))
    return result

def visualize_data(dataset):
    # Useful function for verifying preprocessing steps
    clip = dataset.choice_clip()
    with open(clip.txt_path, "r") as f:
        print(f.read())

    labels = get_labels(clip.txt_path)

    plt.imshow(preprocess(clip.audio_path))
    for label in labels:
        print(label)
        rect = Rectangle(
            ((label[1]-(label[3]/2))*320, (label[2]-(label[4]/2))*320),
            label[3] * 320,
            label[4] * 320, linewidth=1, edgecolor=np.random.rand(3,), facecolor='none')
        plt.gca().add_patch(rect)

    plt.pause(0.1)
    # Uncomment to hear audio with visual
    # y,sr = librosa.load(clip.audio_path)
    # play_audio(y)
    plt.show()

def play_audio(audio_data):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=22050,
                    output=True)
    # Weird that we have to do tobytes()
    # https://stackoverflow.com/questions/34222517/why-does-pyaudio-only-play-half-my-audio
    stream.write((audio_data * 32767).astype(np.int16).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

def train():
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    results = model.train(data="data.yaml", epochs=10, imgsz=320, device="mps")
    print(model.val())
    success = model.export(format="onnx")

if __name__ == "__main__":
    dataset = soundata.initialize('urbansed')
    # dataset.download()  # download the dataset
    # visualize_data(dataset)
    # prepare_data(dataset)
    train()


