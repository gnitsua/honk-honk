import json
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO
import soundata
import torch
from playsound import playsound
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import yaml
import pyaudio
from torch.utils.data import Dataset, DataLoader

# https://github.com/GorillaBus/urban-audio-classifier/blob/master/3-cnn-model-mfcc.ipynb
# https://github.com/mpolinowski/yolo-listen/blob/master/02_YOLO_Classifier.ipynb
def prepare_data(dataset):
    clipid_to_clip = dataset.load_clips()

    # Dataset specifically specifies that images from different folds should not be mixed. Start out with 80/20 training mix
    # https://urbansounddataset.weebly.com/urbansound.html
    dataset_map = {
        1: "train",
        2: "train",
        3: "train",
        4: "train",
        5: "train",
        6: "train",
        7: "train",
        8: "train",
        9: "test",
        10: "validate",
    }

    labels = [
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
        "nc": len(labels),
        "names" : labels
    }
    with open("data.yaml", "w") as f:
        yaml.dump(training_config, f)

    # Now prepare the actual data
    for clip_id, clip in clipid_to_clip.items():
        batch = dataset_map[clip.fold]
        data = preprocess(clip.audio_path)
        im = ImageOps.flip(Image.fromarray(data).convert('L'))
        print(clip_id)
        Path(f"datasets/{batch}/images").mkdir(parents=True, exist_ok=True)
        Path(f"datasets/{batch}/labels").mkdir(parents=True, exist_ok=True)
        im.save(f"datasets/{batch}/images/{clip_id}.png")
        with open(f"datasets/{batch}/labels/{clip_id}.txt", "w") as f:
            # For now we are going to put a bounding box around the entire clip
            f.write(f"{labels.index(clip.class_label)} 0.5 0.5 1 1\n")

def preprocess(input_path):
    y, sr = librosa.load(input_path)
    normalized_y = librosa.util.normalize(y)
    stft = librosa.core.stft(normalized_y, n_fft=2048, hop_length=512)
    mel = librosa.feature.melspectrogram(S=stft, n_mels=128)

    # Convert sound intensity to log amplitude:
    mel_db = librosa.amplitude_to_db(abs(mel))

    # Normalize between 0 and 255
    normalized_mel = (librosa.util.normalize(mel_db) + 1) * 255

    # YOLO likes multiples of 160x160 so let's make sure we have a constant size
    # Pad any heights that are too short
    padded = np.pad(normalized_mel, ((0,160-normalized_mel.shape[0])), 'constant', constant_values=-1)

    # Pad any widths that are too short, or crop if too large
    if(padded.shape[1] > 160):
        padded = padded[:,:160] # crop width
    elif(padded.shape[1] < 160):
        padded = np.pad(padded, ((0,0),(0,160-padded.shape[1])), 'constant', constant_values=-1)

    assert(padded.shape == (160, 160))

    return padded

def visualize_data(spectrogram):
    # Plot spectrogram from STFT
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MEL-Scaled Spectrogram')
    plt.tight_layout()
    plt.show()
    # playsound(clip.audio_path)

# class SoundDataset(Dataset):
#     """Face Landmarks dataset."""
#
#     def __init__(self):
#         """
#         Arguments:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         pass
#
#     def __len__(self):
#         return len(self.landmarks_frame)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample

def train2():
    pass

def train():
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    results = model.train(data="data.yaml", epochs=100, imgsz=160, device="mps")
    print(model.val())
    success = model.export(format="onnx")

if __name__ == "__main__":
    # dataset = soundata.initialize('urbansound8k')
    # dataset.download()  # download the dataset
    # visualize_data(dataset)
    prepare_data(dataset)
    # train()


