import time
from collections import deque

import librosa
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pygments.formatters import img
from ultralytics import YOLO
from PIL import Image, ImageOps
from norfair import Detection, Tracker, Video, draw_tracked_objects, draw_points

# Parameters
FORMAT = pyaudio.paInt16   # 16-bit resolution
CHANNELS = 1               # Mono audio
RATE = 22050               # Sampling rate (Hz) to match librosa default
CHUNK = 2048              # should match fft size

def yolo_result_to_no_fair_detections(yolo_result):
    boxes = yolo_result.boxes.xywh.cpu().tolist()
    classes = yolo_result.boxes.cls.int().cpu().tolist()
    class_name_map = yolo_result.names
    labels = [class_name_map[class_id] for class_id in classes]
            # classes = results[0].boxes.cls.int().cpu().tolist()
            # class_name_map = results[0].names
    result = []
    for (x, y, w, h), label in zip(boxes,labels):
        result.append(Detection(np.array([x, y]), label=label))

    return result

tracker = Tracker(distance_function="euclidean", distance_threshold=100)

# Function to update the plot
def update(frame):
    try:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        as_float = data.astype(np.float32) / 2.0**7 # Half of the 16 bit
        normalized_y = librosa.util.normalize(as_float)
        stft = librosa.core.stft(normalized_y, n_fft=2048, hop_length=512)
        mel = librosa.feature.melspectrogram(S=stft, n_mels=256)
        mel_db = librosa.amplitude_to_db(abs(mel))
        normalized_mel = (librosa.util.normalize(mel_db) + 1) * 255

        padded = np.pad(normalized_mel, ((0, 320 - normalized_mel.shape[0]),(0,0)), 'constant', constant_values=-1)

        audio_buffer.append(padded)
        buffer_data = np.concatenate(list(audio_buffer), axis=1) # TODO: is this efficient?

        if(buffer_data.shape[1] > 320):
            input_image = ImageOps.flip(Image.fromarray(buffer_data[:,-320:]).convert('L'))

            # Get the result of the model prediction
            result = model.predict(input_image, imgsz=320, device='mps', conf=0.01, verbose=False)[0]


            # Pass those predictions to our tracker
            # detections = yolo_result_to_no_fair_detections(result)
            # tracked_objects = tracker.update(detections=detections)
            # draw_points(buffer_data, tracked_objects, text_size=.5, draw_scores=True)

            prediction = result[0]
            labels = result[0].names
            result = ""
            for box in prediction.boxes:
                result += f"{labels[box.cls.item()]} ({box.conf.item() * 100:.2f}%),"
            print(result)


        img.set_array(buffer_data)

    except OSError as e:
        print("failed to read buffer", e)

    return [img]

if __name__ == "__main__":
    # Initialize pyaudio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # Set up the figure and spectrogram plot
    fig, ax = plt.subplots()
    x = np.arange(0, 2 * CHUNK, 2)
    img = ax.imshow(np.zeros((128, CHUNK // 2 + 1)), origin='lower', aspect='auto', cmap='inferno', interpolation='nearest', vmin=0, vmax=255)

    # Initialize a deque to store the rolling buffer of audio chunks
    audio_buffer = deque(maxlen=100)

    model = YOLO("runs/detect/train30/weights/best.pt")

    # Animation function
    ani = FuncAnimation(fig, update, blit=True, interval=50)

    # Show the plot
    plt.show()

    time.sleep(10)

    # Close the stream gracefully when done
    stream.stop_stream()
    stream.close()
    audio.terminate()
