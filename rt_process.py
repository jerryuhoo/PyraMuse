import sounddevice as sd
import numpy as np
import pyworld
from espnet2.bin.asr_inference import Speech2Text
from transformers import pipeline
import threading
import queue
import time
import matplotlib.pyplot as plt
from pythonosc import udp_client

# Set up the OSC client
osc_ip = "127.0.0.1"  # IP address of the OSC server (Max/MSP)
osc_port = 8000  # Port number of the OSC server
osc_client = udp_client.SimpleUDPClient(osc_ip, osc_port)

# define the ASR model and text emotion classifier
asr_model = Speech2Text.from_pretrained(
    "espnet/YushiUeda_iemocap_sentiment_asr_train_asr_conformer_hubert"
)
sentiment_classifier = pipeline(
    "sentiment-analysis", model="michellejieli/emotion_text_classifier"
)

# set the recording parameters
pitch_duration = 0.1  # Duration for pitch detection
asr_duration = 5  # Duration for ASR
sample_rate = 16000
audio_queue = queue.Queue()
asr_audio_queue = queue.Queue()


# Pitch detection function
def pitch_detection():
    while True:
        audio = audio_queue.get()  # Get audio from the queue
        if audio is None:
            break  # Stop the thread if None is received

        # Extract pitch using PyWorld
        audio_f0, timeaxis = pyworld.harvest(audio.astype(np.double), sample_rate)
        audio_f0 = pyworld.stonemask(
            audio.astype(np.double), audio_f0, timeaxis, sample_rate
        )
        avg_audio_f0 = np.nanmean(audio_f0)  # Get the average pitch

        # Extract volume
        audio_volume = np.sqrt(np.mean(audio**2))
        # print(f"Volume: {audio_volume}")

        # Print pitch information
        print(f"Pitch: {avg_audio_f0}")

        threshold = 0.05
        if audio_volume > threshold and avg_audio_f0 > 50 and avg_audio_f0 < 500:
            osc_client.send_message("/pitch", avg_audio_f0)  # Send pitch to Max/MSP


# ASR function
def asr_recognition():
    asr_audio = np.array([])  # Initialize an empty array to store audio for ASR
    while True:
        audio = asr_audio_queue.get()  # Get audio from the queue
        if audio is None:
            break  # Stop the thread if None is received

        asr_audio = np.concatenate((asr_audio, audio))  # Concatenate the audio for ASR

        if len(asr_audio) >= asr_duration * sample_rate:
            # Speech to sentiment and text
            text, *_ = asr_model(asr_audio)[0]

            # Text emotion classification
            output = sentiment_classifier(text)

            print(f"Recognized Text: {text}")
            print(f"Emotion: {output}")

            osc_client.send_message("/text", text)
            osc_client.send_message("/emotion", output[0]["label"])

            asr_audio = np.array([])  # Clear the audio array for the next ASR


# Callback function for the audio stream
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio = indata[:, 0]  # Get mono audio
    audio_queue.put(audio.copy())  # Put audio in the queue for pitch detection
    asr_audio_queue.put(audio.copy())  # Put audio in the queue for ASR


# Start the audio stream
with sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    callback=callback,
    blocksize=int(pitch_duration * sample_rate),
):
    print("Start Real-Time Pitch Detection and ASR, Press Ctrl+C to stop.")

    # Start the pitch detection thread
    pitch_thread = threading.Thread(target=pitch_detection)
    pitch_thread.start()

    # Start the ASR thread
    asr_thread = threading.Thread(target=asr_recognition)
    asr_thread.start()

    try:
        while True:
            time.sleep(0.1)  # Main thread can sleep for a short duration
    except KeyboardInterrupt:
        audio_queue.put(None)  # Stop the threads
        asr_audio_queue.put(None)
        pitch_thread.join()
        asr_thread.join()
        print("Stopped Real-Time Pitch Detection and ASR")
