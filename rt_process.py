import sounddevice as sd
import numpy as np
import librosa
import pyworld
from espnet2.bin.asr_inference import Speech2Text
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import threading
import queue
import time
import matplotlib.pyplot as plt
from pythonosc import udp_client

"""
Real-Time Pitch Detection and ASR

This program demonstrates real-time pitch detection and automatic speech recognition (ASR) using the ESPnet2 and Hugging Face libraries.

ASR and sentiment detection link (unused):
https://github.com/espnet/espnet/tree/ec9760b22654dc04eeecd37e2659ebda0325a786/egs2/iemocap/asr1

ASR fast model link:
https://huggingface.co/openai/whisper-base.en

Text emotion classification link:
https://huggingface.co/michellejieli/emotion_text_classifier

anger 🤬
disgust 🤢
fear 😨
joy 😀
neutral 😐
sadness 😭
surprise 😲

"""

# Set up the OSC client
osc_ip = "127.0.0.1"  # IP address of the OSC server (Max/MSP)
osc_port = 8000  # Port number of the OSC server
osc_client = udp_client.SimpleUDPClient(osc_ip, osc_port)

# define the ASR model and text emotion classifier
# asr_model = Speech2Text.from_pretrained(
#     "espnet/YushiUeda_iemocap_sentiment_asr_train_asr_conformer_hubert"
# )

# Use model from "https://huggingface.co/openai/whisper-base.en"
processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")


sentiment_classifier = pipeline(
    "sentiment-analysis", model="michellejieli/emotion_text_classifier"
)

# set the recording parameters
pitch_duration = 0.1  # Duration for pitch detection
asr_duration = 3  # Duration for ASR
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
        # print(f"Pitch: {avg_audio_f0}")

        threshold = 0.05
        if audio_volume > threshold and avg_audio_f0 > 50 and avg_audio_f0 < 500:
            osc_client.send_message("/pitch", avg_audio_f0)  # Send pitch to Max/MSP


def get_rgb_color(emo_index, sentiment_index=2):
    # define the color for each emotion
    colors = {
        1: (225, 50, 50),  # anger - red
        2: (50, 128, 50),  # disgust - green
        3: (200, 50, 205),  # fear - purple
        4: (225, 225, 50),  # joy - yellow
        5: (225, 225, 225),  # neutral - white
        6: (50, 180, 225),  # sadness - blue
        7: (225, 165, 50),  # surprise - orange
    }

    # based on the sentiment and emotion, adjust the color
    base_color = colors.get(emo_index, (128, 128, 128))  # default to gray
    if sentiment_index == 1:  # Neutral
        return base_color
    elif sentiment_index == 2:  # Positive
        # increase the color value by 30
        return tuple(min(255, c + 30) for c in base_color)
    elif sentiment_index == 3:  # Negative
        # decrease the color value by 30
        return tuple(max(0, c - 30) for c in base_color)
    else:
        return base_color


# emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
# sentiments = ["Neutral", "Positive", "Negative"]

# for emo_index, emotion in enumerate(emotions, start=1):
#     for sentiment_index, sentiment in enumerate(sentiments, start=1):
#         color = get_rgb_color(emo_index, sentiment_index)
#         print(f"{emotion} + {sentiment}: RGB{color}")
# raise ValueError

"""
anger + Neutral: RGB(225, 50, 50)
anger + Positive: RGB(255, 80, 80)
anger + Negative: RGB(195, 20, 20)
disgust + Neutral: RGB(50, 128, 50)
disgust + Positive: RGB(80, 158, 80)
disgust + Negative: RGB(20, 98, 20)
fear + Neutral: RGB(200, 50, 205)
fear + Positive: RGB(230, 80, 235)
fear + Negative: RGB(170, 20, 175)
joy + Neutral: RGB(225, 225, 50)
joy + Positive: RGB(255, 255, 80)
joy + Negative: RGB(195, 195, 20)
neutral + Neutral: RGB(225, 225, 225)
neutral + Positive: RGB(255, 255, 255)
neutral + Negative: RGB(195, 195, 195)
sadness + Neutral: RGB(50, 180, 225)
sadness + Positive: RGB(80, 210, 255)
sadness + Negative: RGB(20, 150, 195)
surprise + Neutral: RGB(225, 165, 50)
surprise + Positive: RGB(255, 195, 80)
surprise + Negative: RGB(195, 135, 20)
"""


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
            # text, *_ = asr_model(asr_audio)[0]
            # print(f"ASR Text: {text}")
            start_time = time.time()
            input_features = processor(
                asr_audio, sampling_rate=sample_rate, return_tensors="pt"
            ).input_features
            predicted_ids = model.generate(input_features)
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"ASR Time: {time.time() - start_time}")

            print("ASR: ", text)
            # try:
            #     sentiment, text = text.split(" ", 1)
            # except ValueError:
            #     sentiment = "Neutral"
            #     text = ""
            # print(f"Sentiment: {sentiment}")
            # print(f"Recognized Text: {text}")

            # Text emotion classification
            if text != "":
                output = sentiment_classifier(text)
                print(f"Emotion: {output}")

                if output[0]["label"] == "anger":
                    emo_index = 1
                elif output[0]["label"] == "disgust":
                    emo_index = 2
                elif output[0]["label"] == "fear":
                    emo_index = 3
                elif output[0]["label"] == "joy":
                    emo_index = 4
                elif output[0]["label"] == "neutral":
                    emo_index = 5
                elif output[0]["label"] == "sadness":
                    emo_index = 6
                elif output[0]["label"] == "surprise":
                    emo_index = 7
                else:
                    emo_index = 0
            else:
                emo_index = 5
                output = None

            # if sentiment == "Neutral":
            #     sentiment_index = 1
            # elif sentiment == "Positive":
            #     sentiment_index = 2
            # elif sentiment == "Negative":
            #     sentiment_index = 3
            # else:
            #     sentiment_index = 0

            light_color = get_rgb_color(emo_index)
            print(f"Light Color: {light_color}")

            osc_client.send_message("/text", text)
            # osc_client.send_message("/sentiment", sentiment_index)
            osc_client.send_message("/emotion", emo_index)
            if output:
                osc_client.send_message("/confidence", output[0]["score"])
            r, g, b = light_color
            osc_client.send_message("/rgb", [r, g, b])

            asr_audio = np.array([])  # Clear the audio array for the next ASR


# Callback function for the audio stream
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio = indata[:, 0]  # Get mono audio
    audio_queue.put(audio.copy())  # Put audio in the queue for pitch detection
    asr_audio_queue.put(audio.copy())  # Put audio in the queue for ASR


def process_realtime():
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


def process_audiofile(path):
    audio, sr = librosa.load(path, sr=sample_rate)

    frame_size = int(
        pitch_duration * sample_rate
    )  # Size of each frame for pitch detection
    asr_frame_size = int(asr_duration * sample_rate)  # Size of each frame for ASR

    asr_audio = np.array([])  # Initialize an empty array for ASR

    for i in range(0, len(audio), frame_size):
        frame = audio[i : i + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(
                frame, (0, frame_size - len(frame)), "constant"
            )  # Pad the last frame if needed

        # Process pitch detection
        pitch_detection_audio(frame)

        # Concatenate frame for ASR
        asr_audio = np.concatenate((asr_audio, frame))

        # Check if it's time to process ASR
        if len(asr_audio) >= asr_frame_size:
            asr_recognition_audio(
                asr_audio[:asr_frame_size]
            )  # Process the first 5s of audio for ASR
            asr_audio = asr_audio[asr_frame_size:]  # Remove the processed audio

        # wait to simulate real-time processing
        time.sleep(pitch_duration)


# Modified pitch detection function
def pitch_detection_audio(audio):
    # Extract pitch using PyWorld
    audio_f0, timeaxis = pyworld.harvest(audio.astype(np.double), sample_rate)
    audio_f0 = pyworld.stonemask(
        audio.astype(np.double), audio_f0, timeaxis, sample_rate
    )
    avg_audio_f0 = np.nanmean(audio_f0)  # Get the average pitch

    # Extract volume
    audio_volume = np.sqrt(np.mean(audio**2))

    # Print pitch information
    # print(f"Pitch: {avg_audio_f0}")

    threshold = 0.05
    if audio_volume > threshold and avg_audio_f0 > 50 and avg_audio_f0 < 500:
        osc_client.send_message("/pitch", avg_audio_f0)  # Send pitch to Max/MSP


# Modified ASR function
def asr_recognition_audio(audio):
    # Speech to sentiment and text
    text, *_ = asr_model(audio)[0]
    try:
        sentiment, text = text.split(" ", 1)
    except ValueError:
        sentiment = "Neutral"
        text = ""
    print(f"Sentiment: {sentiment}")
    print(f"Recognized Text: {text}")

    # Text emotion classification
    if text != "":
        output = sentiment_classifier(text)
    print(f"Emotion: {output}")

    if output[0]["label"] == "anger":
        emo_index = 1
    elif output[0]["label"] == "disgust":
        emo_index = 2
    elif output[0]["label"] == "fear":
        emo_index = 3
    elif output[0]["label"] == "joy":
        emo_index = 4
    elif output[0]["label"] == "neutral":
        emo_index = 5
    elif output[0]["label"] == "sadness":
        emo_index = 6
    elif output[0]["label"] == "surprise":
        emo_index = 7
    else:
        emo_index = 0

    if sentiment == "Neutral":
        sentiment_index = 1
    elif sentiment == "Positive":
        sentiment_index = 2
    elif sentiment == "Negative":
        sentiment_index = 3
    else:
        sentiment_index = 0

    light_color = get_rgb_color(emo_index, sentiment_index)
    print(f"Light Color: {light_color}")

    osc_client.send_message("/text", text)
    osc_client.send_message("/sentiment", sentiment_index)
    osc_client.send_message("/emotion", emo_index)
    osc_client.send_message("/confidence", output[0]["score"])
    r, g, b = light_color
    osc_client.send_message("/rgb", [r, g, b])


if __name__ == "__main__":
    process_realtime()

    # Process the audio file
    # process_audiofile(
    #     "/Users/yyf/Documents/GitHub/hackathon2024/barackobamasenatespeechrosaparks.mp3"
    # )
