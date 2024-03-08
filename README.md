# Hackathon2024 Project - PyraMuse

## About
PyraMuse is a musical interface that transforms voice into music. We mostly share our emotions by talking in our everyday lives. Our idea is to create music based on the emotions expressed in speech. It detects the emotions in what we say and uses the pitch of our speech as input. By speaking into a microphone, users can create music reflecting their emotions. The system facilitates control where the left hand adjusts the pitch, and the right hand modulates the vibrato through two ultrasonic sensors, providing visual feedback with an LED light to reflect the user's emotional state.

## Techniques
The voice signal is captured by a microphone. We use the speech recognition model "Whisper" in Python to identify words and a text emotion recognition model to detect emotions. The pitch of the voice is detected in Python and sent to Max via OSC protocol. Ultrasonic sensors measure the distance of each hand from the sensors, with Arduino gathering the data. This data is then sent to Max and Ableton to control the pitch and the vibrato effects. The MIDI information is sent to Ableton to generate musical compositions. There are four types of music—joy, sadness, anger, and neutrality—are mapped to four patterns in Ableton. Additionally, Max communicates the emotion data to Arduino via serial communication to control the LED light, with each emotion mapping to a distinct light color.

