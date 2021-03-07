import boto3
from playsound import playsound
from threading import Thread
import os

polly_client = boto3.Session(profile_name='default', region_name='us-east-1').client('polly')

def play_music():
    playsound('speech.mp3')
    os.remove('speech.mp3')

def speakText(inputText):
    response = polly_client.synthesize_speech(VoiceId='Joanna', OutputFormat='mp3', Text = inputText)

    file = open('speech.mp3', 'wb')
    file.write(response['AudioStream'].read())
    file.close()

    music_thread = Thread(target=play_music)
    music_thread.start()