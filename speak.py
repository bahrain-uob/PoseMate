import boto3
from playsound import playsound
import os

polly_client = boto3.Session(profile_name='default', region_name='us-east-1').client('polly')

def speakText(inputText):

    response = polly_client.synthesize_speech(VoiceId='Joanna', OutputFormat='mp3', Text = inputText)

    file = open('speech.mp3', 'wb')
    file.write(response['AudioStream'].read())
    file.close()

    playsound('speech.mp3')
    os.remove('speech.mp3')