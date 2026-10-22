
from sarvamai import SarvamAI
import base64


class SarvamSpeech:

    def __init__(self, api_key):

        self.client = SarvamAI(
            api_subscription_key=api_key
        )


    def transcribe(self, audio_path):

        with open(audio_path, "rb") as audio_file:

            response = self.client.speech_to_text.transcribe(
                file=audio_file,
                model="saaras:v3",
                mode="transcribe"
            )

        return response.transcript


    def speak(
        self,
        text,
        output_path="answer.wav"
    ):

        response = self.client.text_to_speech.convert(
            text=text,
            language_code="en-IN",
            speaker="aditya",
            model="bulbul:v3",
            output_audio_codec="wav"
        )

        audio_bytes = base64.b64decode(
            response.audios[0]
        )

        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        return output_path
