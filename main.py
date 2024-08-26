# full conversational model

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from gtts import gTTS
from playsound import playsound
import os
import speech_recognition as sr


# installation requirements (pip install name)
# langchain, langchain-ollama, ollama
# pydantic, gtts, playsound (pip install playsound==1.2.2)
# SpeechRecognition, pyaudio, setuptools
# AppKit or portaudio or pyobjc
# download Ollama and then use "Ollama pull llama3" in your devices terminal - this program uses the llama3 model only
# "python3 main.py" in terminal to run

# template passed to model
template = """
Here is history: {context}
Here is question: {question}
Now you answer:
"""

# chaining response
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# initialise the speech recognizer
recognizer = sr.Recognizer()

# listen via microphone, transcribe and pass to model
def listen():
    """Capture user's speech via microphone and convert it to text."""
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""

# process chatbot response into text to speech
def speak(text):
    """Convert text to speech, play it, and remove the audio file."""
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("response.mp3")
        playsound("response.mp3")
    finally:
        if os.path.exists("response.mp3"):
            os.remove("response.mp3")  

# creates the chatbot response
def handle():
    """Main function to handle the chatbot conversation."""
    print("Welcome to Zain's ChatBot | Say 'exit' to leave")
    context = ""
    
    while True:
        userinput = listen()
        if userinput.lower() == "exit":
            break
        if userinput:
            # get the chatbot's response
            result = chain.invoke({"context": context, "question": userinput})
            print("ChatBot:", result)
            
            # convert the response to speech and play it
            speak(result)

            # update the context with the latest conversation
            context += f"\nUser: {userinput}\nAI: {result}"

if __name__ == "__main__":
    handle()
