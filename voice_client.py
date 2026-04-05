import requests
import time
import sys
import threading
import queue

import speech_recognition as sr
import pyttsx3
import keyboard

PI_IP = "192.168.1.15"
PORT = 8000

URL = f"http://{PI_IP}:{PORT}/generate"
HEALTH_URL = f"http://{PI_IP}:{PORT}/health"

WAKE_WORD = "hey klvr"
EXIT_COMMANDS = {"exit", "quit", "stop program"}

recognizer = sr.Recognizer()
mic = sr.Microphone()

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 180)

trigger_queue = queue.Queue()
running = True
busy = False


def speak(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print("[TTS ERROR]", e)


def check_health():
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        print("Server health:", r.json())
    except Exception as e:
        print("Health check failed:", e)


def ask_llm(prompt):
    global busy
    busy = True

    full_response = ""

    try:
        response = requests.post(
            URL,
            json={"text": prompt},
            stream=True,
            timeout=120
        )

        if response.status_code != 200:
            print("Server Error:", response.text)
            busy = False
            return

        rag_used = response.headers.get("X-RAG-Used", "false")
        print(f"\n[RAG used: {rag_used}]")
        print("Assistant: ", end="", flush=True)

        for chunk in response.iter_content(chunk_size=1):
            if chunk:
                char = chunk.decode("utf-8", errors="ignore")
                full_response += char
                sys.stdout.write(char)
                sys.stdout.flush()

        print("\n")

        if full_response.strip():
            speak(full_response.strip())

    except Exception as e:
        print("\nConnection Error:", e)

    busy = False


def listen_once(timeout=5, phrase_time_limit=8):
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text.lower().strip()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print("Speech recognition request failed:", e)
        return None
    except Exception as e:
        print("Listen error:", e)
        return None


def manual_trigger_listener():
    global running
    print("Press 'm' to talk, 'q' to quit.")

    while running:
        try:
            if keyboard.is_pressed("m"):
                if not busy:
                    trigger_queue.put("manual")
                    time.sleep(1)
            elif keyboard.is_pressed("q"):
                running = False
                trigger_queue.put("quit")
                break
            time.sleep(0.1)
        except Exception:
            pass


def wake_word_listener():
    global running

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"Wake word listener active. Say '{WAKE_WORD}' to activate.")

    while running:
        if busy:
            time.sleep(0.2)
            continue

        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)

            try:
                text = recognizer.recognize_google(audio).lower().strip()
                if text:
                    print(f"[Heard]: {text}")

                if WAKE_WORD in text:
                    print("Wake word detected!")
                    trigger_queue.put("wake")
                    time.sleep(1)

            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print("Wake word recognition error:", e)
                time.sleep(1)

        except sr.WaitTimeoutError:
            pass
        except Exception as e:
            print("Wake listener error:", e)
            time.sleep(1)


def interaction_loop():
    global running

    while running:
        try:
            trigger = trigger_queue.get(timeout=1)
        except queue.Empty:
            continue

        if trigger == "quit":
            break

        if busy:
            continue

        print("\nActivated. Speak your query.")
        query = listen_once(timeout=8, phrase_time_limit=10)

        if not query:
            continue

        if query in EXIT_COMMANDS:
            running = False
            break

        ask_llm(query)


def main():
    global running

    print("Voice Assistant Client")
    print(f"Wake word: '{WAKE_WORD}'")
    print("Press 'm' to manually speak")
    print("Press 'q' to quit\n")

    check_health()
    speak("Voice assistant is ready")

    t1 = threading.Thread(target=manual_trigger_listener, daemon=True)
    t2 = threading.Thread(target=wake_word_listener, daemon=True)

    t1.start()
    t2.start()

    try:
        interaction_loop()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        print("Exiting...")
        speak("Shutting down")


if __name__ == "__main__":
    main()
