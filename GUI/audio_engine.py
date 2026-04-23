import queue
import threading

import pyttsx3


class AudioEngine:
    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text):
        """Queues text to be spoken. Non-blocking, safe to call from the GUI thread."""
        if text:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put(text)

    def _worker(self):
        """Background thread that processes speech requests with a fresh engine each time."""
        # Initialize COM for this thread (required on Windows)
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        while True:
            text = self._queue.get()  # blocks until something is available
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                print(f"TTS Error: {e}")