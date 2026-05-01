import platform
import queue
import re
import subprocess
import threading


class AudioEngine:
    def __init__(self):
        self._queue = queue.Queue()
        self._current_process = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text):
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return
        if len(text) > 1800:
            text = text[:1800] + "..."
        self.clear_queue()
        self.stop()
        self._queue.put(text)

    def clear_queue(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def stop(self):
        proc = self._current_process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _worker(self):
        while True:
            text = self._queue.get()
            try:
                if platform.system() == "Darwin":
                    self._current_process = subprocess.Popen(["say", text])
                    self._current_process.wait()
                    self._current_process = None
                else:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 150)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
            except Exception:
                self._current_process = None
