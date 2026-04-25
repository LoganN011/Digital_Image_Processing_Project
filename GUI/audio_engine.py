import platform
import queue
import subprocess
import threading

# Compatible with both windows/linux and macOS 
class AudioEngine:
    def __init__(self):
        self._queue = queue.Queue()
        self._current_process = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text):
        """Queues text to be spoken. Non-blocking, safe to call from the GUI thread."""
        if not text:
            return

        # Clear older queued speech so only the newest poster is read.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Stop current speech if possible.
        self.stop()

        self._queue.put(text)

    def stop(self):
        """Stops current speech if a subprocess is running."""
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
                    # macOS: use built-in say command instead of pyttsx3/AppKit.
                    self._current_process = subprocess.Popen(["say", text])
                    self._current_process.wait()
                    self._current_process = None
                else:
                    # Fallback for Windows/Linux.
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 150)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()

            except Exception as e:
                print(f"TTS Error: {e}")

# import queue
# import threading

# import pyttsx3


# class AudioEngine:
#     def __init__(self):
#         self._queue = queue.Queue()
#         self._thread = threading.Thread(target=self._worker, daemon=True)
#         self._thread.start()

#     def speak(self, text):
#         """Queues text to be spoken. Non-blocking, safe to call from the GUI thread."""
#         if text:
#             while not self._queue.empty():
#                 try:
#                     self._queue.get_nowait()
#                 except queue.Empty:
#                     break
#             self._queue.put(text)

#     def _worker(self):
#         """Background thread that processes speech requests with a fresh engine each time."""
#         # Initialize COM for this thread (required on Windows)
#         try:
#             import pythoncom
#             pythoncom.CoInitialize()
#         except ImportError:
#             pass

#         while True:
#             text = self._queue.get()  # blocks until something is available
#             try:
#                 engine = pyttsx3.init()
#                 engine.setProperty('rate', 150)
#                 engine.say(text)
#                 engine.runAndWait()
#                 engine.stop()
#                 del engine
#             except Exception as e:
#                 print(f"TTS Error: {e}")