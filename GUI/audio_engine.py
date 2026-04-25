import platform
import subprocess
import threading
import queue

class AudioEngine:
    def __init__(self):
        self._queue = queue.Queue()
        self._current_process = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text):
        """Queues text to be spoken. Interrupts any current speech immediately."""
        if not text:
            return

        # Clear queue and stop current speech to ensure we handle "changing quickly"
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        
        self.stop()
        self._queue.put(text)

    def stop(self):
        """Kills the current speech process immediately."""
        if self._current_process and self._current_process.poll() is None:
            try:
                self._current_process.terminate()
                self._current_process.wait(timeout=0.5)
            except Exception:
                try:
                    self._current_process.kill()
                except:
                    pass
            self._current_process = None

    def _worker(self):
        while True:
            text = self._queue.get()
            if text is None: break
            
            try:
                system = platform.system()
                if system == "Darwin":
                    # Mac
                    cmd = ["say", text]
                elif system == "Windows":
                    # Windows PowerShell (no external library needed, handles interruption via process kill)
                    clean_text = text.replace("'", "''")
                    ps_cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{clean_text}')"
                    cmd = ["powershell", "-Command", ps_cmd]
                else:
                    # Linux fallback
                    cmd = ["spd-say", text]

                self._current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._current_process.wait()
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                self._current_process = None
                self._queue.task_done()