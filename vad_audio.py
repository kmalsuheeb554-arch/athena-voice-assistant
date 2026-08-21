import pyaudio
import wave
import numpy as np

class VADListener:
    def __init__(self, aggressiveness=3, threshold=4000): 
        # المتغيرات موجودة لضمان التوافق مع استدعاء run_ai.py بدون أخطاء
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.audio = pyaudio.PyAudio()

    def wait_for_speech(self, save_path=None):
        print("\n🔴 [Command Window] جاري تسجيل الأمر الصافي لمدة 3.5 ثانية...")
        stream = self.audio.open(format=self.FORMAT,
                                 channels=self.CHANNELS,
                                 rate=self.RATE,
                                 input=True,
                                 frames_per_buffer=self.CHUNK)

        frames = []
        # 🌟 النافذة الصارمة: 3.5 ثانية تكفي تماماً لأي أمر وتمنع تلوث البصمة
        RECORD_SECONDS = 3.5  
        
        # تفريغ المايكروفون من أي تشويش متراكم في اللحظات السابقة
        for _ in range(3):
            stream.read(self.CHUNK, exception_on_overflow=False)

        # تسجيل 3.5 ثانية بالضبط بدون انتظار الصمت وبدون الاعتماد على VAD
        for i in range(0, int(self.RATE / self.CHUNK * RECORD_SECONDS)):
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        print("✅ [Command Window] انتهى التسجيل. جاري التحليل اللحظي!")

        raw_data = b''.join(frames)
        
        if save_path:
            with wave.open(save_path, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(raw_data)
            return save_path
            
        audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np

    def cleanup(self):
        try:
            self.audio.terminate()
        except:
            pass
