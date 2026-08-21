import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model

# 1. إعدادات المايكروفون الأساسية
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("⏳ جاري تحميل محرك الذكاء الاصطناعي الاحترافي الخاص بك...")

# 🌟 التصحيح الجذري هنا: استخدام wakeword_model_paths بدلاً من wakeword_models
owwModel = Model(wakeword_model_paths=["Hey_Athena_20260622_035553.onnx"]) 

print("🟢 النظام مستعد ويستمع بصمت تام! قل كلمتك السحرية (dell)...")

# 3. حلقة الاستماع (الرادار)
try:
    while True:
        # التقاط الصوت من المايكروفون
        audio_data = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        
        # تحليل الصوت عبر ملفك
        prediction = owwModel.predict(audio_data)
        
        # فحص النتيجة (تتجاهل الضجيج وتلتقط الكلمة)
        for model_name, score in prediction.items():
            if score > 0.5: # 0.5 تعني نسبة ثقة 50% فما فوق
                print(f"🚀 أهلاً يا مهندس! لقد التقطت الكلمة بنجاح (نسبة الثقة: {score:.2f})")
                
except KeyboardInterrupt:
    print("\n🛑 تم إيقاف الرادار.")
finally:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()
