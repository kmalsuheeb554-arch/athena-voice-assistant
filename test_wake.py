import openwakeword
from openwakeword.model import Model
import pyaudio
import numpy as np

# 1. إعدادات المايكروفون (التردد القياسي للذكاء الاصطناعي)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# 2. تحميل محرك جارفيس المدمج
print("⏳ جاري تحميل العقل المدبر لجارفيس...")

# 🌟 التعديل الجذري: ترك الأقواس فارغة ليقوم بتحميل النماذج الافتراضية تلقائياً
owwModel = Model() 

print("🟢 جارفيس مستعد ويستمع الآن! قل 'Hey Jarvis' بصوت واضح...")

# 3. حلقة الاستماع الدائمة
try:
    while True:
        # التقاط حزمة صوتية من المايكروفون
        audio_data = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        
        # تمرير الصوت للمحرك لتحليله
        prediction = owwModel.predict(audio_data)
        
        # 🌟 التعديل الجذري: فحص جميع النماذج التي تم تقييمها والتقاط جارفيس منها
        for model_name, score in prediction.items():
            if "jarvis" in model_name.lower() and score > 0.5:
                print(f"🚀 أهلاً يا مهندس! أنا في خدمتك... (نسبة الثقة: {score:.2f})")
            
except KeyboardInterrupt:
    print("\n🛑 تم إيقاف جارفيس.")
finally:
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()
