import sys
import queue
import json
import sounddevice as sd
from vosk import Model, KaldiRecognizer

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        pass # تجاهل رسائل التحذير البسيطة
    q.put(bytes(indata))

print("\n⏳ جاري تحميل نموذج الذكاء الاصطناعي (Vosk)...")
try:
    model = Model("vosk-model-ar-mgb2-0.4")
except Exception as e:
    print("❌ خطأ: تأكد من وجود مجلد النموذج.")
    sys.exit(1)

try:
    # 1. إجبار النظام على استخدام تردد 16000 المتوافق حصرياً مع نموذج Vosk
    samplerate = 16000
    
    rec = KaldiRecognizer(model, samplerate)
    
    print(f"\n✅ تم ضبط المايكروفون إجبارياً على تردد: {samplerate} Hz")
    print("🎙️ تحدث الآن باللغة العربية الفصحى... (اضغط Ctrl+C للإيقاف)\n")

    # 2. فتح المايكروفون بالتردد الجديد
    with sd.RawInputStream(samplerate=samplerate, blocksize=5000, device=None,
                           dtype='int16', channels=1, callback=callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print(f"\n🎯 النتيجة: {text}\n")
            else:
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    # تفريغ السطر وطباعة النص الجديد لتجنب تداخل الحروف
                    print(f"\r\033[K👂 أستمع: {partial_text}", end='', flush=True)

except KeyboardInterrupt:
    print("\n👋 تم الإيقاف.")
except Exception as e:
    print(f"\n❌ حدث خطأ: {e}")
