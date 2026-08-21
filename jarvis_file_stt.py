import os
import wave
import json
import subprocess
from vosk import Model, KaldiRecognizer

def record_audio(filename="temp_audio.wav", duration=4):
    """
    تسجيل الصوت باستخدام أداة arecord المضمونة في لينكس
    """
    print(f"\n🎤 جارفيس يستمع الآن (لديك {duration} ثوانٍ للتحدث)...")
    # أمر التسجيل الصامت -q
    cmd = ["arecord", "-q", "-d", str(duration), "-f", "S16_LE", "-c", "1", "-r", "16000", filename]
    subprocess.run(cmd)
    print("⏳ جاري تحليل الصوت...")

def transcribe_audio(filename="temp_audio.wav", model_path="vosk-model-ar-mgb2-0.4"):
    """
    قراءة الملف الصوتي وتحويله إلى نص باستخدام Vosk
    """
    if not os.path.exists(filename):
        return ""

    model = Model(model_path)
    wf = wave.open(filename, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)
        
    result = json.loads(rec.FinalResult())
    
    # تنظيف الملف المؤقت بعد الانتهاء
    wf.close()
    os.remove(filename)
    
    return result.get("text", "")

def main():
    print("⚙️ جاري إعداد جارفيس...")
    
    while True:
        try:
            # 1. تسجيل الصوت
            record_audio()
            
            # 2. تحويل الصوت إلى نص
            text = transcribe_audio()
            
            # 3. طباعة النتيجة
            if text.strip():
                print(f"🎯 أنت قلت: {text}")
            else:
                print("لم أسمع شيئاً واضحاً.")
                
        except KeyboardInterrupt:
            print("\n👋 تم إيقاف جارفيس.")
            break
        except Exception as e:
            print(f"❌ حدث خطأ: {e}")
            break

if __name__ == "__main__":
    main()
