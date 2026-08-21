import subprocess
import json
import sys
from vosk import Model, KaldiRecognizer

def main():
    print("⏳ جاري تحميل نموذج الذكاء الاصطناعي...")
    try:
        model = Model("vosk-model-ar-mgb2-0.4")
        # نثبت التردد على 16000 لأنه التردد الذهبي لنماذج Vosk
        rec = KaldiRecognizer(model, 16000)
    except Exception as e:
        print(f"❌ خطأ في تحميل النموذج: {e}")
        sys.exit(1)

    # استخدام arecord الخاصة بلينكس لالتقاط الصوت الصافي
    # -q: صامت، -r: التردد 16000، -c: قناة واحدة، -f: صيغة 16-bit، -t: خام
    cmd = ["arecord", "-q", "-r", "16000", "-c", "1", "-f", "S16_LE", "-t", "raw"]
    
    print("\n✅ جارفيس يستمع الآن عبر نواة لينكس مباشرة... (تحدث الآن)")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        
        while True:
            # قراءة حزم الصوت المتتالية
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
                
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text.strip():
                    print(f"🎯 تم التقاط: {text}")
            else:
                # يمكنك تفعيل السطر التالي إذا أردت رؤية الكلمات وهي تتكون حرفياً
                # partial = json.loads(rec.PartialResult())
                # if partial.get("partial", ""): print(partial["partial"])
                pass

    except KeyboardInterrupt:
        print("\n👋 تم إيقاف الفحص.")
        process.terminate()
    except Exception as e:
        print(f"❌ حدث خطأ: {e}")

if __name__ == "__main__":
    main()
