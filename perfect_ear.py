import speech_recognition as sr
import json
import sys
from vosk import Model, KaldiRecognizer

print("\n⏳ جاري تحميل نموذج الذكاء الاصطناعي (Vosk)...")
try:
    model = Model("vosk-model-ar-mgb2-0.4")
except Exception as e:
    print("❌ خطأ: تأكد من وجود مجلد النموذج في نفس المسار.")
    sys.exit(1)

# إعداد محرك Vosk
rec = KaldiRecognizer(model, 16000)

# إعداد مدير المايكروفون الذكي
r = sr.Recognizer()
r.energy_threshold = 300 # حساسية الصوت الافتراضية
r.dynamic_energy_threshold = True # التكيف التلقائي مع الضوضاء

print("✅ تم تحميل النموذج بنجاح.")

with sr.Microphone(sample_rate=16000) as source:
    print("\n🔊 جاري معايرة الضوضاء المحيطة... (يرجى الصمت لثانيتين)")
    # هذه الدالة السحرية تصفي صوت مراوح اللابتوب والضجيج الخلفي
    r.adjust_for_ambient_noise(source, duration=2)
    
    print("\n🎙️ النظام جاهز! تحدث الآن باللغة العربية الفصحى...")
    
    try:
        # الاستماع حتى تتوقف عن الحديث
        audio = r.listen(source, timeout=10, phrase_time_limit=15)
        print("\n⚙️ جاري معالجة الصوت...")
        
        # استخراج البيانات الصوتية الخام النظيفة جداً
        data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        
        # تمرير الصوت النظيف إلى Vosk
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            print(f"🎯 النتيجة: {text}")
        else:
            result = json.loads(rec.FinalResult())
            text = result.get("text", "")
            print(f"🎯 النتيجة النهائية: {text}")

    except sr.WaitTimeoutError:
        print("⚠️ لم يتم التقاط أي صوت.")
    except Exception as e:
        print(f"❌ حدث خطأ غير متوقع: {e}")
