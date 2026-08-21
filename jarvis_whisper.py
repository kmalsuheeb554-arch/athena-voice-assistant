import os
import subprocess
from faster_whisper import WhisperModel

def record_audio(filename="temp_audio.wav", duration=5):
    """تسجيل الصوت عبر نظام لينكس مباشرة"""
    print(f"\n🎤 جارفيس يستمع الآن (لديك {duration} ثوانٍ للتحدث)...")
    cmd = ["arecord", "-q", "-d", str(duration), "-f", "S16_LE", "-c", "1", "-r", "16000", filename]
    subprocess.run(cmd)
    print("⏳ جاري التحليل بذكاء Whisper...")

def main():
    print("⚙️ جاري تحميل عقل Whisper STT (قد يستغرق بضع ثوانٍ في المرة الأولى)...")
    
    # نستخدم نموذج small مع تسريع int8 ليعمل بخفة على المعالج (CPU)
    model = WhisperModel("small", device="cpu", compute_type="int8")
    print("✅ النظام جاهز.")
    
    while True:
        try:
            # 1. التقاط الصوت
            record_audio()
            
            # 2. تحويل الصوت إلى نص بدقة عالية
            # نحدد اللغة ar لتسريع العملية ومنع الترجمة الخاطئة
            segments, info = model.transcribe("temp_audio.wav", language="ar")
            
            # تجميع الكلمات من المقاطع
            text = "".join([segment.text for segment in segments])
            
            # 3. طباعة النتيجة
            if text.strip():
                print(f"🎯 أنت قلت: {text.strip()}")
            else:
                print("لم يتم التقاط صوت واضح.")
                
        except KeyboardInterrupt:
            print("\n👋 تم إيقاف النظام.")
            break
        except Exception as e:
            print(f"❌ حدث خطأ: {e}")
            break

if __name__ == "__main__":
    main()
