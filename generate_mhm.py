import numpy as np
import soundfile as sf

print("🎵 جاري توليد نغمة التنبيه الاحترافية...")

# إعدادات الصوت
sample_rate = 44100

def generate_beep(freq, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # توليد الموجة الصوتية
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    # إضافة تأثير التلاشي (Fade out) ليكون الصوت ناعماً واحترافياً
    envelope = np.exp(-5 * t / duration)
    return wave * envelope

# النغمة الأولى (منخفضة)
beep1 = generate_beep(415.30, 0.15) 
# فاصل زمني صامت
silence = np.zeros(int(sample_rate * 0.04))
# النغمة الثانية (أعلى، لتعطي إيحاء التساؤل والاستماع)
beep2 = generate_beep(554.37, 0.25)  

# دمج الأصوات
final_chime = np.concatenate([beep1, silence, beep2])

# حفظ الملف في نفس المجلد
sf.write('mhm.wav', final_chime, sample_rate)
print("✅ تم إنشاء ملف mhm.wav بنجاح في مجلد المشروع!")
