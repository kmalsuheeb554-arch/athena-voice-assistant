import soundfile as sf
import numpy as np

print("✂️ جاري معالجة الملف الصوتي...")

# 1. قراءة الملف الصوتي
file_path = "mhm.wav"
data, sample_rate = sf.read(file_path)

# 2. تحديد المدة المراد حذفها (0.1 ثانية = عُشر ثانية)
# يمكنك زيادة الرقم إلى 0.2 أو 0.3 إذا كان النشاز أطول
trim_seconds = 3.0 
frames_to_trim = int(trim_seconds * sample_rate)

# 3. قص الجزء الأخير من المصفوفة
trimmed_data = data[:-frames_to_trim]

# 4. إضافة تلاشي ناعم (Fade-out) في آخر 0.05 ثانية لمنع الطقطقة عند التوقف
fade_duration = 0.05
fade_frames = int(fade_duration * sample_rate)

# إنشاء منحنى التلاشي (من 100% صوت إلى 0%)
fade_curve = np.linspace(1.0, 0.0, fade_frames)

# تطبيق التلاشي بذكاء (سواء كان الملف مونو أو ستيريو)
if len(trimmed_data.shape) > 1:
    for i in range(trimmed_data.shape[1]):
        trimmed_data[-fade_frames:, i] *= fade_curve
else:
    trimmed_data[-fade_frames:] *= fade_curve

# 5. حفظ الملف النقي فوق الملف القديم
sf.write("mhm.wav", trimmed_data, sample_rate)
print("✅ تم قص النشاز وإضافة التلاشي الناعم بنجاح!")
