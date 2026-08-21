import sounddevice as sd
from kokoro_onnx import Kokoro

print("⏳ Loading Kokoro Neural Model (v1.0)...")
# تحميل محرك النطق والأصوات بالإصدار الجديد
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# النص الاختباري 
text = "Hello sir! I am Athena. I have successfully upgraded my vocal cords. I can now speak with a natural human voice, pause for breath, and even change my tone. What do you think?"

print(f"💬 Generating speech for: '{text}'")

# توليد الصوت 
samples, sample_rate = kokoro.create(text, voice="af_sarah", speed=1.0, lang="en-us")

print("🔊 Playing audio...")
sd.play(samples, sample_rate)
sd.wait()

print("✅ Done!")
