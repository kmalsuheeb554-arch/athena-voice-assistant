import os
import sys
import json
import pyaudio
import pygame
import asyncio
import edge_tts
import numpy as np
import ollama
from vosk import Model, KaldiRecognizer

# إعدادات المسارات
VOSK_MODEL_DIR = "vosk-model-small-ar-0.3"

# تهيئة البيئة
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
pygame.mixer.init()

if not os.path.exists(VOSK_MODEL_DIR):
    print(f"خطأ: مجلد {VOSK_MODEL_DIR} غير موجود!")
    sys.exit(1)

print("جاري تهيئة المحركات...")
model = Model(VOSK_MODEL_DIR)
rec = KaldiRecognizer(model, 16000)
p = pyaudio.PyAudio()

async def generate_neural_speech(text, output_file):
    # صوت "حامد" السعودي الاحترافي
    communicate = edge_tts.Communicate(text, "ar-SA-HamedNeural")
    await communicate.save(output_file)

def speak(text):
    print(f"\nجارفيس: {text}")
    audio_file = "response.mp3"
    asyncio.run(generate_neural_speech(text, audio_file))
    
    if os.path.exists(audio_file):
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
        try: os.remove(audio_file)
        except: pass

print("\n=== النظام الاحترافي قيد التشغيل (تم تفعيل فلتر الضوضاء) ===")

try:
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=4000)
    stream.start_stream()
    
    print("=== جارفيس يستمع الآن... ===")
    
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        
        # [فلتر تصفية الضوضاء]
        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        # أي صوت تحت مستوى 500 يعتبر تشويشاً ويتم تحويله لصمت
        audio_data[np.abs(audio_data) < 100] = 0 
        clean_data = audio_data.astype(np.int16).tobytes()

        if rec.AcceptWaveform(clean_data):
            res = json.loads(rec.Result())
            text = res.get("text", "").strip()
            
            if text:
                print(f"\nأنت: {text}")
                stream.stop_stream()
                
                # المعالجة
                response = ollama.generate(model='qwen2.5:1.5b', prompt=text)
                speak(response['response'])
                
                print("\n=== جارفيس يستمع مجدداً... ===")
                rec.Reset()
                stream.start_stream()
        else:
            # تحديث مرئي فقط إذا كان هناك كلام حقيقي
            partial = json.loads(rec.PartialResult()).get("partial", "")
            if len(partial) > 2:
                print(f"\rيسمعك الآن: {partial}", end="", flush=True)

except KeyboardInterrupt:
    print("\nتم الإغلاق.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()
