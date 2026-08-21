import pyaudio
import webrtcvad
import collections
import wave
import struct
import math

# ==========================================
# إعدادات الصوت
# ==========================================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)

# حد مستوى الصوت (الضجيج). سيتم تجاهل أي صوت أقل من هذا الرقم.
# قد تحتاج لتعديل هذا الرقم بناءً على ما يظهر لك في التيرمنال.
MIN_VOLUME_THRESHOLD = 3700 

def get_rms(audio_chunk):
    """دالة لحساب مستوى طاقة الصوت (الشدة)"""
    count = len(audio_chunk) // 2
    formats = "%dh" % (count)
    shorts = struct.unpack(formats, audio_chunk)
    sum_squares = sum(s**2 for s in shorts)
    return math.sqrt(sum_squares / count) if count > 0 else 0

def main():
    vad = webrtcvad.Vad(3) 
    audio = pyaudio.PyAudio()
    
    # إخفاء أخطاء ALSA المزعجة (اختياري، يعمل على لينكس)
    import ctypes
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    def py_error_handler(filename, line, function, err, fmt): pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    try:
        asound = ctypes.cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)
    except:
        pass

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)
    
    print("="*50)
    print("🎤 النظام يستمع... (تم تفعيل فلتر المراوح)")
    print("="*50)
    
    frames = []
    ring_buffer = collections.deque(maxlen=20) 
    triggered = False
    silence_counter = 0
    MAX_SILENCE_CHUNKS = 20 
    
    try:
        while True:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            rms_volume = get_rms(chunk)
            
            # فلتر مزدوج: يجب أن يتخطى الصوت حاجز الضجيج + يجب أن يكون صوتاً بشرياً
            is_speech = (rms_volume > MIN_VOLUME_THRESHOLD) and vad.is_speech(chunk, RATE)
            
            if not triggered:
                # طباعة مستوى الصوت الحالي لتسهيل المعايرة (عندما لا يكون هناك تسجيل)
                print(f"\rمستوى الصوت المحيط: {rms_volume:.0f} (الحد: {MIN_VOLUME_THRESHOLD})", end="")
                
                ring_buffer.append(chunk)
                if is_speech:
                    print("\n🗣️ تم رصد كلام واضح! جاري التسجيل...")
                    triggered = True
                    frames.extend(ring_buffer)
                    ring_buffer.clear()
            else:
                frames.append(chunk)
                if not is_speech:
                    silence_counter += 1
                    if silence_counter > MAX_SILENCE_CHUNKS:
                        print("🔇 تم رصد صمت! إيقاف التسجيل...")
                        break
                else:
                    silence_counter = 0 
                    
    except KeyboardInterrupt:
        print("\n👋 تم الإيقاف يدوياً.")
        
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        if frames:
            with wave.open("test_vad.wav", 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            print("✅ تم الحفظ. جرب تشغيله عبر الأمر: aplay test_vad.wav")

if __name__ == '__main__':
    main()
