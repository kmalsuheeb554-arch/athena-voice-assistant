import os
import subprocess
import re
from faster_whisper import WhisperModel

# ==========================================
# 1. Audio Recording (arecord - Ubuntu)
# ==========================================
# تم إضافة خيار quiet لكي يستمع بصمت دون إزعاج في التيرمنال
def record_audio(filename="temp_audio.wav", duration=5, quiet=False):
    if not quiet:
        print(f"\n🎤 Listening for command... ({duration}s)")
    cmd = ["arecord", "-q", "-d", str(duration), "-f", "S16_LE", "-c", "1", "-r", "16000", filename]
    subprocess.run(cmd)
    if not quiet:
        print("⏳ Analyzing...")

# ==========================================
# 2. Brain (Qwen + Prompt Engineering)
# ==========================================
def ask_qwen(prompt):
    print("🧠 Thinking...")
    system_prompt = (
        "You are Jarvis, an advanced AI on Ubuntu. Reply concisely in English. "
        "You control the system using these specific tags. Use them whenever requested:\n"
        "- To resume or play paused music: <CMD: playerctl play>\n"
        "- To pause music: <CMD: playerctl pause>\n"
        "- To skip to next song: <CMD: playerctl next>\n"
        "- To play new music from internet/YouTube: <PLAY: song name>\n"
        "- To stop internet music: <CMD: killall mpv>\n"
        "- To download a song: <DOWNLOAD: song name>\n"
        "- To increase volume: <CMD: amixer -D pulse sset Master 10%+>\n"
        "- To decrease volume: <CMD: amixer -D pulse sset Master 10%->\n"
        "- To open apps: <CMD: application-name> (e.g., google-chrome, gnome-calculator)\n"
        "If using a tag, say a short confirmation like 'Right away, sir' along with the tag."
    )
    
    full_prompt = f"{system_prompt}\nUser: {prompt}"
    try:
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:1.5b", full_prompt],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        return "Error: Brain not responding."

# ==========================================
# 3. Speech & Text Cleaning
# ==========================================
def speak(text):
    clean_text = re.sub(r'<[^>]+>', '', text).strip()
    if not clean_text:
        return
        
    print(f"🔊 Jarvis: {clean_text}")
    try:
        tts_cmd = f'edge-tts --voice "en-US-GuyNeural" --text "{clean_text}" --write-media reply.mp3 && mpv --no-video reply.mp3 > /dev/null 2>&1'
        subprocess.run(tts_cmd, shell=True)
    except:
        pass

# ==========================================
# 4. Smart Command Execution Engine
# ==========================================
def execute_commands(reply):
    cmd_match = re.search(r'<CMD:(.*?)>', reply)
    if cmd_match:
        cmd = cmd_match.group(1).strip()
        print(f"⚙️ Executing System Command: {cmd}")
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    play_match = re.search(r'<PLAY:(.*?)>', reply)
    if play_match:
        query = play_match.group(1).strip()
        print(f"🎵 Streaming from Internet: {query}")
        cmd = ["mpv", "--no-video", f"--force-media-title={query}", f"ytdl://ytsearch1:{query}"]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    dl_match = re.search(r'<DOWNLOAD:(.*?)>', reply)
    if dl_match:
        query = dl_match.group(1).strip()
        print(f"⬇️ Downloading: {query}")
        desktop_music_dir = os.path.expanduser("~/Desktop/music")
        os.makedirs(desktop_music_dir, exist_ok=True)
        dl_cmd = f'yt-dlp -x --audio-format mp3 -o "{desktop_music_dir}/%(title)s.%(ext)s" "ytsearch1:{query}"'
        subprocess.Popen(dl_cmd, shell=True)

# ==========================================
# 5. Main Core (Wake Word Logic)
# ==========================================
def main():
    os.system("clear")
    print("="*60)
    print("🤖 Jarvis - Wake Word Edition")
    print("="*60)
    
    print("⚙️ Loading Whisper Base.en model...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    speak("System online. Just say my name, Jarvis, when you need me.")

    while True:
        try:
            # 1. استماع صامت في الخلفية لالتقاط كلمة "جارفيس"
            record_audio("temp_audio.wav", duration=4, quiet=True)
            segments, _ = model.transcribe("temp_audio.wav", language="en")
            text = "".join([s.text for s in segments]).strip().lower()
            
            # 2. فحص الكلمات: هل ناداني أحد؟
            if "jarvis" in text:
                print(f"\n🗣️ Heard: {text}")
                
                # استخراج الأمر الذي يأتي بعد كلمة جارفيس
                parts = text.split("jarvis", 1)
                command = parts[1].strip()
                
                # إذا قال "جارفيس" وسكت (لا يوجد أمر بعدها)، نرد وننتظر الأمر
                if len(command) < 3:
                    speak("Yes, sir?")
                    # نستمع الآن بشكل علني للأمر
                    record_audio("temp_audio.wav", duration=5, quiet=False)
                    segments, _ = model.transcribe("temp_audio.wav", language="en")
                    command = "".join([s.text for s in segments]).strip().lower()
                
                if not command:
                    continue
                    
                print(f"👤 You: {command}")
                
                if any(word in command for word in ["exit", "stop", "close", "quit"]):
                    speak("Shutting down. Goodbye, sir.")
                    break
                    
                # 3. إرسال الأمر للعقل والتنفيذ
                ai_response = ask_qwen(command)
                speak(ai_response)
                execute_commands(ai_response)
                
        except KeyboardInterrupt:
            print("\n👋 System stopped manually.")
            break
        except Exception as e:
            print(f"❌ Critical Error: {e}")
            break

if __name__ == "__main__":
    main()
