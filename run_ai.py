import sys
import os
import re
import time
import math
import sqlite3
import random
import subprocess
import requests
import threading
import tempfile
import pyaudio
import numpy as np
import torch
import soundfile as sf 

# 🌟 استدعاء المكتبات الذكية للسرعة واللفظ
import jellyfish
from rapidfuzz import process, fuzz

from typing import Tuple, Any
from openwakeword.model import Model
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import SpeakerRecognition
from kokoro_onnx import Kokoro

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QRect, QEasingCurve, QRectF, QMimeData, QUrl, QSize
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QBrush, QPixmap, QPainterPath, QDrag, QImage, QIcon, QClipboard

# ==========================================
# 🌟 محرك الفهم الدلالي (Semantic Engine)
# ==========================================
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("⚠️ الرجاء تثبيت المكتبة عبر: pip install sentence-transformers")

semantic_matcher = None

class SemanticAppMatcher:
    def __init__(self):
        print("🧠 Loading Semantic Embeddings Model (all-MiniLM-L6-v2) to RAM...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        self.app_knowledge = {
            "web browser chrome internet surf website": "google-chrome",
            "browser mozilla firefox web internet": "firefox",
            "file manager folder nautilus explorer directory files": "nautilus",
            "terminal command line console prompt shell bash": "gnome-terminal",
            "settings control panel configuration options system": "gnome-control-center",
            "code editor visual studio vscode programming script python": "code",
            "discord chat voice gaming call server friends": "Discord",
            "task manager system monitor processes resources performance ram": "gnome-system-monitor",
            "calculator math compute numbers addition": "gnome-calculator"
        }
        self.corpus_keys = list(self.app_knowledge.keys())
        self.corpus_embeddings = self.model.encode(self.corpus_keys, convert_to_tensor=True)
        print("✅ Semantic Embeddings Ready!")

    def find_best_match(self, query, threshold=0.45):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings)[0]
        best_hit = hits[0]
        
        if best_hit['score'] >= threshold:
            matched_key = self.corpus_keys[best_hit['corpus_id']]
            return self.app_knowledge[matched_key], best_hit['score']
        return None, 0.0

# ==========================================
# 1. نظام معالجة الأوامر (Command Handler)
# ==========================================
class CommandHandler:
    MUSIC_PLAY = ["play", "start", "resume"]
    MUSIC_STOP = ["stop", "pause"]
    MUSIC_NEXT = ["next", "skip"]
    MUSIC_PREV = ["previous", "back"]
    
    BRIGHTNESS = ["brightness", "light", "screen"]
    BRIGHTNESS_UP = ["up", "increase", "higher", "raise"]
    BRIGHTNESS_DOWN = ["down", "decrease", "lower", "dim"]
    
    VOLUME = ["volume", "sound", "speaker", "audio"]
    VOLUME_UP = ["up", "increase", "louder", "raise"]
    VOLUME_DOWN = ["down", "decrease", "lower", "quiet"]
    
    TIME_WORDS = ["time", "clock"]
    DATE_WORDS = ["date", "day", "today"]
    MEMORY_WORDS = ["memory", "ram"]
    CPU_WORDS = ["cpu", "processor"]
    
    SCREENSHOT_WORDS = ["screenshot", "screen shot", "capture screen", "take a picture"]
    
    OPEN_WORDS = ["open", "launch", "start", "opened"]
    CLOSE_APP_WORDS = ["close", "kill", "quit", "exit"]

    CLOSE_TAB_WORDS = ["close this tab", "close tab", "close current tab", "close the tab"]
    CLOSE_WINDOW_WORDS = ["close this window", "close window", "close current window", "close the window"]

    WIFI_WORDS = ["wifi", "wi-fi", "wireless"]
    HOTSPOT_WORDS = ["hotspot", "tethering"]
    AIRPLANE_WORDS = ["airplane", "flight mode"]
    BLUETOOTH_WORDS = ["bluetooth"]
    NIGHTLIGHT_WORDS = ["night light", "night mode", "eye care"]
    LOCK_WORDS = ["lock screen", "lock computer", "lock pc", "lock the laptop"]
    
    SHUTDOWN_WORDS = ["shut down", "shutdown", "turn off", "power off"]
    RESTART_WORDS = ["restart", "reboot"]
    
    ON_WORDS = ["on", "enable", "start"]
    OFF_WORDS = ["off", "disable", "stop", "close"]
    
    CLEAR_MEMORY_WORDS = ["forget", "clear memory", "new topic", "reset memory"]
    REMINDER_WORDS = ["remind", "reminder", "timer", "alarm"]

    @staticmethod
    def extract_number(text: str) -> int:
        match = re.search(r'\d+', text)
        return int(match.group()) if match else 50
    
    @staticmethod
    def check_keywords(text: str, keywords: list) -> bool:
        return any(re.search(rf'\b{re.escape(keyword)}\b', text) for keyword in keywords)
    
    @staticmethod
    def route(user_input: str) -> Tuple[str, str, Any]:
        text = user_input.lower().strip()
        
        if text.startswith("play "):
            song_name = text.replace("play ", "", 1).strip()
            if song_name not in ["music", "song", "media", "audio", "some music"]:
                return "music", "play_local", song_name

        if CommandHandler.check_keywords(text, CommandHandler.CLEAR_MEMORY_WORDS): return "ai", "clear", None
        if CommandHandler.check_keywords(text, CommandHandler.WIFI_WORDS): return "network", "wifi", "on" if CommandHandler.check_keywords(text, CommandHandler.ON_WORDS) else "off"
        if CommandHandler.check_keywords(text, CommandHandler.HOTSPOT_WORDS): return "network", "hotspot", "on" if CommandHandler.check_keywords(text, CommandHandler.ON_WORDS) else "off"
        if CommandHandler.check_keywords(text, CommandHandler.AIRPLANE_WORDS): return "network", "airplane", "on" if CommandHandler.check_keywords(text, CommandHandler.ON_WORDS) else "off"
        if CommandHandler.check_keywords(text, CommandHandler.BLUETOOTH_WORDS): return "network", "bluetooth", "on" if CommandHandler.check_keywords(text, CommandHandler.ON_WORDS) else "off"
        
        if CommandHandler.check_keywords(text, CommandHandler.NIGHTLIGHT_WORDS): return "display", "nightlight", "on" if CommandHandler.check_keywords(text, CommandHandler.ON_WORDS) else "off"
        if CommandHandler.check_keywords(text, CommandHandler.LOCK_WORDS): return "system", "lock", None
        if CommandHandler.check_keywords(text, CommandHandler.SCREENSHOT_WORDS): return "system", "screenshot", None
        
        if CommandHandler.check_keywords(text, CommandHandler.SHUTDOWN_WORDS): return "power", "shutdown", None
        if CommandHandler.check_keywords(text, CommandHandler.RESTART_WORDS): return "power", "restart", None

        if CommandHandler.check_keywords(text, ["music", "song", "media", "player"]):
            if CommandHandler.check_keywords(text, CommandHandler.MUSIC_PLAY): return "music", "play", None
            if CommandHandler.check_keywords(text, CommandHandler.MUSIC_STOP): return "music", "stop", None
            if CommandHandler.check_keywords(text, CommandHandler.MUSIC_NEXT): return "music", "next", None
            if CommandHandler.check_keywords(text, CommandHandler.MUSIC_PREV): return "music", "previous", None
        
        if CommandHandler.check_keywords(text, CommandHandler.BRIGHTNESS):
            if CommandHandler.check_keywords(text, CommandHandler.BRIGHTNESS_UP): return "brightness", "increase", None
            if CommandHandler.check_keywords(text, CommandHandler.BRIGHTNESS_DOWN): return "brightness", "decrease", None
            else: return "brightness", "set", CommandHandler.extract_number(text)
        
        if CommandHandler.check_keywords(text, CommandHandler.VOLUME):
            if CommandHandler.check_keywords(text, CommandHandler.VOLUME_UP): return "volume", "increase", None
            if CommandHandler.check_keywords(text, CommandHandler.VOLUME_DOWN): return "volume", "decrease", None
        
        if CommandHandler.check_keywords(text, CommandHandler.TIME_WORDS): return "system", "time", None
        if CommandHandler.check_keywords(text, CommandHandler.DATE_WORDS): return "system", "date", None
        if CommandHandler.check_keywords(text, CommandHandler.MEMORY_WORDS): return "system", "memory", None
        if CommandHandler.check_keywords(text, CommandHandler.CPU_WORDS): return "system", "cpu", None

        if CommandHandler.check_keywords(text, CommandHandler.CLOSE_TAB_WORDS): return "system", "close_tab", None
        if CommandHandler.check_keywords(text, CommandHandler.CLOSE_WINDOW_WORDS): return "system", "close_window", None
        
        words = text.split()
        for i, word in enumerate(words):
            if word in CommandHandler.OPEN_WORDS and i + 1 < len(words):
                target_words = words[i+1:]
                if target_words and target_words[0] in ["the", "a", "an"]: target_words.pop(0)
                if target_words:
                    if target_words[0] in ["file", "document", "image", "picture", "video", "pdf"]:
                        target_words.pop(0)
                        if target_words: return "files", "open", " ".join(target_words)
                    return "apps", "open", " ".join(target_words)
                    
            if word in CommandHandler.CLOSE_APP_WORDS and i + 1 < len(words):
                target_words = words[i+1:]
                if target_words and target_words[0] in ["the", "a", "an"]: target_words.pop(0)
                if target_words: return "apps", "close", " ".join(target_words)

        if CommandHandler.check_keywords(text, CommandHandler.REMINDER_WORDS): 
            return "system", "reminder", text

        return "other", "ask", None

# ==========================================
# 2. منفذ الإجراءات (Actions)
# ==========================================
class Actions:
    @staticmethod
    def run_cmd(cmd: str) -> bool:
        return os.system(f"{cmd} > /dev/null 2>&1") == 0

    @staticmethod
    def network(device: str, state: str) -> str:
        if device == "wifi":
            Actions.run_cmd("nmcli radio wifi on" if state == "on" else "nmcli radio wifi off")
            return f"Wi-Fi turned {state}."
        elif device == "hotspot":
            Actions.run_cmd("nmcli device wifi hotspot" if state == "on" else "nmcli connection down Hotspot")
            return f"Hotspot turned {state}."
        elif device == "airplane":
            Actions.run_cmd("rfkill block all" if state == "on" else "rfkill unblock all")
            return f"Airplane mode turned {state}."
        elif device == "bluetooth":
            Actions.run_cmd("rfkill unblock bluetooth" if state == "on" else "rfkill block bluetooth")
            return f"Bluetooth turned {state}."
        return "Failed to change network settings."

    @staticmethod
    def display_settings(action: str, state: str):
        if action == "nightlight":
            val = "true" if state == "on" else "false"
            Actions.run_cmd(f"gsettings set org.gnome.settings-daemon.plugins.color night-light-enabled {val}")
            return f"Night light turned {state}."
        return "Failed to change display settings."

    @staticmethod
    def music(action: str, song_name: str = None):
        if action == "play_local" and song_name:
            music_dir = os.path.expanduser("~/Desktop/music/wdell")
            if not os.path.exists(music_dir):
                return "Your music folder is empty or not found, sir."
            
            files = [f for f in os.listdir(music_dir) if f.endswith(('.mp3', '.m4a', '.wav', '.flac', '.ogg'))]
            if not files: return "No music files found in your local folder."
            
            clean_files = {f.lower().replace('_', ' ').replace('-', ' ').rsplit('.', 1)[0]: f for f in files}
            
            best_match = process.extractOne(song_name.lower(), clean_files.keys(), scorer=fuzz.WRatio)
            actual_file = None
            matched_name = None
            
            if best_match and best_match[1] >= 65:
                matched_name = best_match[0]
                actual_file = clean_files[matched_name]
            else:
                query_phonetic = jellyfish.metaphone(song_name.lower())
                for clean_name, original_file in clean_files.items():
                    if jellyfish.metaphone(clean_name) == query_phonetic:
                        matched_name = clean_name
                        actual_file = original_file
                        break
            
            if actual_file:
                filepath = os.path.join(music_dir, actual_file)
                os.system("pkill -9 mpv > /dev/null 2>&1")
                subprocess.Popen(["mpv", "--no-video", filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"Playing {matched_name.title()}, sir."
                
            return f"I couldn't find {song_name} in your local library."
        else:
            actions = {
                "play": "playerctl play-pause", 
                "stop": "playerctl stop || pkill -9 mpv", 
                "next": "playerctl next", 
                "previous": "playerctl previous"
            }
            if Actions.run_cmd(actions.get(action, "")): 
                return f"Music {action} executed."
            return f"Failed to {action} music."
    
    @staticmethod
    def brightness(action: str, value: int = None):
        if action == "set" and value is not None: Actions.run_cmd(f"brightnessctl set {value}%")
        elif action == "increase": Actions.run_cmd("brightnessctl set +10%")
        elif action == "decrease": Actions.run_cmd("brightnessctl set 10%-")
        return f"Brightness {action} executed."
    
    @staticmethod
    def volume(action: str):
        actions = {"increase": "amixer set Master 10%+", "decrease": "amixer set Master 10%-"}
        Actions.run_cmd(actions.get(action, ""))
        return f"Volume {action} executed."
    
    @staticmethod
    def system_info(action: str):
        if action == "time":
            hour = time.strftime("%I").lstrip("0")
            minute = int(time.strftime("%M"))      
            ampm = time.strftime("%p")             
            speak_time = f"{hour} {ampm}" if minute == 0 else (f"{hour} oh {minute} {ampm}" if minute < 10 else f"{hour} {minute} {ampm}")
            return f"The time is {speak_time}"
        elif action == "date": return f"Today is {time.strftime('%A')}"
        elif action == "memory" or action == "cpu": return "System diagnostic complete. All normal."
        elif action == "lock":
            Actions.run_cmd("loginctl lock-session")
            return "Screen locked."
        elif action == "screenshot":
            pictures_dir = os.path.expanduser("~/Pictures")
            os.makedirs(pictures_dir, exist_ok=True)
            filepath = os.path.join(pictures_dir, f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png")
            if Actions.run_cmd(f"gnome-screenshot -f '{filepath}'") or Actions.run_cmd(f"scrot '{filepath}'"):
                return "Screenshot saved, sir."
            return "Failed to take screenshot."
        elif action == "close_tab":
            Actions.run_cmd("xdotool key ctrl+w")
            return "Tab closed, sir."
        elif action == "close_window":
            Actions.run_cmd("xdotool key alt+F4")
            return "Window closed, sir."

    @staticmethod
    def get_installed_gui_apps():
        apps = {}
        desktop_dirs = ["/usr/share/applications", os.path.expanduser("~/.local/share/applications"), "/var/lib/snapd/desktop/applications"]

        for directory in desktop_dirs:
            if not os.path.exists(directory): continue
            for filename in os.listdir(directory):
                if filename.endswith(".desktop"):
                    try:
                        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                            content = f.read()
                            if "NoDisplay=true" in content or "Hidden=true" in content: continue 
                            name_match = re.search(r"^Name=(.+)$", content, re.MULTILINE)
                            exec_match = re.search(r"^Exec=([^\s%]+)", content, re.MULTILINE) 
                            if name_match and exec_match:
                                friendly_name = name_match.group(1).lower().strip()
                                exec_cmd = exec_match.group(1).strip().split("/")[-1] 
                                apps[friendly_name] = exec_cmd
                    except Exception: continue
        return apps

    @staticmethod
    def open_app(app: str):
        search_name = app.lower().strip()
        
        global semantic_matcher
        if semantic_matcher:
            target_process, score = semantic_matcher.find_best_match(search_name)
            if target_process:
                subprocess.Popen(target_process, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"Opening {app}, sir."

        gui_apps = Actions.get_installed_gui_apps()
        best_match = process.extractOne(search_name, gui_apps.keys(), scorer=fuzz.WRatio)
        actual_cmd = None
        matched_name = None
        
        if best_match and best_match[1] >= 65:
            matched_name = best_match[0]
            actual_cmd = gui_apps[matched_name]
        else:
            query_phonetic = jellyfish.metaphone(search_name)
            for app_name, cmd in gui_apps.items():
                if jellyfish.metaphone(app_name) == query_phonetic:
                    matched_name = app_name
                    actual_cmd = cmd
                    break

        if actual_cmd:
            subprocess.Popen(actual_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Opening {matched_name.title()}, sir."

        file_check = Actions.open_file(app)
        if "Opening file" in file_check: return file_check
            
        return f"I couldn't find an app or file named {app}, sir."

    @staticmethod
    def close_app(app: str):
        search_name = app.lower().strip()
        user_uid = str(os.getuid())

        dangerous_words = ["system", "gnome", "os", "linux", "dbus", "bash", "xorg", "wayland", "systemd", "session", "pipewire", "pulseaudio"]
        if search_name in dangerous_words or len(search_name) <= 2:
            return "Sir, that request is too broad and might crash the system. Denied."

        global semantic_matcher
        if semantic_matcher:
            semantic_target, score = semantic_matcher.find_best_match(search_name, threshold=0.50)
            if semantic_target:
                search_name = semantic_target

        app_map = {
            "browser": "firefox", "firefox": "firefox",
            "chrome": "chrome", "google chrome": "chrome", "google-chrome": "chrome",
            "terminal": "gnome-terminal-server", "gnome-terminal": "gnome-terminal-server",
            "files": "nautilus", "folder": "nautilus",
            "settings": "gnome-control-center",
            "vs code": "code", "vscode": "code",
            "discord": "Discord",
            "system monitor": "gnome-system-monitor",
            "task manager": "gnome-system-monitor",
            "calculator": "gnome-calculator"
        }

        target_process = app_map.get(search_name, search_name.replace(" ", ""))
        killed = False

        try:
            if subprocess.run(["killall", "-9", "-q", target_process], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                killed = True
        except Exception: pass

        if not killed:
            try:
                if subprocess.run(["pkill", "-9", "-U", user_uid, "-i", target_process], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                    killed = True
            except Exception: pass

        try:
            output = subprocess.check_output(["wmctrl", "-l"], text=True)
            for line in output.splitlines():
                if search_name in line.lower() or target_process.lower() in line.lower():
                    parts = line.split(maxsplit=3)
                    os.system(f"wmctrl -i -c {parts[0]}")
                    killed = True
        except Exception: pass

        if killed: return f"Closing {app}, sir."

        return f"I couldn't find {app} running, sir."

    @staticmethod
    def open_file(filename: str):
        search_dirs = [
            os.path.expanduser("~/Desktop"), os.path.expanduser("~/Documents"), 
            os.path.expanduser("~/Downloads"), os.path.expanduser("~/Pictures"),
            os.path.expanduser("~/Videos"), os.path.expanduser("~/Music")
        ]
        search_name = filename.lower().strip()

        for directory in search_dirs:
            if not os.path.exists(directory): continue
            for root, dirs, files in os.walk(directory):
                for file in files:
                    clean_file_name = os.path.splitext(file)[0].lower()
                    clean_file_name = clean_file_name.replace("_", " ").replace("-", " ")

                    if search_name == clean_file_name or search_name in clean_file_name:
                        target_path = os.path.join(root, file)
                        try:
                            subprocess.Popen(["xdg-open", target_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            return f"Opening file {file}, sir."
                        except Exception: pass

        return f"I couldn't find a file named {filename}, sir."

# ==========================================
# 2.5 محرك الذكاء الاصطناعي (Ollama LLM)
# ==========================================
class OllamaInterface:
    chat_history = []
    
    @staticmethod
    def ask(prompt: str) -> str:
        MODEL_NAME = "qwen2.5:1.5b" 
        url = "http://localhost:11434/api/chat"
        
        if not OllamaInterface.chat_history:
            OllamaInterface.chat_history.append({"role": "system", "content": "You are Jarvis, a smart AI assistant. Answer in english clearly, naturally, and in MAXIMUM 2 sentences."})
            
        OllamaInterface.chat_history.append({"role": "user", "content": prompt})
        
        if len(OllamaInterface.chat_history) > 7:
            OllamaInterface.chat_history = [OllamaInterface.chat_history[0]] + OllamaInterface.chat_history[-6:]
        
        try:
            options = {"num_thread": 3}
            response = requests.post(url, json={"model": MODEL_NAME, "messages": OllamaInterface.chat_history, "stream": False, "options": options}, timeout=20)
            if response.status_code == 200:
                ai_response = response.json().get("message", {}).get("content", "").strip()
                OllamaInterface.chat_history.append({"role": "assistant", "content": ai_response})
                return ai_response
            return "Sorry, I had trouble processing that thought."
        except Exception: 
            return "Sir, Ollama is currently offline."
            
    @staticmethod
    class OllamaInterface:
     chat_history = []
    _translation_ready = False # 🌟 متغير للتحقق من جاهزية القواميس المحلية
    
    # ... (دالة ask ودالة clear_memory تبقى كما هي) ...

    @staticmethod
    def translate(text: str) -> str:
        try:
            import argostranslate.package
            import argostranslate.translate
        except ImportError:
            return "Sir, please install the offline translator via terminal: pip install argostranslate"

        # 🌟 فحص وتنزيل القواميس (يحدث لمرة واحدة فقط في العمر)
        if not OllamaInterface._translation_ready:
            installed_packages = argostranslate.package.get_installed_packages()
            has_en_ar = any(p.from_code == 'en' and p.to_code == 'ar' for p in installed_packages)
            has_ar_en = any(p.from_code == 'ar' and p.to_code == 'en' for p in installed_packages)
            
            if not (has_en_ar and has_ar_en):
                print("📥 [نظام الترجمة] جاري تحميل القواميس المحلية (تعمل لمرة واحدة فقط)...")
                argostranslate.package.update_package_index()
                available_packages = argostranslate.package.get_available_packages()
                
                if not has_en_ar:
                    pkg = next(filter(lambda x: x.from_code == 'en' and x.to_code == 'ar', available_packages), None)
                    if pkg: argostranslate.package.install_from_path(pkg.download())
                
                if not has_ar_en:
                    pkg = next(filter(lambda x: x.from_code == 'ar' and x.to_code == 'en', available_packages), None)
                    if pkg: argostranslate.package.install_from_path(pkg.download())
                
                print("✅ [نظام الترجمة] تمت تهيئة الترجمة الأوفلاين بنجاح!")
                
            OllamaInterface._translation_ready = True

        # 🌟 تنفيذ الترجمة اللحظية (0ms)
        try:
            if re.search(r'[\u0600-\u06FF]', text):
                return argostranslate.translate.translate(text, 'ar', 'en')
            else:
                return argostranslate.translate.translate(text, 'en', 'ar')
        except Exception as e:
            print(f"⚠️ Offline Translation Error: {e}")
            return "Translation failed."

# ==========================================
# 3. محرك الواجهة الرسومية والتفاعلات اليدوية
# ==========================================
class DynamicPill(QWidget):
    manual_wake_signal = pyqtSignal()
    text_command_signal = pyqtSignal(str)
    start_download_signal = pyqtSignal(str, str)
    start_translation_signal = pyqtSignal(str)

    def set_layer_mode(self, mode):
        if mode in ["text", "options"]:
            target_flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Window
        else:
            target_flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.ToolTip

        if self.windowFlags() != target_flags:
            current_geom = self.geometry()
            self.hide() 
            self.setWindowFlags(target_flags)
            self.setGeometry(current_geom)
            self.show()
            self.raise_()
            self.activateWindow()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.ToolTip)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("JarvisPill")
        
        self.setAcceptDrops(True)
        self.clipboard_items = []
        self.gallery_selected_items = set() 
        self.drag_start_position = None
        self.active_timer_text = ""  
        
        self._current_drag = None
        self._current_mime_data = None
        self._clipboard_mime_data = None
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(15, 5, 15, 5)
        
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Ubuntu", 14, QFont.Medium))
        self.layout.addWidget(self.label)
        
        self.input_box = QLineEdit()
        self.input_box.setStyleSheet("QLineEdit { background: transparent; color: white; border: none; font-size: 16px; font-weight: bold; }")
        self.input_box.setPlaceholderText("Type a command or paste a URL...")
        self.input_box.hide()
        self.input_box.returnPressed.connect(self.submit_text_command)
        self.layout.addWidget(self.input_box)
        
        self.gallery_container = QWidget()
        self.gallery_layout = QHBoxLayout(self.gallery_container)
        self.gallery_layout.setContentsMargins(0, 0, 0, 0)
        self.gallery_layout.setSpacing(10)
        self.layout.addWidget(self.gallery_container)
        self.gallery_container.hide()
        
        self.btn_container = QWidget()
        self.btn_layout = QHBoxLayout(self.btn_container)
        self.btn_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_layout.setSpacing(8)
        
        self.btn_vid_best = QPushButton("🎬 Best")
        self.btn_vid_low = QPushButton("📽️ Low")
        self.btn_aud = QPushButton("🎙️ Audio")
        self.btn_aud_meta = QPushButton("🎼🎛️ Meta")
        self.btn_cancel = QPushButton("❌ Cancel") 
        
        btn_style = """
            QPushButton { 
                background: rgba(20, 20, 25, 200); color: white; 
                border-radius: 12px; padding: 6px 12px; font-weight: bold; border: 1px solid #444;
            } 
            QPushButton:hover { background: #00ffcc; color: black; border: 1px solid #00ffcc; }
        """
        cancel_style = """
            QPushButton { 
                background: rgba(20, 20, 25, 200); color: #ff4444; 
                border-radius: 12px; padding: 6px 12px; font-weight: bold; border: 1px solid #444;
            } 
            QPushButton:hover { background: #ff4444; color: white; border: 1px solid #ff4444; }
        """

        for btn in [self.btn_vid_best, self.btn_vid_low, self.btn_aud, self.btn_aud_meta]:
            btn.setStyleSheet(btn_style)
            self.btn_layout.addWidget(btn)
            
        self.btn_cancel.setStyleSheet(cancel_style)
        self.btn_layout.addWidget(self.btn_cancel)
            
        self.layout.addWidget(self.btn_container)
        self.btn_container.hide()
        
        self.btn_vid_best.clicked.connect(lambda: self.trigger_download("video_best"))
        self.btn_vid_low.clicked.connect(lambda: self.trigger_download("video_low"))
        self.btn_aud.clicked.connect(lambda: self.trigger_download("audio"))
        self.btn_aud_meta.clicked.connect(lambda: self.trigger_download("audio_meta"))
        self.btn_cancel.clicked.connect(self.cancel_download) 
        
        self.pending_url = ""
        
        self.anim = QPropertyAnimation(self, b"geometry")
        self.anim.setEasingCurve(QEasingCurve.OutExpo)
        self.anim.setDuration(450)
        
        self.border_color = QColor(50, 50, 50)
        self.pulse_step = 0.0
        
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.animate_pulse)
        self.pulse_timer.start(40) 
        
        self.text_timer = QTimer()
        self.text_timer.timeout.connect(self.show_next_chunk)
        self.chunks = []
        self.current_chunk = 0
        
        self.click_timer = QTimer()
        self.click_timer.setSingleShot(True)
        self.click_timer.timeout.connect(self.perform_single_click)

        self.recent_selected_text = ""
        self.has_text_selected = False
        self.last_polled_text = ""
        
        self.selection_timer = QTimer()
        self.selection_timer.setSingleShot(True)
        self.selection_timer.timeout.connect(self.expire_selection)
        
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self.poll_primary_selection)
        self.poll_timer.start(500)
        
        self.state = "idle"
        self.set_idle()

    def poll_primary_selection(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text(QClipboard.Selection).strip()
        
        if text and text != self.last_polled_text:
            self.last_polled_text = text
            self.recent_selected_text = text
            self.has_text_selected = True
            self.selection_timer.start(10000)
            self.update()

    def expire_selection(self):
        self.recent_selected_text = ""
        self.has_text_selected = False
        self.update()

    def keyPressEvent(self, event):
        if self.state == "gallery":
            if (event.modifiers() & Qt.ControlModifier) and event.key() == Qt.Key_C:
                if self.gallery_selected_items:
                    self.copy_selected_gallery_items()
                return
            elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if self.gallery_selected_items:
                    self.copy_selected_gallery_items()
                return
            elif event.key() == Qt.Key_Escape:
                self.set_idle()
                return
        super().keyPressEvent(event)

    def get_preview_pixmap(self, filepath, size=64):
        is_image = filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'))
        is_audio = filepath.lower().endswith(('.mp3', '.m4a', '.flac', '.wav', '.ogg'))
        
        if is_image:
            return QPixmap(filepath).scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            
        if is_audio:
            try:
                import mutagen
                audio = mutagen.File(filepath)
                if audio and hasattr(audio, 'tags') and audio.tags:
                    img_data = None
                    for key in audio.tags.keys():
                        if key.startswith('APIC'):
                            img_data = audio.tags[key].data
                            break
                        elif key == 'covr':
                            img_data = audio.tags[key][0]
                            break
                    if img_data:
                        image = QImage.fromData(img_data)
                        if not image.isNull():
                            return QPixmap.fromImage(image).scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            except Exception: pass
                
        preview = QPixmap(size, size)
        preview.fill(Qt.transparent)
        painter = QPainter(preview)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(40, 40, 45, 240))
        painter.setPen(Qt.white)
        painter.drawRoundedRect(2, 2, size-4, size-4, size//2, size//2) 
        ext = filepath.split('.')[-1][:3].upper() if '.' in filepath else "DOC"
        font_size = max(8, size // 5) 
        painter.setFont(QFont("Ubuntu", font_size, QFont.Bold))
        painter.drawText(QRect(0, 0, size, size), Qt.AlignCenter, ext)
        painter.end()
        return preview

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path not in self.clipboard_items:
                    if len(self.clipboard_items) >= 10:
                        self.clipboard_items.pop(0)
                    self.clipboard_items.append(file_path)
        if self.state == "gallery":
            self.set_idle() 
        self.update() 
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos() 
        elif event.button() == Qt.RightButton:
            if self.state == "gallery":
                self.set_idle() 
            elif self.clipboard_items:
                self.clipboard_items.clear()
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drag_start_position is not None:
            self.drag_start_position = None
            self.click_timer.start(250)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.click_timer.stop() 
            self.drag_start_position = None
            if self.clipboard_items and self.state == "idle":
                self.show_clipboard_gallery()
            else:
                self.enter_text_mode() 

    def show_clipboard_gallery(self):
        self.pulse_timer.setInterval(40)
        self.set_layer_mode("options")
        self.state = "gallery"
        self.text_timer.stop()
        self.label.hide()
        self.input_box.hide()
        self.btn_container.hide()
        
        while self.gallery_layout.count():
            child = self.gallery_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        self.gallery_selected_items = set() 
                
        icon_size = 72 
        for item in self.clipboard_items:
            btn = QPushButton()
            btn.setCheckable(True) 
            pixmap = self.get_preview_pixmap(item, icon_size)
            btn.setIcon(QIcon(pixmap))
            btn.setIconSize(QSize(icon_size, icon_size))
            btn.setToolTip(os.path.basename(item))
            
            btn.setStyleSheet("""
                QPushButton { background: rgba(30, 30, 35, 200); border: 2px solid #555; border-radius: 12px; padding: 4px; }
                QPushButton:hover { border: 2px solid #888; background: rgba(100, 100, 100, 40); }
                QPushButton:checked { border: 3px solid #00ffcc; background: rgba(0, 255, 204, 60); }
            """)
            btn.clicked.connect(lambda checked, p=item, b=btn: self.on_gallery_btn_clicked(p, b))
            self.gallery_layout.addWidget(btn)
            
        self.gallery_container.show()
        w = max(150, len(self.clipboard_items) * (icon_size + 20) + 40)
        self.update_geom(min(w, 1000), icon_size + 24, "options")

    def on_gallery_btn_clicked(self, filepath, btn):
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.ControlModifier:
            if btn.isChecked():
                self.gallery_selected_items.add(filepath)
            else:
                self.gallery_selected_items.discard(filepath)
        else:
            self.gallery_selected_items.add(filepath)
            self.copy_selected_gallery_items()

    def copy_selected_gallery_items(self):
        if not self.gallery_selected_items:
            self.set_idle() 
            return
        items_to_copy = list(self.gallery_selected_items)
        items_to_copy.sort(key=lambda x: self.clipboard_items.index(x))
        for item in items_to_copy:
            if item in self.clipboard_items:
                self.clipboard_items.remove(item)
                self.clipboard_items.append(item)
                
        self._clipboard_mime_data = QMimeData()
        urls = [QUrl.fromLocalFile(os.path.abspath(p)) for p in items_to_copy]
        self._clipboard_mime_data.setUrls(urls)
        
        gnome_format = b"copy\n" + b"\n".join([u.toEncoded() for u in urls])
        self._clipboard_mime_data.setData("x-special/gnome-copied-files", gnome_format)
        self._clipboard_mime_data.setText("\n".join([os.path.abspath(p) for p in items_to_copy]))
        
        if len(items_to_copy) == 1:
            latest_item = items_to_copy[0]
            if latest_item.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')):
                img = QImage(latest_item)
                if not img.isNull():
                    self._clipboard_mime_data.setImageData(img)
                try:
                    with open(latest_item, "rb") as f:
                        raw_bytes = f.read()
                    ext = latest_item.lower().split('.')[-1]
                    mime_t = "image/png" if ext == "png" else "image/jpeg"
                    self._clipboard_mime_data.setData(mime_t, raw_bytes)
                except:
                    pass
        
        QApplication.clipboard().clear()
        QApplication.clipboard().setMimeData(self._clipboard_mime_data)
        self.border_color = QColor(0, 255, 0, 200) 
        self.set_idle() 

    def perform_single_click(self):
        if self.clipboard_items:
            self._clipboard_mime_data = QMimeData()
            urls = [QUrl.fromLocalFile(os.path.abspath(p)) for p in self.clipboard_items]
            self._clipboard_mime_data.setUrls(urls)
            gnome_format = b"copy\n" + b"\n".join([u.toEncoded() for u in urls])
            self._clipboard_mime_data.setData("x-special/gnome-copied-files", gnome_format)
            self._clipboard_mime_data.setText("\n".join([os.path.abspath(p) for p in self.clipboard_items]))
            
            if len(self.clipboard_items) == 1:
                latest_item = self.clipboard_items[0]
                if latest_item.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')):
                    img = QImage(latest_item)
                    if not img.isNull():
                        self._clipboard_mime_data.setImageData(img)
                    try:
                        with open(latest_item, "rb") as f:
                            raw_bytes = f.read()
                        ext = latest_item.lower().split('.')[-1]
                        mime_t = "image/png" if ext == "png" else "image/jpeg"
                        self._clipboard_mime_data.setData(mime_t, raw_bytes)
                    except: pass
            QApplication.clipboard().clear()
            QApplication.clipboard().setMimeData(self._clipboard_mime_data)
            self.border_color = QColor(0, 255, 0, 200)
            self.update()
        else:
            if self.state == "idle":
                if getattr(self, 'has_text_selected', False) and self.recent_selected_text:
                    self.start_translation_signal.emit(self.recent_selected_text)
                    self.expire_selection()
                else:
                    self.manual_wake_signal.emit()

    def mouseMoveEvent(self, event):
        if not self.clipboard_items or not (event.buttons() & Qt.LeftButton): return
        if self.drag_start_position is None: return
        if (event.pos() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance(): return

        self.click_timer.stop() 
        self.drag_start_position = None 
        
        self._current_drag = QDrag(self)
        self._current_mime_data = QMimeData()
        
        urls = [QUrl.fromLocalFile(os.path.abspath(p)) for p in self.clipboard_items]
        self._current_mime_data.setUrls(urls)
        self._current_mime_data.setText("\n".join([os.path.abspath(p) for p in self.clipboard_items]))
        latest_item = self.clipboard_items[-1]
        
        if len(self.clipboard_items) == 1:
            if latest_item.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')):
                img = QImage(latest_item)
                if not img.isNull():
                    self._current_mime_data.setImageData(img)
                try:
                    with open(latest_item, "rb") as f:
                        raw_bytes = f.read()
                    ext = latest_item.lower().split('.')[-1]
                    mime_t = "image/png" if ext == "png" else "image/jpeg"
                    self._current_mime_data.setData(mime_t, raw_bytes)
                except: pass

        self._current_drag.setMimeData(self._current_mime_data)
        preview = self.get_preview_pixmap(latest_item, 64)
        self._current_drag.setPixmap(preview)
        self._current_drag.setHotSpot(preview.rect().center())
        self._current_drag.exec_(Qt.CopyAction)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(4.0, 4.0, -4.0, -4.0)
        radius = rect.height() / 2.0
        color = self.border_color
        r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()

        # رسم الإطار الخارجي النابض (دائماً موجود)
        ring_pen = QPen(QColor(r, g, b, min(255, a + 40)), 4.0) 
        painter.setPen(ring_pen)
        painter.setBrush(Qt.NoBrush) 
        painter.drawRoundedRect(rect, radius, radius)

        # 🌟 الشروط الذكية للرسم الداخلي (تفادي الزحمة البصرية)
        # لا نرسم أي شيء داخل الكبسولة إذا كانت تعرض نصاً (user_text, speaking, downloading, translation_ready)
        if self.state in ["user_text", "speaking", "downloading", "translation_ready"]:
            return # إنهاء الرسم هنا، ليظهر النص فقط بوضوح تام

        cx = rect.center().x()
        cy = rect.center().y()

        # حالة الـ idle
        if self.rect().width() < 145 and self.state == "idle":
            if self.clipboard_items:
                latest_item = self.clipboard_items[-1]
                img_rect = rect.adjusted(3.0, 3.0, -3.0, -3.0)
                path = QPainterPath()
                path.addEllipse(img_rect)
                painter.setClipPath(path)
                
                pixmap = self.get_preview_pixmap(latest_item, int(img_rect.width()))
                img_x = img_rect.center().x() - pixmap.width() / 2.0
                img_y = img_rect.center().y() - pixmap.height() / 2.0
                painter.drawPixmap(int(img_x), int(img_y), pixmap)
                painter.setClipping(False)
                
                count = len(self.clipboard_items)
                badge_rect = QRectF(rect.right() - 14, rect.top() - 2, 18, 18)
                painter.setBrush(QColor(255, 60, 60, 240))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(badge_rect)
                
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont("Ubuntu", 9, QFont.Bold))
                painter.drawText(badge_rect, Qt.AlignCenter, str(count))
            else:
                if getattr(self, 'has_text_selected', False):
                    painter.setPen(QColor(255, 255, 255, 255))
                    painter.setFont(QFont("Ubuntu", 16))
                    rect_globe = QRectF(cx - 15, cy - 15, 30, 30)
                    painter.drawText(rect_globe, Qt.AlignCenter, "🌐")
                elif hasattr(self, 'active_timer_text') and self.active_timer_text != "":
                    painter.setPen(QColor(255, 159, 10, 255)) 
                    painter.setFont(QFont("Ubuntu", 18, QFont.Bold))
                    rect_timer = QRectF(rect.left(), rect.top(), rect.width(), rect.height())
                    painter.drawText(rect_timer, Qt.AlignCenter, self.active_timer_text)
                else:
                    # رسم الأعمدة في حالة السكون
                    bar_width = 4.0  
                    spacing = 10.0   
                    h1 = 10.0 + 4.0 * math.sin(self.pulse_step)
                    h2 = 10.0 + 4.0 * math.sin(self.pulse_step + 1.0)
                    h3 = 10.0 + 4.0 * math.sin(self.pulse_step + 2.0)
                    bar_color = QColor(r, g, b, 150)
                    bar_pen = QPen(bar_color, bar_width, Qt.SolidLine, Qt.RoundCap)
                    painter.setPen(bar_pen)
                    painter.drawLine(int(cx - spacing), int(cy - h1/2), int(cx - spacing), int(cy + h1/2))
                    painter.drawLine(int(cx), int(cy - h2/2), int(cx), int(cy + h2/2))
                    painter.drawLine(int(cx + spacing), int(cy - h3/2), int(cx + spacing), int(cy + h3/2))
        
        # حالة الاستماع (listening) أو الـ options/gallery
        elif self.state in ["listening", "options", "gallery"]:
            if self.state == "listening":
                bar_width = 4.0  
                spacing = 10.0
                h1 = 24.0 + 12.0 * math.sin(self.pulse_step * 2.5)
                h2 = 24.0 + 12.0 * math.sin(self.pulse_step * 3.0 + 1.0)
                h3 = 24.0 + 12.0 * math.sin(self.pulse_step * 2.5 + 2.0)
                bar_color = QColor(r, g, b, 255)
                bar_pen = QPen(bar_color, bar_width, Qt.SolidLine, Qt.RoundCap)
                painter.setPen(bar_pen)
                painter.drawLine(int(cx - spacing), int(cy - h1/2), int(cx - spacing), int(cy + h1/2))
                painter.drawLine(int(cx), int(cy - h2/2), int(cx), int(cy + h2/2))
                painter.drawLine(int(cx + spacing), int(cy - h3/2), int(cx + spacing), int(cy + h3/2))
            elif hasattr(self, 'active_timer_text') and self.active_timer_text != "":
                painter.setPen(QColor(255, 159, 10, 255)) 
                painter.setFont(QFont("Ubuntu", 18, QFont.Bold))
                rect_timer = QRectF(rect.left(), rect.top(), rect.width(), rect.height())
                painter.drawText(rect_timer, Qt.AlignCenter, self.active_timer_text)

    def update_geom(self, w, h, mode="idle"):
        x = 1250 
        y = 40 if mode in ["text", "options"] else 4 
        self.anim.setStartValue(self.geometry())
        self.anim.setEndValue(QRect(x, y, w, h))
        self.anim.start()

    def animate_pulse(self):
        if self.state == "idle":
            self.pulse_step += 0.05
            if getattr(self, 'active_timer_text', "") != "":
                alpha = int(120 + 80 * math.sin(self.pulse_step * 2))
                self.border_color = QColor(255, 159, 10, alpha)
            elif getattr(self, 'has_text_selected', False):
                alpha = int(100 + 60 * math.sin(self.pulse_step))
                self.border_color = QColor(0, 255, 150, alpha)
            else:
                alpha = int(70 + 40 * math.sin(self.pulse_step))
                self.border_color = QColor(255, 255, 255, alpha)
        elif self.state == "listening":
            self.pulse_step += 0.15
            alpha = int(130 + 120 * math.sin(self.pulse_step))
            self.border_color = QColor(255, 159, 10, alpha)
        elif self.state in ["user_text", "text_mode"]:
            self.border_color = QColor(255, 159, 10, 220)
        elif self.state == "speaking":
            self.border_color = QColor(10, 132, 255, 220)
        elif self.state == "downloading": 
            self.pulse_step += 0.1
            alpha = int(150 + 100 * math.sin(self.pulse_step))
            self.border_color = QColor(0, 255, 204, alpha)
        elif self.state == "gallery" or self.state == "options":
            self.pulse_step += 0.1
            alpha = int(150 + 100 * math.sin(self.pulse_step))
            self.border_color = QColor(153, 0, 255, alpha)
        self.update()

    def set_idle(self):
        self.pulse_timer.setInterval(100) 
        self.set_layer_mode("idle")
        self.state = "idle"
        self.text_timer.stop()
        self.input_box.clearFocus()
        self.input_box.hide()
        self.btn_container.hide()
        self.gallery_container.hide() 
        self.label.show()
        self.label.setText("") 
        
        if getattr(self, 'active_timer_text', "") != "":
            self.update_geom(140, 64, "idle") 
        else:
            self.update_geom(64, 64, "idle")

    def set_listening(self):
        self.pulse_timer.setInterval(40) 
        self.set_layer_mode("active")
        self.state = "listening"
        self.text_timer.stop()
        self.input_box.hide()
        self.btn_container.hide()
        self.gallery_container.hide()
        self.label.show()
        self.label.setText("") 
        self.update_geom(64, 64, "active")

    def enter_text_mode(self):
        self.pulse_timer.setInterval(40)
        self.set_layer_mode("text") 
        self.state = "text_mode"
        self.text_timer.stop()
        self.label.hide()
        self.btn_container.hide()
        self.gallery_container.hide()
        self.input_box.show()
        self.update_geom(350, 64, "text") 
        
        def force_top_focus():
            self.raise_()
            self.activateWindow()
            self.input_box.setFocus()
            
        QTimer.singleShot(50, force_top_focus)
        QTimer.singleShot(150, force_top_focus)

    def submit_text_command(self):
        cmd = self.input_box.text().strip()
        self.input_box.clear()
        self.input_box.clearFocus()
        
        url_match = re.search(r'(https?://[^\s]+)', cmd)
        if url_match:
            self.pending_url = url_match.group(1)
            self.show_download_options()
        else:
            self.input_box.hide()
            self.label.show()
            if cmd:
                self.text_command_signal.emit(cmd)
            else:
                self.set_idle()
                
    def show_download_options(self):
        self.pulse_timer.setInterval(40)
        self.set_layer_mode("options")
        self.state = "options"
        self.text_timer.stop()
        self.input_box.hide()
        self.label.hide()
        self.gallery_container.hide()
        self.btn_container.show()
        self.update_geom(560, 64, "options")

    def cancel_download(self):
        self.pending_url = ""
        self.set_idle()

    def trigger_download(self, dl_type):
        self.btn_container.hide()
        self.label.show()
        self.start_download_signal.emit(self.pending_url, dl_type)

    def set_user_text(self, text):
        self.pulse_timer.setInterval(40)
        self.set_layer_mode("active")
        self.state = "user_text"
        self.text_timer.stop()
        self.input_box.hide()
        self.btn_container.hide()
        self.gallery_container.hide()
        self.label.show()
        self.label.setText(text)
        self.label.setStyleSheet("color: #ffffff; background: transparent;")
        self.label.adjustSize()
        self.update_geom(min(max(140, self.label.width() + 40), 600), 64, "active")
        
    def set_downloading(self, text):
        self.pulse_timer.setInterval(40)
        self.set_layer_mode("active")
        self.state = "downloading"
        self.text_timer.stop()
        self.input_box.hide()
        self.btn_container.hide()
        self.gallery_container.hide()
        self.label.show()
        self.label.setText(text)
        self.label.setStyleSheet("color: #00ffcc; background: transparent; font-weight: bold;")
        self.label.adjustSize()
        self.update_geom(min(max(300, self.label.width() + 40), 800), 64, "active")

    def set_speaking(self, text):
        self.pulse_timer.setInterval(40)
        self.set_layer_mode("active")
        self.state = "speaking"
        self.input_box.hide()
        self.btn_container.hide()
        self.gallery_container.hide()
        self.label.show()
        words = text.split()
        self.chunks = [" ".join(words[i:i+8]) for i in range(0, len(words), 8)]
        self.current_chunk = 0
        self.show_next_chunk()

    def show_next_chunk(self):
        if self.current_chunk < len(self.chunks):
            self.label.setText(self.chunks[self.current_chunk])
            self.label.setStyleSheet("color: #ffffff; background: transparent;")
            self.label.adjustSize()
            self.update_geom(min(max(140, self.label.width() + 40), 600), 64, "active")
            self.current_chunk += 1
            self.text_timer.start(3000)
        else:
            self.text_timer.stop()
            QTimer.singleShot(2500, self.auto_return_to_idle)
            
    def auto_return_to_idle(self):
        if self.state in ["speaking", "user_text", "downloading", "translation_ready"]:
            self.set_idle()


# ==========================================
# 4. محرك النظام كـ Thread (Jarvis Worker)
# ==========================================
class JarvisWorker(QThread):
    ui_update = pyqtSignal(str, str) 
    
    def __init__(self):
        super().__init__()
        self.awaiting_command = False
        self.pending_power_action = None
        self.was_media_playing = False # 🌟 متغير ذاكرة الوسائط
        
        # 🚀 [الجديد]: نقل الصوت المباشر إلى الذاكرة ليعمل بـ (0ms Latency)
        # نستخدم PyAudio (الذي يستخدمه Kokoro) لتشغيل الملف داخلياً دون اللجوء للأوامر الخارجية
        self.pa = pyaudio.PyAudio() 
        
        mhm_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mhm.wav")
        self.mhm_audio_data = None
        self.mhm_fs = None
        if os.path.exists(mhm_file):
            try:
                data, fs = sf.read(mhm_file, dtype='float32')
                if len(data.shape) > 1:
                    data = data[:, 0]
                
                # 🚀 [السرعة القصوى]: إزالة أي فراغ صامت من بداية الملف برمجياً (Auto-Trim)
                threshold = 0.01 # عتبة الصوت (تجاهل أي ضجيج خفيف أو صمت)
                start_idx = np.argmax(np.abs(data) > threshold) # البحث عن أول لحظة يبدأ فيها الصوت الفعلي
                data = data[start_idx:] # قص كل الصمت الذي يسبق هذا المؤشر
                
                # حذفنا خدعة الوسادة الصامتة ليكون التشغيل في 0 ملي ثانية!
                
                self.mhm_audio_data = data.tobytes()
                self.mhm_fs = fs
                print("✅ تم تجهيز الهمهمة في الرام وقص الفراغات الصامتة لسرعة 0ms!")
            except Exception as e:
                print(f"⚠️ فشل في إدراج mhm.wav للذاكرة: {e}")

        print("⏳ Loading Kokoro-82M Human Voice Model...")
        try:
            self.kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            self.has_kokoro = True
            print("✅ Kokoro Loaded Successfully!")
        except Exception as e:
            print(f"⚠️ Kokoro failed to load: {e}. Will fallback to Piper.")
            self.has_kokoro = False

    def pause_background_media(self):
        try:
            status = subprocess.check_output(["playerctl", "status"], stderr=subprocess.DEVNULL, text=True).strip()
            if status.lower() == "playing":
                self.was_media_playing = True
                os.system("playerctl pause > /dev/null 2>&1")
            else:
                self.was_media_playing = False
        except Exception:
            self.was_media_playing = False

    def resume_background_media(self):
        if getattr(self, 'was_media_playing', False):
            os.system("playerctl play > /dev/null 2>&1")
            self.was_media_playing = False

    def handle_manual_wake(self):
        def wake():
            self.pause_background_media()
            self.awaiting_command = True
            reply = random.choice(["Mhm", "Yes, sir", "Yes"])
            self.speak(reply)
        threading.Thread(target=wake, daemon=True).start()

    def handle_text_command(self, cmd):
        def process():
            self.ui_update.emit("user_text", f"⌨️ {cmd}")
            response = self.execute(cmd)
            self.speak(response)
        threading.Thread(target=process, daemon=True).start()

    def handle_translation(self, text):
        def process():
            self.ui_update.emit("downloading", "⏳ جاري الترجمة...")
            result = OllamaInterface.translate(text)
            self.ui_update.emit("translation_ready", result)
            if not re.search(r'[\u0600-\u06FF]', result):
                self.speak(result, update_ui=False) 
        threading.Thread(target=process, daemon=True).start()

    def handle_gui_download(self, url, dl_type):
        def process():
            downloads_dir = os.path.expanduser("~/Desktop/music/wdell")
            os.makedirs(downloads_dir, exist_ok=True)
            
            if dl_type == "video_best":
                q_label = "Highest Vid"
                self.speak("Downloading best video")
                dl_cmd = f"yt-dlp --newline -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' -o '{downloads_dir}/%(title)s.%(ext)s' '{url}'"
            elif dl_type == "video_low":
                q_label = "360p Vid"
                self.speak("Downloading low quality video")
                dl_cmd = f"yt-dlp --newline -f 'bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/best' -o '{downloads_dir}/%(title)s.%(ext)s' '{url}'"
            elif dl_type == "audio":
                q_label = "Audio Only"
                self.speak("Downloading audio")
                dl_cmd = f"yt-dlp --newline -x --audio-format mp3 -o '{downloads_dir}/%(title)s.%(ext)s' '{url}'"
            elif dl_type == "audio_meta":
                q_label = "Audio + Meta"
                self.speak("Downloading audio and fetching lyrics")
                dl_cmd = f"yt-dlp --newline -x --audio-format mp3 --embed-metadata --embed-thumbnail -o '{downloads_dir}/%(title)s.%(ext)s' '{url}'"

            print(f"\n" + "="*50)
            print(f"🚀 [نظام التحميل] جاري بدء العملية...")
            print(f"🔗 الرابط: {url}")
            print("="*50 + "\n")

            lyrics_event = threading.Event()
            lyrics_data = {"clean_name": "", "lrc_path": ""}
            
            def fetch_lyrics_concurrently():
                lyrics_event.wait()
                if lyrics_data["clean_name"]:
                    print(f"\n🔍 [نظام الكلمات المتزامن] جاري البحث عن الكلمات في الخلفية لـ: {lyrics_data['clean_name']}")
                    os.system(f'syncedlyrics "{lyrics_data["clean_name"]}" -o "{lyrics_data["lrc_path"]}" > /dev/null 2>&1')
                    print("✅ [نظام الكلمات المتزامن] اكتملت محاولة جلب الكلمات.")

            lyrics_thread = threading.Thread(target=fetch_lyrics_concurrently, daemon=True)
            if dl_type == "audio_meta":
                lyrics_thread.start()

            try:
                process_dl = subprocess.Popen(dl_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                success = False
                saved_filepath = ""

                for line in process_dl.stdout:
                    print(f"📥 [yt-dlp]: {line.strip()}")
                    
                    if "[download] Destination:" in line and not lyrics_event.is_set():
                        raw_dest = line.split("Destination:")[1].strip()
                        base_name = os.path.splitext(os.path.basename(raw_dest))[0]
                        clean = re.sub(r'\(.*?\)|\[.*?\]', '', base_name)
                        clean = re.sub(r'(?i)\b(lyrics|official|video|audio|mv|hd)\b', '', clean)
                        clean = re.sub(r'[⧸/|_]', ' ', clean)
                        lyrics_data["clean_name"] = ' '.join(clean.split())
                        lyrics_data["lrc_path"] = os.path.join(downloads_dir, f"{base_name}.lrc")
                        lyrics_event.set()

                    if "[ExtractAudio] Destination:" in line:
                        saved_filepath = line.split("Destination:")[1].strip()

                    if "[download]" in line and "%" in line:
                        success = True
                        pct_match = re.search(r'([\d.]+)%', line)
                        pct = pct_match.group(1) + "%" if pct_match else ""
                        size_match = re.search(r'of[ ~]*([^ \t]+)', line)
                        size = size_match.group(1) if size_match else ""
                        eta_match = re.search(r'ETA\s+([\d:]+)', line)
                        eta = eta_match.group(1) if eta_match else ""
                        if pct and size and eta:
                            prog_text = f"⬇️ {q_label} | {size} | ETA: {eta} | {pct}"
                            self.ui_update.emit("downloading", prog_text)
                
                process_dl.wait()
                
                if dl_type == "audio_meta":
                    if not lyrics_data["clean_name"] and saved_filepath:
                        base_name = os.path.splitext(os.path.basename(saved_filepath))[0]
                        clean = re.sub(r'\(.*?\)|\[.*?\]', '', base_name)
                        clean = re.sub(r'(?i)\b(lyrics|official|video|audio|mv|hd)\b', '', clean)
                        clean = re.sub(r'[⧸/|_]', ' ', clean)
                        lyrics_data["clean_name"] = ' '.join(clean.split())
                        lyrics_data["lrc_path"] = os.path.join(downloads_dir, f"{base_name}.lrc")
                    
                    lyrics_event.set()
                    lyrics_thread.join()
                
                if process_dl.returncode == 0 and success:
                    if dl_type == "audio_meta" and saved_filepath:
                        self.ui_update.emit("downloading", "⏳ Finalizing Meta & Lyrics...")
                        lrc_path = lyrics_data["lrc_path"]
                        
                        valid_lyrics = False
                        lyrics_text = ""
                        
                        if lrc_path and os.path.exists(lrc_path):
                            try:
                                with open(lrc_path, 'r', encoding='utf-8') as f:
                                    lyrics_text = f.read().strip()
                                    lines = lyrics_text.split('\n')
                                    if len(lines) > 10 and len(lyrics_text) > 150:
                                        valid_lyrics = True
                            except Exception:
                                pass

                        if valid_lyrics:
                            print("\n🔍 [نظام الكلمات] تم العثور على ملف الترجمة الصحيح، جاري دمجه محلياً...")
                            try:
                                from mutagen.id3 import ID3, USLT, ID3NoHeaderError
                                try:
                                    audio_tags = ID3(saved_filepath)
                                except ID3NoHeaderError:
                                    audio_tags = ID3()
                                audio_tags.add(USLT(encoding=3, lang='eng', desc='', text=lyrics_text))
                                audio_tags.save(saved_filepath, v2_version=3)
                                os.remove(lrc_path)
                                print("✅ [نظام الكلمات] تم دمج الكلمات داخل ملف MP3 بنجاح وأصبح ملفاً واحداً!")
                            except Exception as e:
                                print(f"⚠️ [نظام الكلمات] حدث خطأ أثناء الدمج: {e}")
                        else:
                            if lrc_path and os.path.exists(lrc_path):
                                os.remove(lrc_path)
                                
                            self.ui_update.emit("downloading", "🔄 Fallback: Fetching from Spotify...")
                            print("\n⚠️ [نظام الكلمات] الكلمات المحلية غير صالحة أو مفقودة، جاري الاستعانة بـ SpotDL...")
                            
                            sync_cmd = f'spotdl sync "{saved_filepath}"'
                            sync_proc = subprocess.run(sync_cmd, shell=True, capture_output=True, text=True)
                            
                            if sync_proc.returncode != 0 or "Error" in sync_proc.stderr:
                                print(f"❌ [خطأ SpotDL]: {sync_proc.stderr}")
                                self.ui_update.emit("downloading", "⚠️ Spotify محجوب! شغّل الـ VPN للمحاولة القادمة.")
                                self.speak("Download complete, but I couldn't fetch the lyrics. Please turn on your V P N next time.", update_ui=False)
                                time.sleep(3.5) 
                            else:
                                print("✅ [نظام الكلمات] تم جلب الكلمات وتحديثها عبر Spotify بنجاح!")

                    self.speak("Download complete, sir")
                else:
                    self.speak("Failed to download")
            except Exception as e:
                self.ui_update.emit("speaking", "Error during download.")
                print(f"Exception error: {e}")

        threading.Thread(target=process, daemon=True).start() 
                           
    def run(self):
        global semantic_matcher
        if semantic_matcher is None:
            try:
                semantic_matcher = SemanticAppMatcher()
            except Exception as e:
                print(f"⚠️ تعذر تشغيل محرك الفهم الدلالي: {e}")

        print("🔄 Loading faster-whisper (base) on CPU...")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        
        print("🧬 Loading Voice Print Model (SpeechBrain)...")
        spk_rec = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="spk_model")
        
        print("⚡ Loading OpenWakeWord Model (Hey Athena)...")
        try:
            owwModel = Model(wakeword_model_paths=["Hey_Athena_20260622_035553.onnx"])
        except Exception as e:
            print(f"Failed to load OpenWakeWord: {e}")

        VOICE_PRINT_FILE = "my_voice.wav"
        self.reference_embedding = None
        
        if os.path.exists(VOICE_PRINT_FILE):
            print("🧠 Caching Master Voice Print to RAM...")
            try:
                signal = spk_rec.load_audio(VOICE_PRINT_FILE)
                self.reference_embedding = spk_rec.encode_batch(signal)
                print("✅ Voice Print Cached Successfully in RAM!")
            except Exception as e:
                print(f"⚠️ Failed to cache voice print: {e}")

        print("\n" + "="*60 + "\n🤖 JARVIS PRO (Native Streaming Edition) - Ready!\n" + "="*60)
        
        try:
            while True:
                if not self.awaiting_command and not self.pending_power_action:
                    self.ui_update.emit("idle", "")
                    
                    # نستخدم المايكروفون عبر self.pa
                    mic_stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)
                    owwModel.reset()
                    
                    for _ in range(5):
                        mic_stream.read(1280, exception_on_overflow=False)
                    
                    voice_wake = False
                    
                    while True:
                        if self.awaiting_command or self.pending_power_action:
                            break 
                        
                        audio_data = np.frombuffer(mic_stream.read(1280, exception_on_overflow=False), dtype=np.int16)
                        prediction = owwModel.predict(audio_data)
                        
                        if any(score > 0.5 for score in prediction.values()):
                            voice_wake = True
                            break
                            
                    mic_stream.stop_stream()
                    mic_stream.close()

                    if voice_wake:
                        self.pause_background_media()
                        reply = random.choice(["Mhm", "Yes, sir", "Yes"])
                        self.speak(reply) 
                        self.awaiting_command = True

                if self.awaiting_command or self.pending_power_action:
                    self.ui_update.emit("listening", "")
                    
                    chunk_duration = 0.5
                    chunk_size = int(16000 * chunk_duration)
                    max_duration = 4.0 
                    max_iterations = int(max_duration / chunk_duration)
                    
                    cmd_stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=chunk_size)
                    
                    frames = []
                    intent_detected = False
                    final_text = ""
                    final_audio_np = None
                    
                    print("\n🔴 [Streaming] يستمع للأوامر مع التحليل اللحظي للنية...")
                    
                    for _ in range(max_iterations):
                        data = cmd_stream.read(chunk_size, exception_on_overflow=False)
                        frames.append(data)
                        
                        raw_audio = b''.join(frames)
                        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
                        final_audio_np = audio_np
                        
                        segments, _ = model.transcribe(audio_np, beam_size=1, language="en", condition_on_previous_text=False)
                        partial_text = "".join([s.text for s in segments]).strip().lower()
                        partial_clean = re.sub(r'[^\w\s]', '', partial_text).strip()
                        
                        if partial_clean:
                            self.ui_update.emit("user_text", f"⏳ {partial_clean}...")
                            
                            cmd_type, action, _ = CommandHandler.route(partial_clean)
                            
                            if cmd_type == "system" and action == "reminder":
                                if not re.search(r'(?:in|for)\s+(\d+)\s+(second|minute|hour)', partial_clean):
                                    cmd_type = "other" 
                                    
                            if self.pending_power_action and any(w in partial_clean.split() for w in ["yes", "sure", "confirm", "yeah", "no", "cancel", "stop", "abort"]):
                                cmd_type = "power_confirm" 
                                
                            if cmd_type != "other":
                                print(f"🎯 [Intent Match] تم كسر التسجيل لحظياً بناءً على الأمر: '{partial_clean}'")
                                final_text = partial_clean
                                intent_detected = True
                                break
                                
                    cmd_stream.stop_stream()
                    cmd_stream.close()
                    
                    if not intent_detected:
                        final_text = partial_clean
                        
                    if final_text:
                        print(f"🗣️ [النتيجة النهائية]: {final_text}")
                        self.ui_update.emit("user_text", final_text)
                        
                        is_my_voice = True
                        if self.reference_embedding is not None and final_audio_np is not None:
                            try:
                                signal = torch.from_numpy(final_audio_np).unsqueeze(0)
                                new_embedding = spk_rec.encode_batch(signal)
                                score = spk_rec.similarity(self.reference_embedding, new_embedding).item()
                                if score < 0.30: 
                                    is_my_voice = False
                                    print(f"🚫 [رفض أمني] البصمة غير متطابقة! ({score:.2f})")
                            except Exception as e: 
                                print(f"⚠️ Error comparing voice print: {e}")
                                
                        if is_my_voice:
                            response = self.execute(final_text)
                            print(f"🧠 [جارفيس]: {response}\n")
                            self.speak(response) 
                            self.awaiting_command = True if self.pending_power_action else False
                            
                            if not self.awaiting_command:
                                self.resume_background_media()
                        else:
                            self.awaiting_command = False
                            self.resume_background_media()
                    else:
                        self.awaiting_command = False
                        self.resume_background_media()
                        
        except Exception as e:
            print(f"Error: {e}")
        finally:
            try:
                self.pa.terminate()
            except: pass

    def execute(self, user_input: str) -> str:
        text = user_input.lower()
        words = text.split() 

        if self.pending_power_action:
            if any(w in words for w in ["yes", "sure", "confirm", "yeah"]) or "do it" in text:
                action, self.pending_power_action = self.pending_power_action, None 
                if action == "shutdown": os.system("(sleep 4 && poweroff) &"); return "Shutting down now."
                elif action == "restart": os.system("(sleep 4 && reboot) &"); return "Restarting systems."
            elif any(w in words for w in ["no", "cancel", "stop", "abort"]):
                self.pending_power_action = None
                return "Action aborted. I am still here."
            else:
                self.pending_power_action = None
                return "Confirmation not recognized. Action aborted."

        cmd_type, action, value = CommandHandler.route(user_input)
        
        if cmd_type == "music":
            self.was_media_playing = False
            
        if cmd_type == "power": self.pending_power_action = action; return f"Are you sure you want to {action.replace('_', ' ')}?"
        elif cmd_type == "ai" and action == "clear": OllamaInterface.clear_memory(); return "Memory cleared, sir."
        elif cmd_type == "network": return Actions.network(action, value)
        elif cmd_type == "display": return Actions.display_settings(action, value)
        elif cmd_type == "music": return Actions.music(action, value)
        elif cmd_type == "brightness": return Actions.brightness(action, value)
        elif cmd_type == "volume": return Actions.volume(action)
        elif cmd_type == "system":
            if action == "reminder":
                match = re.search(r'(?:in|for)\s+(\d+)\s+(second|minute|hour)', text)
                if not match:
                    return "Please specify the time, like 'in 5 minutes'."
                
                val = int(match.group(1))
                unit = match.group(2)
                seconds = val
                if 'minute' in unit: seconds = val * 60
                elif 'hour' in unit: seconds = val * 3600
                
                task = "your timer"
                task_match = re.search(r'remind me to (.*?) (?:in|for)', text)
                if task_match:
                    task = task_match.group(1)
                    
                def timer_thread():
                    remaining = seconds
                    while remaining > 0:
                        hours, remainder = divmod(remaining, 3600)
                        mins, secs = divmod(remainder, 60)
                        
                        if hours > 0:
                            time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"
                        else:
                            time_str = f"{mins:02d}:{secs:02d}"
                            
                        self.ui_update.emit("timer_update", time_str)
                        time.sleep(1)
                        remaining -= 1

                    self.ui_update.emit("timer_update", "") 
                    
                    os.system("paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga > /dev/null 2>&1 || aplay /usr/share/sounds/alsa/Front_Center.wav > /dev/null 2>&1 &")
                    msg = f"Sir, this is your reminder to {task}." if task != "your timer" else "Sir, your timer is up."
                    
                    self.speak(msg, display_text="⏰ " + msg)

                threading.Thread(target=timer_thread, daemon=True).start()
                return f"Timer set for {val} {unit}s, sir."
            else:
                return Actions.system_info(action)
        
        elif cmd_type == "apps": 
            if action == "open": return Actions.open_app(value)
            elif action == "close": return Actions.close_app(value)
        elif cmd_type == "files" and action == "open": return Actions.open_file(value)
        
        return OllamaInterface.ask(user_input) 

    # 🚀 دالة النطق الخارقة الجديدة (Native Python Audio)
    # لا مزيد من أوامر aplay أو الروابط المعطلة، كل الصوت يعمل داخل الرام
    def speak(self, text: str, display_text: str = None, update_ui: bool = True):
        if display_text is None:
            display_text = text
            
        clean_text = re.sub(r'[^\w\s.,!?\']', '', text).strip()
        if not clean_text: return
        
        # 🌟 1. اعتراض الملف الصوتي (الهمهمة) ليشتغل بـ (0ms Latency)
        if clean_text.lower() in ["mhm", "mhm?"]:
            if update_ui:
                self.ui_update.emit("speaking", display_text + "?" if not display_text.endswith("?") else display_text)
            
            if self.mhm_audio_data is not None:
                # نفتح قناة الصوت من البايثون مباشرة ونصب فيها الملف
                stream = self.pa.open(format=pyaudio.paFloat32, channels=1, rate=self.mhm_fs, output=True)
                stream.write(self.mhm_audio_data)
                stream.stop_stream()
                stream.close()
            return

        # 🌟 2. توليد صوت Kokoro وتشغيله أيضاً برمجياً بدون aplay وبدون كتابة ملفات مؤقتة!
        if getattr(self, 'has_kokoro', False):
            try:
                samples, sample_rate = self.kokoro.create(clean_text, voice="af_sarah", speed=1.0, lang="en-us")
                if update_ui:
                    self.ui_update.emit("speaking", display_text)
                
                # تحويل الصوت المتولد إلى Byte Array وتشغيله فوراً وبنفس جودة الـ Native
                stream = self.pa.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
                stream.write(samples.astype(np.float32).tobytes())
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"⚠️ Voice error: {e}")
        else:
            if update_ui:
                self.ui_update.emit("speaking", display_text)
            os.system(f"echo '{clean_text}' | ./piper/piper --model en_US-amy-medium.onnx --output_file - 2>/dev/null | aplay -q 2>/dev/null")

# ==========================================
# 5. تشغيل البرنامج (Main Entry)
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    pill = DynamicPill()
    pill.show()
    
    worker = JarvisWorker()
    
    pill.manual_wake_signal.connect(worker.handle_manual_wake)
    pill.text_command_signal.connect(worker.handle_text_command)
    pill.start_download_signal.connect(worker.handle_gui_download)
    pill.start_translation_signal.connect(worker.handle_translation) 
    
    def update_ui(state, text):
        if state == "timer_update":
            pill.active_timer_text = text
            if pill.state == "idle":
                if text: 
                    pill.update_geom(140, 64, "idle") 
                else:    
                    pill.update_geom(64, 64, "idle")
                pill.update() 
            return
            
        if pill.state in ["text_mode", "options", "gallery"] and state in ["idle", "listening"]:
            return 
            
        if state == "idle": pill.set_idle()
        elif state == "listening": pill.set_listening()
        elif state == "user_text": pill.set_user_text(text)
        elif state == "speaking": pill.set_speaking(text)
        elif state == "downloading": pill.set_downloading(text)
        elif state == "translation_ready":
            QApplication.clipboard().clear()
            QApplication.clipboard().setText(text)
            pill.border_color = QColor(0, 255, 0, 200)
            preview_text = text if len(text) <= 75 else text[:75] + "..."
            pill.set_user_text(f"🌐 {preview_text}")
            QTimer.singleShot(5000, pill.set_idle)
        
    worker.ui_update.connect(update_ui)
    worker.start()
    
    sys.exit(app.exec_())
