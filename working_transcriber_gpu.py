#!/usr/bin/env python3
"""
视频转录工具 - GPU加速优化版（完整单文件）
支持本地视频 + 在线视频（YouTube/Bilibili） + Cookie 登录
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import os
import sys
from pathlib import Path
from datetime import datetime
import torch
import tempfile
import shutil


class GPUTranscriber:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("视频转录工具 - GPU加速版")
        self.window.geometry("850x700")
        
        self.is_running = False
        
        # GPU状态
        self.has_gpu = torch.cuda.is_available()
        self.gpu_info = self.get_gpu_info()
        
        # 初始化变量
        self.url_var = tk.StringVar()
        self.file_path = tk.StringVar()
        self.cookie_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.home() / "Desktop" / "Transcripts"))
        self.model_var = tk.StringVar(value="small")
        self.language_var = tk.StringVar(value="auto")
        self.vad_filter = tk.BooleanVar(value=True)
        self.save_srt = tk.BooleanVar(value=True)
        self.save_txt = tk.BooleanVar(value=True)
        self.compute_mode = tk.StringVar(value="auto")
        self.batch_size = tk.StringVar(value="auto")
        
        self.setup_ui()
    
    def get_gpu_info(self):
        """获取GPU信息"""
        if not self.has_gpu:
            return None
        try:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                "name": gpu_name,
                "vram": vram_gb,
                "cores": torch.cuda.get_device_properties(0).multi_processor_count
            }
        except:
            return None
    
    def setup_ui(self):
        """设置用户界面"""
        title_text = "🎬 视频转录工具"
        if self.has_gpu and self.gpu_info:
            title_text += f" | 🎮 {self.gpu_info['name']} ({self.gpu_info['vram']:.1f}GB)"
        
        title_frame = tk.Frame(self.window, bg='#2c3e50')
        title_frame.pack(fill='x')
        tk.Label(title_frame, text=title_text, 
                font=("Microsoft YaHei", 16, "bold"),
                fg='white', bg='#2c3e50').pack(pady=15)
        
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        left_panel = tk.Frame(main_frame)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        self.setup_file_section(left_panel)
        self.setup_settings_section(left_panel)
        self.setup_output_section(left_panel)
        self.setup_gpu_control(left_panel)
        
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side='right', fill='both', expand=True)
        self.setup_performance_section(right_panel)
        self.setup_progress_section(right_panel)
        self.setup_log_section(right_panel)
        
        self.setup_control_buttons()
    
    def setup_file_section(self, parent):
        frame = ttk.LabelFrame(parent, text="📁 视频来源", padding=10)
        frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(frame, text="在线视频链接 (YouTube/Bilibili等):", font=("Microsoft YaHei", 9)).pack(anchor='w')
        url_entry = ttk.Entry(frame, textvariable=self.url_var, width=30)
        url_entry.pack(fill='x', pady=5)
        
        tk.Label(frame, text="或选择本地视频文件:", font=("Microsoft YaHei", 9)).pack(anchor='w', pady=(5, 0))
        file_frame = tk.Frame(frame)
        file_frame.pack(fill='x', pady=5)
        ttk.Entry(file_frame, textvariable=self.file_path, state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(file_frame, text="浏览", command=self.select_video_file, width=8).pack(side='right', padx=(5, 0))
        
        cookie_frame = tk.Frame(frame)
        cookie_frame.pack(fill='x', pady=5)
        tk.Label(cookie_frame, text="Cookie 文件 (用于会员/登录视频):", font=("Microsoft YaHei", 9)).pack(anchor='w')
        cookie_subframe = tk.Frame(cookie_frame)
        cookie_subframe.pack(fill='x', pady=2)
        ttk.Entry(cookie_subframe, textvariable=self.cookie_path, state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(cookie_subframe, text="选择", command=self.select_cookie_file, width=8).pack(side='right', padx=(5, 0))
    
    def setup_settings_section(self, parent):
        frame = ttk.LabelFrame(parent, text="⚙️ 转录设置", padding=10)
        frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(frame, text="模型大小:", font=("Microsoft YaHei", 9)).grid(row=0, column=0, sticky='w')
        model_combo = ttk.Combobox(frame, textvariable=self.model_var,
                                  values=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                                  state="readonly", width=12)
        model_combo.grid(row=0, column=1, padx=5, sticky='w')
        
        tk.Label(frame, text="语言:", font=("Microsoft YaHei", 9)).grid(row=1, column=0, sticky='w', pady=5)
        lang_combo = ttk.Combobox(frame, textvariable=self.language_var,
                                 values=["auto", "zh", "en", "ja", "ko", "fr", "de", "es"],
                                 state="readonly", width=12)
        lang_combo.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Checkbutton(frame, text="启用 VAD（静音检测）", variable=self.vad_filter).grid(row=2, column=0, columnspan=2, sticky='w', pady=5)
    
    def setup_output_section(self, parent):
        frame = ttk.LabelFrame(parent, text="📤 输出设置", padding=10)
        frame.pack(fill='x', pady=(0, 15))
        
        dir_frame = tk.Frame(frame)
        dir_frame.pack(fill='x')
        ttk.Entry(dir_frame, textvariable=self.output_dir, state='readonly').pack(side='left', fill='x', expand=True)
        ttk.Button(dir_frame, text="选择目录", command=self.select_output_dir, width=10).pack(side='right', padx=(5, 0))
        
        ttk.Checkbutton(frame, text="保存 .srt 字幕文件", variable=self.save_srt).pack(anchor='w', pady=2)
        ttk.Checkbutton(frame, text="保存 .txt 文本文件", variable=self.save_txt).pack(anchor='w', pady=2)
    
    def setup_gpu_control(self, parent):
        frame = ttk.LabelFrame(parent, text="🎮 GPU加速控制", padding=10)
        frame.pack(fill='x', pady=(0, 15))
        
        if self.has_gpu and self.gpu_info:
            info_text = f"✅ {self.gpu_info['name']}\nVRAM: {self.gpu_info['vram']:.1f}GB | 核心: {self.gpu_info['cores']}"
            status_color = "green"
        else:
            info_text = "❌ 未检测到NVIDIA GPU\n使用CPU模式"
            status_color = "red"
        
        tk.Label(frame, text=info_text, fg=status_color, justify='left', font=("Microsoft YaHei", 9)).pack(anchor='w', pady=5)
        
        tk.Label(frame, text="计算模式:", font=("Microsoft YaHei", 9)).pack(anchor='w', pady=(10,5))
        modes = [
            ("🚀 自动优化 (推荐)", "auto"),
            ("⚡ 极速模式 (float16)", "float16"),
            ("🎯 高精度模式 (float32)", "float32"),
            ("💾 低显存模式 (int8)", "int8"),
            ("🖥️ 强制CPU模式", "cpu")
        ]
        for text, value in modes:
            ttk.Radiobutton(frame, text=text, variable=self.compute_mode, value=value).pack(anchor='w', padx=15, pady=2)
        
        batch_frame = tk.Frame(frame)
        batch_frame.pack(fill='x', pady=10)
        tk.Label(batch_frame, text="批处理大小:").pack(side='left')
        ttk.Combobox(batch_frame, textvariable=self.batch_size, 
                    values=["auto", "1", "2", "4", "8", "16"], 
                    width=8, state="readonly").pack(side='left', padx=5)
        ttk.Button(batch_frame, text="测试性能", command=self.run_gpu_benchmark, width=10).pack(side='right')
    
    def setup_performance_section(self, parent):
        frame = ttk.LabelFrame(parent, text="📈 性能监控", padding=10)
        frame.pack(fill='x', pady=(0, 15))
        
        metrics_frame = tk.Frame(frame)
        metrics_frame.pack(fill='x')
        self.fps_var = tk.StringVar(value="实时速度: -- 字/秒")
        tk.Label(metrics_frame, textvariable=self.fps_var, font=("Microsoft YaHei", 10)).pack(side='left')
        self.gpu_usage_var = tk.StringVar(value="GPU: --%")
        tk.Label(metrics_frame, textvariable=self.gpu_usage_var, font=("Microsoft YaHei", 10)).pack(side='right')
        
        if self.has_gpu:
            temp_frame = tk.Frame(frame)
            temp_frame.pack(fill='x', pady=5)
            self.gpu_temp_var = tk.StringVar(value="温度: --°C")
            tk.Label(temp_frame, textvariable=self.gpu_temp_var, font=("Microsoft YaHei", 9)).pack(side='left')
            self.vram_var = tk.StringVar(value="VRAM: --/-- GB")
            tk.Label(temp_frame, textvariable=self.vram_var, font=("Microsoft YaHei", 9)).pack(side='right')
    
    def setup_progress_section(self, parent):
        frame = ttk.LabelFrame(parent, text="⏳ 转录进度", padding=10)
        frame.pack(fill='x', pady=(0, 15))
        self.progress = ttk.Progressbar(frame, mode='determinate')
        self.progress.pack(fill='x', pady=5)
        self.progress_label = tk.StringVar(value="就绪")
        tk.Label(frame, textvariable=self.progress_label, font=("Microsoft YaHei", 9)).pack(anchor='w')
    
    def setup_log_section(self, parent):
        frame = ttk.LabelFrame(parent, text="📋 日志", padding=10)
        frame.pack(fill='both', expand=True)
        self.log_text = scrolledtext.ScrolledText(frame, height=8, state='disabled', font=("Consolas", 9))
        self.log_text.pack(fill='both', expand=True)
    
    def setup_control_buttons(self):
        button_frame = tk.Frame(self.window)
        button_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        if self.has_gpu:
            ttk.Button(button_frame, text="⚡ GPU优化设置", command=self.open_gpu_settings).pack(side='left', padx=5)
        
        self.start_btn = ttk.Button(button_frame, text="🚀 开始转录 (GPU加速)", command=self.start_transcription)
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹️ 停止", command=self.stop_transcription, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="📂 打开输出目录", command=self.open_output_dir).pack(side='right', padx=5)
        ttk.Button(button_frame, text="📊 性能报告", command=self.generate_performance_report).pack(side='right', padx=5)
    
    # --- 辅助方法 ---
    def select_video_file(self):
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频", "*.mp4 *.mkv *.avi *.mov *.flv *.webm"), ("所有文件", "*.*")]
        )
        if path:
            self.file_path.set(path)

    def select_cookie_file(self):
        path = filedialog.askopenfilename(
            title="选择 cookies.txt",
            filetypes=[("Cookies", "cookies.txt"), ("所有文件", "*.*")]
        )
        if path:
            self.cookie_path.set(path)

    def select_output_dir(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_dir.set(path)

    def open_output_dir(self):
        output = self.output_dir.get()
        if os.path.exists(output):
            os.startfile(output) if sys.platform == "win32" else os.system(f'open "{output}"')
        else:
            messagebox.showinfo("提示", "输出目录不存在")

    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state='normal')
        self.log_text.insert('end', f"[{timestamp}] {message}\n")
        self.log_text.config(state='disabled')
        self.log_text.see('end')

    # --- 核心功能 ---
    def download_audio_from_url(self, url, temp_dir):
        try:
            import yt_dlp
            ydl_opts = {
                'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
                'noprogress': True,
            }
            cookie_file = self.cookie_path.get().strip()
            if cookie_file and os.path.isfile(cookie_file):
                ydl_opts['cookiefile'] = cookie_file
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return os.path.splitext(ydl.prepare_filename(info))[0] + ".mp3"
        except Exception as e:
            self.log(f"下载失败: {str(e)}", "ERROR")
            return None

    def load_model(self):
        try:
            from faster_whisper import WhisperModel
            model_size = self.model_var.get()
            self.log(f"加载 {model_size} 模型...")
            
            if self.compute_mode.get() == "cpu" or not self.has_gpu:
                device = "cpu"
                compute_type = "float32"
                self.log("使用CPU模式", "INFO")
            else:
                device = "cuda"
                compute_mode = self.compute_mode.get()
                if compute_mode == "auto":
                    compute_type = "float16" if self.gpu_info['vram'] >= 4 else "int8"
                elif compute_mode == "float16":
                    compute_type = "float16"
                elif compute_mode == "int8":
                    compute_type = "int8"
                else:
                    compute_type = "float32"
                self.log(f"使用GPU加速 ({compute_type}精度)", "SUCCESS")
            
            return WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            self.log(f"❌ 模型加载失败: {e}", "ERROR")
            return None

    def start_transcription(self):
        if self.is_running:
            return
        url = self.url_var.get().strip()
        local_file = self.file_path.get().strip()
        if not url and not local_file:
            messagebox.showwarning("输入错误", "请提供视频链接或选择本地文件")
            return
        
        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress['value'] = 0
        self.progress_label.set("准备中...")
        
        thread = threading.Thread(target=self.transcribe_worker, args=(url, local_file))
        thread.daemon = True
        thread.start()

    def stop_transcription(self):
        self.is_running = False
        self.log("用户请求停止...", "WARNING")

    def transcribe_worker(self, url, local_file):
        temp_dir = None
        input_file = None
        try:
            if url:
                temp_dir = tempfile.mkdtemp()
                self.log("正在下载音频...", "INFO")
                input_file = self.download_audio_from_url(url, temp_dir)
                if not input_file or not os.path.exists(input_file):
                    raise Exception("音频下载失败")
            else:
                input_file = local_file
            
            model = self.load_model()
            if not model:
                raise Exception("模型加载失败")
            
            self.log("开始转录...", "INFO")
            segments, info = model.transcribe(
                input_file,
                beam_size=5,
                language=None if self.language_var.get() == "auto" else self.language_var.get(),
                vad_filter=self.vad_filter.get(),
                task="transcribe"
            )
            
            results = []
            for segment in segments:
                if not self.is_running:
                    break
                results.append(segment)
                self.log(f"[{segment.start:.1f}s] {segment.text}", "RESULT")
                self.window.after(0, lambda s=segment: self.progress_label.set(f"已转录: {s.end:.1f}秒"))
            
            if self.is_running:
                self.save_transcript(results)
                self.log("✅ 转录完成！", "SUCCESS")
            else:
                self.log("⏹️ 转录已停止", "WARNING")
                
        except Exception as e:
            self.log(f"❌ 转录出错: {e}", "ERROR")
        finally:
            self.is_running = False
            self.window.after(0, lambda: self.start_btn.config(state='normal'))
            self.window.after(0, lambda: self.stop_btn.config(state='disabled'))
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def save_transcript(self, segments):
        output_dir = self.output_dir.get()
        os.makedirs(output_dir, exist_ok=True)
        base_name = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.save_txt.get():
            txt_path = os.path.join(output_dir, base_name + ".txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                for seg in segments:
                    f.write(seg.text + "\n")
        
        if self.save_srt.get():
            srt_path = os.path.join(output_dir, base_name + ".srt")
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{self._format_time(seg.start)} --> {self._format_time(seg.end)}\n")
                    f.write(f"{seg.text}\n\n")
        
        self.log(f"💾 已保存到: {output_dir}", "SUCCESS")

    def _format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # --- GPU相关（简化版）---
    def run_gpu_benchmark(self):
        if not self.has_gpu:
            messagebox.showinfo("提示", "未检测到GPU，无法运行性能测试")
            return
        thread = threading.Thread(target=self._benchmark_thread)
        thread.daemon = True
        thread.start()

    def _benchmark_thread(self):
        self.log("🧪 GPU性能测试暂未实现（跳过）", "INFO")

    def open_gpu_settings(self):
        settings = tk.Toplevel(self.window)
        settings.title("GPU优化设置")
        settings.geometry("500x300")
        advice = """
根据你的GPU配置，推荐设置：
1. VRAM ≥ 8GB: float16 + medium
2. VRAM 4-8GB: float16 + small  
3. VRAM 2-4GB: int8 + tiny/small
4. VRAM < 2GB: CPU 模式
        """
        tk.Label(settings, text=advice, justify='left', font=("Microsoft YaHei", 10)).pack(pady=20)
        ttk.Button(settings, text="关闭", command=settings.destroy).pack()

    def generate_performance_report(self):
        self.log("📊 性能报告功能暂未实现", "INFO")

    def run(self):
        style = ttk.Style()
        style.configure("TButton", font=("Microsoft YaHei", 9))
        self.window.mainloop()


def main():
    print("="*50)
    print("视频转录工具 - GPU加速版")
    print("="*50)
    
    if torch.cuda.is_available():
        print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️  未检测到NVIDIA GPU，将使用CPU模式")
    
    required = ['faster_whisper', 'yt_dlp']
    missing = []
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print("\n❌ 缺少必要依赖，请运行：")
        for m in missing:
            print(f"  pip install {m.replace('_', '-')}")
        input("\n按回车退出...")
        return
    
    app = GPUTranscriber()
    app.run()


if __name__ == "__main__":
    main()