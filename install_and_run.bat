@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   视频转录工具 - 一键安装运行（GPU版）
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python
    pause
    exit /b 1
)

echo [2/4] 检查GPU支持...
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo ⚠️ PyTorch未安装，将安装CUDA版本
    set TORCH_URL=https://download.pytorch.org/whl/cu118
)

echo [3/4] 安装必要依赖...
echo   安装 torch (CUDA版本)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo   安装 faster-whisper...
pip install faster-whisper -i https://pypi.tuna.tsinghua.edu.cn/simple

echo   安装 yt-dlp...
pip install yt-dlp -i https://pypi.tuna.tsinghua.edu.cn/simple

echo   安装其他依赖...
pip install pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [4/4] 运行GPU检测...
python check_gpu.py

echo.
echo ✅ 安装完成！
echo.
echo 选择启动方式：
echo 1. 启动主程序（推荐）
echo 2. 查看GPU详情
echo 3. 退出
choice /c 123 /n /m "请输入选择: "

if errorlevel 3 goto exit
if errorlevel 2 goto check_gpu
if errorlevel 1 goto run_main

:check_gpu
python check_gpu.py
goto run_main

:run_main
echo 启动转录工具...
python working_transcriber.py

:exit
pause