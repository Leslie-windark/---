#!/usr/bin/env python3
"""
GPU检测脚本
"""

import sys
import torch

def check_gpu():
    print("=" * 50)
    print("GPU 兼容性检测")
    print("=" * 50)
    
    # 检查PyTorch版本
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print("✅ CUDA 可用")
        
        # 获取GPU信息
        device_count = torch.cuda.device_count()
        print(f"GPU 数量: {device_count}")
        
        for i in range(device_count):
            print(f"\nGPU {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            print(f"  CUDA能力: {torch.cuda.get_device_capability(i)}")
            
            # VRAM信息
            props = torch.cuda.get_device_properties(i)
            print(f"  总VRAM: {props.total_memory / 1024**3:.1f} GB")
            print(f"  多处理器: {props.multi_processor_count}")
        
        # 测试张量计算
        print("\n🧪 性能测试...")
        try:
            a = torch.randn(10000, 10000).cuda()
            b = torch.randn(10000, 10000).cuda()
            result = torch.matmul(a, b)
            print("✅ GPU 计算测试通过")
        except Exception as e:
            print(f"❌ GPU 计算测试失败: {e}")
            
    else:
        print("❌ CUDA 不可用")
        print("\n可能的原因:")
        print("1. 没有NVIDIA GPU")
        print("2. 未安装NVIDIA驱动")
        print("3. PyTorch未安装CUDA版本")
        print("\n解决方案:")
        print("1. 检查GPU: nvidia-smi")
        print("2. 安装CUDA版本PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # 检查faster-whisper兼容性
    print("\n" + "=" * 50)
    print("faster-whisper 兼容性")
    print("=" * 50)
    
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper 已安装")
        
        # 测试模型加载
        print("测试模型加载...")
        model = WhisperModel("tiny", device="cpu", compute_type="float32")
        print("✅ 模型加载测试通过")
        
        # GPU测试
        if torch.cuda.is_available():
            print("\n测试GPU模型加载...")
            try:
                model = WhisperModel("tiny", device="cuda", compute_type="float16")
                print("✅ GPU模型加载测试通过")
            except Exception as e:
                print(f"❌ GPU模型加载失败: {e}")
                
    except ImportError:
        print("❌ faster-whisper 未安装")
        print("安装命令: pip install faster-whisper")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    check_gpu()
    input("\n按回车键退出...")