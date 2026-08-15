# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

import torch
import sys
import os

def check_gpu():
    print("--- AutoLabeler GPU/Environment Check ---")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {'YES' if gpu_available else 'NO'}")
    
    if gpu_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Test allocation
        try:
            x = torch.randn(1, 3, 224, 224).cuda()
            print("GPU Memory Allocation Test: SUCCESS")
        except Exception as e:
            print(f"GPU Memory Allocation Test: FAILED ({e})")
    else:
        print("[! WARNING !] No GPU detected. AI labeling will be slow on CPU.")
        print("Note: If you have an NVIDIA GPU, ensure CUDA Toolkit and Drivers are installed.")

    print("\n--- Model Directory Check ---")
    models_dir = os.path.join(os.getcwd(), 'backend', 'models')
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        print(f"Models directory found: {models_dir}")
        print(f"Files found: {files}")
    else:
        print(f"[!] Models directory NOT found at {models_dir}")

if __name__ == "__main__":
    check_gpu()
