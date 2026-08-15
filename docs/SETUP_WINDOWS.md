# Windows Setup Guide - AutoLabeler

This guide will walk you through setting up the AutoLabeler application on Windows without Docker.

## Prerequisites

### Required Software

1. **Python 3.10 or higher**
   - Download from: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"
   - Verify: `python --version`

2. **Node.js 18 or higher**
   - Download from: https://nodejs.org/ (LTS version recommended)
   - Verify: `node --version` and `npm --version`

3. **Git**
   - Download from: https://git-scm.com/download/win
   - Verify: `git --version`

4. **CUDA Toolkit 11.8 or 12.1** (for NVIDIA GPU support)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Only needed if you have an NVIDIA GPU
   - Verify: `nvcc --version`

### Hardware Requirements

- **Minimum**: 8GB RAM, 10GB free disk space
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM, 50GB+ free disk space
- **For 50K-100K images**: 32GB+ RAM or GPU with 16GB+ VRAM recommended

## Step-by-Step Setup

### Step 1: Create Project Directory

```cmd
mkdir C:\auto-labeler
cd C:\auto-labeler
```

### Step 2: Setup Python Backend

```cmd
:: Create virtual environment
python -m venv backend\venv

:: Activate virtual environment
backend\venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install PyTorch with CUDA support (adjust for your CUDA version)
:: For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: For CUDA 12.1:
:: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: For CPU only (no GPU):
:: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

:: Install other requirements
pip install fastapi uvicorn python-multipart aiofiles pydantic pydantic-settings
pip install ultralytics segment-anything opencv-python-headless pillow numpy
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
pip install pycocotools shapely scikit-learn tqdm rich

:: Create directory structure
mkdir backend\app\api backend\app\core backend\app\models backend\app\services
mkdir backend\models backend\uploads backend\outputs backend\logs
```

### Step 3: Setup React Frontend

```cmd
:: Initialize Vite React project with TypeScript
cd frontend
npm create vite@latest . -- --template react-ts

:: Install dependencies
npm install

:: Install UI libraries
npm install @radix-ui/react-dialog @radix-ui/react-dropdown-menu @radix-ui/react-select
npm install @radix-ui/react-tabs @radix-ui/react-progress @radix-ui/react-slider
npm install @radix-ui/react-checkbox @radix-ui/react-label @radix-ui/react-toast
npm install lucide-react clsx tailwind-merge class-variance-authority
npm install axios react-dropzone recharts

:: Initialize Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### Step 4: Configure Tailwind CSS

Edit `frontend/tailwind.config.js`:

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

Edit `frontend/src/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

## Running the Application

### Terminal 1: Start Backend

```cmd
cd C:\auto-labeler\backend
venv\Scripts\activate
python -m app.main
```

Backend will start at: `http://localhost:8000`

API docs at: `http://localhost:8000/docs`

### Terminal 2: Start Frontend

```cmd
cd C:\auto-labeler\frontend
npm run dev
```

Frontend will start at: `http://localhost:5173`

### Open Browser

Navigate to: `http://localhost:5173`

## Troubleshooting

### CUDA/GPU Issues

```cmd
:: Check if PyTorch sees your GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

:: If False, reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Port Already in Use

```cmd
:: Find process using port 8000
netstat -ano | findstr :8000

:: Kill process (replace <PID> with actual process ID)
taskkill /PID <PID> /F
```

### Memory Issues

For large datasets, reduce batch size in `backend/app/core/config.py`:

```python
BATCH_SIZE = 8  # Reduce from default 16
MAX_WORKERS = 2  # Reduce parallel workers
```

### Model Download Failures

Models are auto-downloaded on first use. If download fails:

```cmd
:: Manually download models
mkdir backend\models
cd backend\models

:: Download YOLOv8
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"

:: Download SAM
python -c "from segment_anything import sam_model_registry; sam_model_registry['vit_h'](checkpoint='sam_vit_h.pth')"
```

## Updating the Application

```cmd
cd C:\auto-labeler

:: Update backend
cd backend
venv\Scripts\activate
pip install --upgrade -r requirements.txt

:: Update frontend
cd ..\frontend
npm update
```

## Next Steps

1. Read [USAGE.md](USAGE.md) for detailed usage instructions
2. Check [API_REFERENCE.md](API_REFERENCE.md) for API documentation
3. See [MODELS.md](MODELS.md) for model information and customization
