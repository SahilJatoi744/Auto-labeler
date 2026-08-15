# AutoLabeler - Quick Start Guide

## Prerequisites

- Windows 10/11
- Python 3.10 or higher
- Node.js 18 or higher
- NVIDIA GPU with CUDA 11.8+ (optional but recommended)

## Installation

### Option 1: Automated Setup (Recommended)

1. **Download and extract the project** to `C:\auto-labeler`

2. **Run the setup script:**
   ```cmd
   cd C:\auto-labeler
   scripts\setup_windows.bat
   ```

3. **Follow the prompts** to select your CUDA version

### Option 2: Manual Setup

#### Step 1: Backend Setup

```cmd
cd C:\auto-labeler\backend

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install PyTorch (select one based on your CUDA version)
:: For CUDA 11.8:
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

:: For CUDA 12.1:
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

:: For CPU only:
pip install torch==2.1.2 torchvision==2.1.2 --index-url https://download.pytorch.org/whl/cpu

:: Install other dependencies
pip install fastapi uvicorn python-multipart aiofiles websockets
pip install pydantic pydantic-settings
pip install ultralytics opencv-python-headless pillow
pip install numpy pandas scikit-learn scikit-image
pip install pycocotools shapely tqdm rich pyyaml imagehash psutil
```

#### Step 2: Frontend Setup

```cmd
cd C:\auto-labeler\frontend

:: Install dependencies
npm install

:: Install additional packages
npm install axios react-dropzone react-router-dom
```

## Running the Application

### Terminal 1: Start Backend

```cmd
cd backend
venv\Scripts\activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> [!NOTE]
> The backend now supports asynchronous model downloading. On first run, it will automatically start downloading YOLO and SAM weights.

### Terminal 2: Start Frontend

```cmd
cd frontend
npm run dev
```

### Accessing the UI

- **Frontend**: [http://localhost:5173](http://localhost:5173)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

## First Time Setup Checklist

- [x] Dashboard shows **"AI Model Readiness"** widget.
- [x] YOLO and SAM models are status "Ready" (or "Downloading").
- [x] "System Healthy" status is visible.
- [x] Can upload a ZIP dataset in the "Upload" page.
- [x] Can create a job with your specific classes.

## Automated Verification

If you want to verify the entire pipeline (Upload -> Label -> Export) without using the UI, run the E2E test script:

```cmd
:: From the project root
.\backend\venv\Scripts\python.exe scripts\e2e_api_test.py
```

## Step-by-Step Labeling Flow

1. **Check Readiness**: Open [Dashboard](http://localhost:5173) and wait for AI Models to show "Ready".
2. **Upload**: Go to "Upload Dataset", drag & drop a ZIP file of images.
3. **Configure**: Go to "Labeling Job", select your dataset, and click "Add Class" for each label you need.
4. **Run**: Click "Create Labeling Job" and switch back to the Dashboard to monitor progress.
5. **Export**: Once complete, go to "Export", select your job, and choose "COCO" or "YOLO" format.

## Common Issues

### "ModuleNotFoundError"

```cmd
cd backend
venv\Scripts\activate
pip install -r requirements.txt
```

### "CUDA out of memory"

Edit `backend/app/core/config.py`:
```python
BATCH_SIZE = 8  # Reduce from 16
```

### "Port already in use"

Find and kill process using port 8000:
```cmd
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### "npm install fails"

```cmd
cd frontend
del node_modules
del package-lock.json
npm install
```

## Next Steps

1. Read the full [Usage Guide](docs/USAGE.md)
2. Review the [Architecture](docs/ARCHITECTURE.md)
3. Explore the [API Documentation](http://localhost:8000/docs)

## Support

- Check logs: `backend/logs/autolabeler.log`
- Review system status on Dashboard
- Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
