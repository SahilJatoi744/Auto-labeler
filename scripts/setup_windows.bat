@echo off
chcp 65001 >nul
echo ============================================
echo    AutoLabeler - Windows Setup Script
echo ============================================
echo.

REM Check if running as administrator (optional but recommended)
echo Checking permissions...
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running with administrator privileges.
) else (
    echo Running without administrator privileges.
    echo Some features may require admin rights.
)
echo.

REM Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
python --version
echo.

REM Check Node.js
echo [2/6] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)
node --version
echo.

REM Check Git
echo [3/6] Checking Git installation...
git --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Git is not installed
    echo Git is recommended for cloning repositories
)
echo.

REM Create project directory structure
echo [4/6] Creating directory structure...
if not exist "backend\app\api" mkdir backend\app\api
if not exist "backend\app\core" mkdir backend\app\core
if not exist "backend\app\models" mkdir backend\app\models
if not exist "backend\app\services" mkdir backend\app\services
if not exist "backend\models" mkdir backend\models
if not exist "backend\uploads" mkdir backend\uploads
if not exist "backend\outputs" mkdir backend\outputs
if not exist "backend\logs" mkdir backend\logs
if not exist "frontend\src" mkdir frontend\src
if not exist "docs" mkdir docs
echo Directory structure created.
echo.

REM Setup Python virtual environment
echo [5/6] Setting up Python backend...
cd backend

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch (this may take a while)...
echo Select your CUDA version:
echo 1. CUDA 11.8 (recommended for most GPUs)
echo 2. CUDA 12.1 (for newer GPUs)
echo 3. CPU only (no GPU)
set /p cuda_choice="Enter choice (1-3): "

if "%cuda_choice%"=="1" (
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
) else if "%cuda_choice%"=="2" (
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
) else (
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
)

echo Installing FastAPI and web server dependencies...
pip install fastapi==0.109.0 uvicorn[standard]==0.27.0 python-multipart==0.0.6 aiofiles==23.2.1 websockets==12.0

echo Installing data validation and settings...
pip install pydantic==2.5.3 pydantic-settings==2.1.0

echo Installing computer vision libraries...
pip install ultralytics==8.0.239 opencv-python-headless==4.9.0.80 pillow==10.2.0

echo Installing data processing libraries...
pip install numpy==1.26.3 pandas==2.1.4 scikit-learn==1.3.2 scikit-image==0.22.0

echo Installing COCO and geometry libraries...
pip install pycocotools==2.0.7 shapely==2.0.2

echo Installing utility libraries...
pip install tqdm==4.66.1 rich==13.7.0 python-json-logger==2.0.7 pyyaml==6.0.1 imagehash==4.3.1 psutil==5.9.6

echo Installing segment-anything (SAM)...
pip install git+https://github.com/facebookresearch/segment-anything.git

cd ..
echo Backend setup complete!
echo.

REM Setup React frontend
echo [6/6] Setting up React frontend...
cd frontend

if not exist "package.json" (
    echo Initializing React project with Vite...
    call npm create vite@latest . -- --template react-ts
    
    echo Installing dependencies...
    call npm install
    
    echo Installing UI libraries...
    call npm install @radix-ui/react-dialog @radix-ui/react-dropdown-menu @radix-ui/react-select @radix-ui/react-tabs @radix-ui/react-progress @radix-ui/react-slider @radix-ui/react-checkbox @radix-ui/react-label @radix-ui/react-toast
    
    echo Installing icon and styling libraries...
    call npm install lucide-react clsx tailwind-merge class-variance-authority
    
    echo Installing utility libraries...
    call npm install axios react-dropzone recharts
    
    echo Installing Tailwind CSS...
    call npm install -D tailwindcss postcss autoprefixer
    call npx tailwindcss init -p
    
    echo Creating Tailwind config...
    (
        echo /** @type {import('tailwindcss').Config} */
        echo export default {
        echo   content: [
        echo     "./index.html",
        echo     "./src/**/*.{js,ts,jsx,tsx}",
        echo   ],
        echo   theme: {
        echo     extend: {},
        echo   },
        echo   plugins: [],
        echo }
    ) > tailwind.config.js
    
    echo Creating CSS file...
    (
        echo @tailwind base;
        echo @tailwind components;
        echo @tailwind utilities;
    ) > src/index.css
) else (
    echo Frontend already initialized.
    call npm install
)

cd ..
echo Frontend setup complete!
echo.

REM Create startup scripts
echo Creating startup scripts...
(
    echo @echo off
    echo echo Starting AutoLabeler Backend...
    echo cd backend
    echo call venv\Scripts\activate
    echo python -m app.main
    echo pause
) > start_backend.bat

(
    echo @echo off
    echo echo Starting AutoLabeler Frontend...
    echo cd frontend
    echo npm run dev
    echo pause
) > start_frontend.bat

(
    echo @echo off
    echo echo ============================================
    echo echo    AutoLabeler - Quick Start
echo ============================================
    echo echo.
    echo echo 1. Start the Backend (Terminal 1):
    echo    start_backend.bat
echo.
    echo echo 2. Start the Frontend (Terminal 2):
    echo    start_frontend.bat
echo.
    echo echo 3. Open browser to: http://localhost:5173
echo.
    echo echo API Documentation: http://localhost:8000/docs
echo.
    echo pause
) > QUICKSTART.bat

echo.
echo ============================================
echo    Setup Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Run start_backend.bat to start the backend
echo 2. Run start_frontend.bat to start the frontend
echo 3. Open http://localhost:5173 in your browser
echo.
echo Or simply run QUICKSTART.bat for instructions.
echo.
pause
