# 🏷️ AutoLabeler

**AI-powered automatic image dataset labeling platform**

AutoLabeler takes the tedious, hours-eating part of computer vision work drawing boxes and masks by hand and automates it. It combines state-of-the-art object detection (YOLO, YOLO-World) and instance/semantic segmentation (SAM, SAM 2, SAM 3) into one local pipeline, with a human-in-the-loop editor to review and fix anything the AI gets wrong.

Built by [Sahil Jatoi](https://github.com/SahilJatoi744) and [Bushra Abro](https://github.com/) now open for public use and contributions.

---

## ✨ Why AutoLabeler

Manual dataset labeling doesn't scale. AutoLabeler handles the full lifecycle from spinning up the backend, to running automated annotation and quality checks, to human review, to exporting a training-ready dataset so you spend your time on the model, not the mouse.

### Key Features

- **AI-Powered Annotation**: automatic bounding boxes and segmentation masks via YOLO and SAM
- **Human-in-the-Loop (HITL) Editor**: an interactive canvas to review and correct AI-generated labels
- **Active Learning & Quality Assurance**: flags low-confidence predictions and surfaces dataset health issues
- **Zero-Shot Detection**: Grounding DINO integration for text-prompt object detection
- **Flexible Export**: COCO JSON, YOLO, and Pascal VOC, with automatic train/val/test splitting
- **Runs Fully Local**: your data never leaves your machine; GPU acceleration supported

---

## 🎥 Demo

<!--
Add your demo video here. Drop the video file at:
  readme_docs/video.webm
Then either embed it directly (GitHub renders mp4 previews on the repo page)
or link to it, e.g.:
-->


https://github.com/user-attachments/assets/4989c2f3-5e03-4a50-9af7-94b32152a65c

[![Watch the demo](https://img.shields.io/badge/▶-Watch%20the%20demo-blue?style=for-the-badge)](readme_docs/video.mp4)

---

## 🚀 Quick Start

### Prerequisites

- Windows 10/11
- Python 3.10+
- Node.js 18+
- NVIDIA GPU with CUDA 11.8+ (optional, but strongly recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/SahilJatoi744/Auto-labeler.git
cd Auto-labeler

# Run the automated setup script
scripts\setup_windows.bat
```

Follow the prompts to select your CUDA version.

### Running the App

**Backend API:**

```bash
cd backend
venv\Scripts\activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Frontend UI:**

```bash
cd frontend
npm run dev
```

- Web UI: [http://localhost:5173](http://localhost:5173)
- API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📖 Labeling Workflow

1. **Create a Project**: set up a new workspace from the Platform menu
2. **Upload a Dataset**: drag & drop a `.zip` of your images
3. **Configure & Launch**: define your classes (e.g. `car`, `person`) and start an AI-assisted labeling job
4. **Review**: use the HITL canvas to check and correct AI-generated annotations
5. **Export**: download your dataset as COCO, YOLO, or Pascal VOC, split into train/val/test

For more detail, see [`QUICKSTART.md`](QUICKSTART.md) and [`usage.md`](usage.md).

---

## 🤖 Advanced Integrations

AutoLabeler supports modular adapters for more advanced CV workloads:

- **SAM 3** concept segmentation
- **Grounding DINO** zero-shot, text-prompt object detection

---

## 🤝 Contributing

AutoLabeler is open source and open for collaboration. Whether it's a bug fix, a new export format, a UI improvement, or a whole new adapter contributions are welcome.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Commit your changes
4. Open a pull request

If you're not sure where to start, open an issue and we can figure it out together.

---

## 👨‍💻 Authors

Built by **[Sahil Jatoi](https://github.com/SahilJatoi744)** and **Bushra Abro**.

If AutoLabeler saves you time on your next CV project, consider starring the repo ⭐ it helps others find it too.

---

## 📄 License

Add your license of choice here (MIT is a common pick for open collaboration projects).
