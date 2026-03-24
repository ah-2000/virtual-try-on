# Virtual Try-On (VTO) MVP

An AI-powered virtual try-on application that lets users see how clothes look on them using the FASHN VTO v1.5 diffusion model.

---

## Project Structure

```
vto-mvp/
├── backend/                  # Python FastAPI server
│   ├── main.py               # API server (main entry point)
│   ├── tryon.py              # Standalone try-on script
│   ├── extract_garments.py   # Garment extraction utility
│   ├── requirements.txt      # Python dependencies
│   ├── fashnvton/            # FASHN VTO v1.5 model
│   │   ├── src/              # Model source code
│   │   └── weights/          # Model weights (.safetensors, .pth, .onnx)
│   ├── data/
│   │   └── dresses/          # Extracted garment images
│   ├── notebooks/
│   │   └── vto_backend.ipynb # Jupyter experiments
│   └── storage/              # Generated results (auto-created)
│
├── frontend/                 # Next.js web app
│   ├── src/
│   │   ├── app/              # Next.js pages
│   │   └── components/       # React components
│   └── public/garments/      # Garment images served to frontend
│
└── assets/                   # Sample images for testing
```

---

## Prerequisites

- **Python 3.10**
- **Node.js 18+** and npm
- **Git**

---

## Backend Setup

### 1. Create and activate virtual environment

```bash
cd backend
python3.10 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install fashnvton package

```bash
pip install -e fashnvton/
```

### 4. Verify weights are in place

Make sure the following files exist inside `fashnvton/weights/`:

```
fashnvton/weights/
├── model.safetensors
├── RealESRGAN_x4plus.pth
└── dwpose/
    ├── yolox_l.onnx
    └── dw-ll_ucoco_384.onnx
```

> If weights are missing, run: `python fashnvton/scripts/download_weights.py`

### 5. Run the backend

```bash
python main.py
```

Backend will start at: **http://localhost:8000**

To verify it's running, open: http://localhost:8000/health

---

## Frontend Setup

### 1. Install dependencies

```bash
cd frontend
npm install
```

### 2. Run the frontend

```bash
npm run dev
```

Frontend will start at: **http://localhost:3000**

---

## Running Both Together

Open two terminal windows:

**Terminal 1 — Backend:**
```bash
cd vto-mvp/backend
source venv/bin/activate
python main.py
```

**Terminal 2 — Frontend:**
```bash
cd vto-mvp/frontend
npm run dev
```

Then open **http://localhost:3000** in your browser.

---

## How to Use

1. Open the app at `http://localhost:3000`
2. Browse the garment gallery and select a clothing item
3. Upload a photo of yourself **or** use your webcam to take one
4. Click **Try On**
5. Wait for the AI to process (~30–60 seconds on CPU)
6. View and download your try-on result

### Supported Garment Categories
| Category | Description |
|----------|-------------|
| `tops` | T-shirts, shirts, jackets |
| `bottoms` | Pants, skirts |
| `one-pieces` | Dresses, jumpsuits |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check if server and model are ready |
| `POST` | `/tryon` | Run virtual try-on inference |
| `GET` | `/results/{filename}` | Retrieve a generated result image |

### `/tryon` Request (multipart/form-data)

| Field | Type | Description |
|-------|------|-------------|
| `person_image` | File | Photo of the person |
| `garment_image` | File | Photo of the garment |
| `category` | String | `tops`, `bottoms`, or `one-pieces` |

---

## Notes

- First startup takes longer as models load into memory
- CPU inference takes ~30–60 seconds per image; GPU is significantly faster
- Results are saved in `backend/storage/vto_results/`
- The model runs on CPU by default; CUDA GPU is used automatically if available
