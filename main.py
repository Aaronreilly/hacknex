from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import shutil
import uuid
import json

from person_bag_movement import process_video


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="VisionDetect AI Backend"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SERVE OUTPUT VIDEOS
# ============================================================

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "message": "VisionDetect AI Backend is running"
    }


# ============================================================
# PROCESS VIDEO
# ============================================================

@app.post("/process-video")
async def process_uploaded_video(
    file: UploadFile = File(...)
):

    job_id = uuid.uuid4().hex

    extension = (
        Path(file.filename).suffix
        if file.filename
        else ".mp4"
    )

    input_path = (
        UPLOAD_DIR /
        f"{job_id}{extension}"
    )

    output_path = (
        OUTPUT_DIR /
        f"{job_id}_result.mp4"
    )

    # --------------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------------

    with open(
        input_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # --------------------------------------------------------
    # Run AI
    # --------------------------------------------------------

    try:

        result = process_video(
            input_path,
            output_path
        )

    except Exception as error:

        if input_path.exists():
            input_path.unlink()

        return {
            "success": False,
            "error": str(error)
        }

    # --------------------------------------------------------
    # JSON RESULT
    # --------------------------------------------------------

    result["success"] = True

    result["video_url"] = (
        f"/outputs/{output_path.name}"
    )

    result["job_id"] = job_id

    # Save JSON
    json_path = (
        OUTPUT_DIR /
        f"{job_id}_events.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    result["json_url"] = (
        f"/outputs/{json_path.name}"
    )

    return result