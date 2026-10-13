import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from rembg import new_session, remove

app = FastAPI(
    title="Photo Background Remover API",
    version="1.0"
)

# अपना GitHub Pages URL यहाँ डालें
FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://YOUR_USERNAME.github.io"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Model एक बार load होगा और requests में reuse होगा
session = new_session("isnet-general-use")

ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

MAX_FILE_SIZE_MB = 12
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@app.get("/")
async def home():
    return {
        "status": "running",
        "service": "Photo Background Remover",
        "usage": {
            "endpoint": "/remove-bg",
            "method": "POST",
            "form_field": "file",
            "optional_field": "bg_color",
            "example_bg_color": "FFFFFF"
        }
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/remove-bg")
async def remove_background(
    file: UploadFile = File(...),
    bg_color: str = Form(default="")
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="केवल JPG, PNG या WEBP image upload करें।"
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image file खाली है।")

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image {MAX_FILE_SIZE_MB}MB से बड़ी है।"
        )

    try:
        # Background हटाएँ
        cutout_bytes = remove(
            image_bytes,
            session=session,
            post_process_mask=True
        )

        result_image = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")

        # Optional background color
        if bg_color:
            clean_color = bg_color.strip().lstrip("#")

            if len(clean_color) != 6:
                raise HTTPException(
                    status_code=400,
                    detail="bg_color hex format में भेजें, जैसे FFFFFF।"
                )

            try:
                red = int(clean_color[0:2], 16)
                green = int(clean_color[2:4], 16)
                blue = int(clean_color[4:6], 16)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="bg_color valid hex color नहीं है।"
                )

            background = Image.new(
                "RGBA",
                result_image.size,
                (red, green, blue, 255)
            )
            background.paste(
                result_image,
                mask=result_image.split()[3]
            )
            result_image = background

        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format="PNG", optimize=True)
        output_buffer.seek(0)

        return Response(
            content=output_buffer.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=no-background.png"
            }
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Background remove करने में समस्या आई। कृपया दूसरी image try करें।"
        )
