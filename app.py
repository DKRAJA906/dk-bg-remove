import os

# Numerical libraries import होने से पहले thread limits
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import asyncio
import gc
import io
import logging
import re

import onnxruntime as ort

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from rembg import remove
from rembg.sessions.u2netp import U2netpSession
from starlette.concurrency import run_in_threadpool


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("background-remover")

app = FastAPI(
    title="Photo Background Remover API",
    version="1.2"
)

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://dkraja906.github.io"
).rstrip("/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "POST", "OPTIONS"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

MAX_FILE_SIZE_BYTES = 12 * 1024 * 1024

# यह decoded image के pixel count की सीमा है
MAX_INPUT_PIXELS = 6_000_000

# Output अधिकतम 768 × 768; अनुपात बना रहेगा
MAX_OUTPUT_SIZE = (768, 768)

session = None
processing_lock = asyncio.Lock()


def get_session():
    global session

    if session is None:
        logger.info("Loading u2netp model")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False

        session = U2netpSession(
            "u2netp",
            options,
            providers=["CPUExecutionProvider"]
        )

        logger.info("u2netp model loaded")

    return session


def process_image(image_bytes, background_rgb):
    resized = None
    cutout = None
    result = None

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            width, height = source.size

            if width * height > MAX_INPUT_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Photo का resolution बहुत बड़ा है। "
                        "पहले इसे अधिकतम 6 megapixels तक छोटा करें।"
                    )
                )

            source.thumbnail(
                MAX_OUTPUT_SIZE,
                Image.Resampling.LANCZOS
            )

            resized = ImageOps.exif_transpose(source)
            converted = resized.convert("RGB")

            if converted is not resized:
                resized.close()

            resized = converted

        model_session = get_session()

        logger.info(
            "Processing image: %s x %s",
            resized.width,
            resized.height
        )

        cutout = remove(
            resized,
            session=model_session,
            alpha_matting=False,
            post_process_mask=False
        )

        if background_rgb is not None:
            result = Image.new(
                "RGBA",
                cutout.size,
                (*background_rgb, 255)
            )

            with cutout.getchannel("A") as alpha:
                result.paste(cutout, (0, 0), alpha)
        else:
            result = cutout

        with io.BytesIO() as output:
            result.save(output, format="PNG")
            png_bytes = output.getvalue()

        logger.info("Image processing complete")
        return png_bytes

    finally:
        if result is not None and result is not cutout:
            result.close()

        if cutout is not None:
            cutout.close()

        if resized is not None:
            resized.close()

        gc.collect()


@app.get("/")
def home():
    return {
        "status": "running",
        "service": "Photo Background Remover",
        "model": "u2netp",
        "model_loaded": session is not None,
        "endpoint": "/remove-bg",
        "max_output_size": 768
    }


@app.head("/")
def head_home():
    return Response(status_code=200)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": session is not None
    }


@app.post("/remove-bg")
async def remove_background(
    file: UploadFile = File(...),
    bg_color: str = Form(default="")
):
    if file.content_type not in ALLOWED_TYPES:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="केवल JPG, PNG या WEBP photo चुनें।"
        )

    background_rgb = None
    clean_color = bg_color.strip().lstrip("#")

    if clean_color:
        if not re.fullmatch(r"[0-9a-fA-F]{6}", clean_color):
            await file.close()
            raise HTTPException(
                status_code=400,
                detail="bg_color में FFFFFF जैसा hex color भेजें।"
            )

        background_rgb = tuple(
            int(clean_color[i:i + 2], 16)
            for i in (0, 2, 4)
        )

    # Busy होने पर दूसरी फोटो memory में नहीं पढ़ें
    if processing_lock.locked():
        await file.close()
        raise HTTPException(
            status_code=503,
            detail=(
                "Server पर एक photo process हो रही है। "
                "थोड़ी देर बाद दोबारा कोशिश करें।"
            )
        )

    async with processing_lock:
        image_bytes = None

        try:
            image_bytes = await file.read(
                MAX_FILE_SIZE_BYTES + 1
            )

            if not image_bytes:
                raise HTTPException(
                    status_code=400,
                    detail="Image file खाली है।"
                )

            if len(image_bytes) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Photo 12 MB से बड़ी है।"
                )

            png_bytes = await run_in_threadpool(
                process_image,
                image_bytes,
                background_rgb
            )

            return Response(
                content=png_bytes,
                media_type="image/png",
                headers={
                    "Content-Disposition":
                        "attachment; filename=no-background.png"
                }
            )

        except HTTPException:
            raise

        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning
        ):
            raise HTTPException(
                status_code=400,
                detail="Photo invalid है या उसका resolution बहुत बड़ा है।"
            )

        except Exception:
            logger.exception("Background removal failed")

            raise HTTPException(
                status_code=500,
                detail=(
                    "Photo processing में समस्या आई। "
                    "Render Logs में error देखें।"
                )
            )

        finally:
            await file.close()
            image_bytes = None
            gc.collect()
