from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from rembg import remove

app = FastAPI(title="Background Remover API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sirf image file upload karein")

    try:
        image_bytes = await file.read()
        output_bytes = remove(image_bytes)

        return Response(
            content=output_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=no-background.png"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
