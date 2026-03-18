from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import translate, stories
import uvicorn

app = FastAPI(
    title="ASL Translation API",
    description="English text → ASL gloss → 3D keypoint animation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(translate.router, prefix="/api")
app.include_router(stories.router, prefix="/api")

@app.get("/health")
def health():
    from services.inference import model_is_loaded
    from services.vocab import get_vocab_size
    return {
        "status":       "ok",
        "model_loaded": model_is_loaded(),
        "vocab_size":   get_vocab_size(),
        "output_dim":   "3D (x, y, z)",
        "sign_language":"ASL",
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)