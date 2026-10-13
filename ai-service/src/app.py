
import torch
import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="Drone Formation Identification AI Service",
    version="0.1.0",
    description="AI service for drone detection and formation identification.",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "drone-formation-ai",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
    )
