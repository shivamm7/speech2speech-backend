import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "speech_fastapi_test:app",
        host="0.0.0.0",
        port=8000,
        workers=100,
        log_level="info",
    )