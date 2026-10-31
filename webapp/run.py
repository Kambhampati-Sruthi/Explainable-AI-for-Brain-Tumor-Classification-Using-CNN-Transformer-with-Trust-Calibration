import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # CRITICAL: host must be "0.0.0.0" so Render's proxy can route traffic to your container
    uvicorn.run("app:app", host="0.0.0.0", port=port)