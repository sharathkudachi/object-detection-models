import sys
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Starting AI Military Surveillance & Convoy Analysis Platform on http://{host}:{port} ...")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=True)
