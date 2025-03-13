import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import predict  # Import the predict module
import database  # Import the database module
import training  # Import the training module

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Initialize database
database.init_db()

# Include routers
app.include_router(predict.router)
app.include_router(database.router, prefix="/api")  # Add prefix for database routes
app.include_router(training.router)  # Include training router

@app.on_event("startup")
async def startup_event():
    """Runs when the API server starts."""
    if predict.ml_model is None:
        predict.logger.error("⚠️ ML model failed to load")
    else:
        predict.logger.info("✅ ML model is ready")
    
    print("🚀 API Server starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Runs when the API server is shutting down."""
    print("👋 API Server shutting down...")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Render's assigned PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
