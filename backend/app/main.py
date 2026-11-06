from fastapi import FastAPI
from app.core.config import settings
from app.core.database import engine, Base
from app.api.v1.videos import router as videos_router

# Create database tables automatically (for rapid prototyping before Alembic migrations)
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME)

# Register routers
app.include_router(videos_router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {"status": "online", "system": settings.PROJECT_NAME}
