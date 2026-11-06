from app.models.media import Camera, Video
from app.core.database import Base

print("Camera table:", Camera.__tablename__)
print("Video table:", Video.__tablename__)

print("\nRegistered tables:")

for table in Base.metadata.tables:
    print(table)
