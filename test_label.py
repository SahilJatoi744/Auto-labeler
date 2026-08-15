
import asyncio
import os
import cv2
import numpy as np

# Create dummy image
img = np.zeros((640, 640, 3), dtype=np.uint8)
cv2.imwrite('backend/test_img.jpg', img)

from backend.app.services.model_manager import get_model_manager
manager = get_model_manager()
manager.load_yolov26('yolov8s-world.pt', for_segmentation=False)
model = manager.models['yolov26']
res = model('backend/test_img.jpg')
print(res)

