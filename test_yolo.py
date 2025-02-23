# from ultralytics import YOLO
# import cv2
# import torch
# import clip
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # Load YOLOv8 model (pre-trained)
# model = YOLO("yolov8n.pt")

# # Load OpenAI CLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, preprocess = clip.load("ViT-B/32", device=device)

# breakpoint()
# def classify_with_clip(image_crop):
#     """
#     Classifies an image crop using CLIP to check if it is a massage ball.
#     """
#     image = preprocess(Image.fromarray(image_crop)).unsqueeze(0).to(device)
#     text_inputs = clip.tokenize(["a massage ball", "a random object"]).to(device)

#     with torch.no_grad():
#         image_features = clip_model.encode_image(image)
#         text_features = clip_model.encode_text(text_inputs)

#     similarity = (image_features @ text_features.T).softmax(dim=-1)
#     best_match = similarity.argmax().item()
#     return ["massage ball", "not a ball"][best_match], float(similarity[0, best_match])

# def detect_massage_ball(image_path):
#     """
#     Uses YOLO to detect all objects, then filters for massage balls using CLIP.
#     """
#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error: Could not load image.")
#         return

#     # Run YOLO inference
#     results = model(image)

#     detected_balls = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box

#             # Crop detected object
#             cropped_object = image[y1:y2, x1:x2]

#             # Classify using CLIP
#             label, confidence = classify_with_clip(cropped_object)
#             if label == "massage ball":
#                 detected_balls.append((x1, y1, x2, y2, confidence))

#                 # Draw bounding box
#                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(image, f"Massage Ball {confidence:.2f}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Convert BGR to RGB for displaying in Matplotlib
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Show the image
#     plt.figure(figsize=(10, 6))
#     plt.imshow(image_rgb)
#     plt.axis("off")
#     plt.title("Massage Ball Detection")
#     plt.show()

#     return detected_balls

# # Test with an image
# detect_massage_ball("massage_ball.jpg")  # Replace with your image path

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

print("finish import")
# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # Use 'yolov8n.pt' (nano), 'yolov8s.pt' (small), or 'yolov8m.pt' (medium) for better accuracy

print("finish load model")
def test_yolo_on_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image. Check the path.")
        return

    # Run YOLO inference
    print("before inference")
    results = model(image)
    print("end inference")

    # Process results
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get class ID
            confidence = float(box.conf[0])  # Get confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            # Check if detected object is a ball (YOLO uses "sports ball" for most balls)
            if model.names[class_id] == "sports ball":
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Ball {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert BGR to RGB for displaying in Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Show the image
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("YOLO Ball Detection")
    plt.show()

# Test with an image
test_yolo_on_image("massage_ball.jpg")  # Change "test.jpg" to your actual image path
