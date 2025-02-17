# from ultralytics import YOLOWorld

# # Initialize a YOLO-World model
# model = YOLOWorld("/home/ang/桌面/EXP/ALSS-YOLO-world-VOC_0.641/weights/best.pt")  # or select yolov8m/l-world.pt for different sizes

# # Execute inference with the YOLOv8s-world model on the specified image
# results = model.predict("0000000352_0000000000_0000000155.jpg")

# # Show results
# results[0].show()




from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld("/home/ang/桌面/EXP/ALSS-YOLO-world-VOC_0.641/weights/best.pt")  # or select yolov8m/l-world.pt

# Define custom classes
model.set_classes(["elephant", "animal", "African elephant"])
# model.set_classes(["skirt", "red skirt", "red dress"])
# model.set_classes(["Plantation", "Fruit trees", "plant"])


# Save the model with the defined offline vocabulary
# model.save("custom_yolov8s.pt")



results = model.predict("0000000352_0000000000_0000000155.jpg")
# results = model.predict("frame_227_jpg.jpg")

# Show results
results[0].show()
