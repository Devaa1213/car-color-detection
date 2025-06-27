import cv2
import numpy as np
from tkinter import filedialog, Tk, Label, Button
from PIL import Image, ImageTk

# Load YOLOv3-tiny model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Configure model
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load Haar cascade for people detection
person_cascade = cv2.CascadeClassifier("haarcascade_fullbody (1).xml")
if person_cascade.empty():
    raise IOError("Failed to load haarcascade_fullbody.xml. Check path.")

# Function to detect cars, color and people
def detect(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # YOLO blob and forward pass
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                cx, cy, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(cx - w / 2), int(cy - h / 2)
                class_name = classes[class_id]

                if class_name in ["car", "truck", "bus", "motorbike"]:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    total_cars = 0
    blue_cars = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        car_roi = image[y:y+h, x:x+w]
        total_cars += 1

        # Color detection
        avg_color = np.mean(car_roi, axis=(0, 1))
        b, g, r = avg_color
        if b > r and b > g and b > 100:
            color = "blue"
            color_box = (0, 0, 255)
            blue_cars += 1
        else:
            color = "other"
            color_box = (255, 0, 0)

        cv2.rectangle(image, (x, y), (x + w, y + h), color_box, 2)
        cv2.putText(image, color, (x, y - 5), font, 0.5, color_box, 1)

    # Detect people
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    people = person_cascade.detectMultiScale(gray, 1.1, 4)
    people_count = len(people)

    for (x, y, w, h) in people:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Add counts on top
    cv2.putText(image, f"Total Cars: {total_cars}", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(image, f"Blue Cars: {blue_cars}", (10, 60), font, 1, (0, 0, 255), 2)
    cv2.putText(image, f"People: {people_count}", (10, 90), font, 1, (0, 255, 0), 2)

    return image

# GUI Setup
def select_image():
    path = filedialog.askopenfilename()
    if len(path) > 0:
        output = detect(path)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(output)
        imgtk = ImageTk.PhotoImage(image=img)
        panel.config(image=imgtk)
        panel.image = imgtk

root = Tk()
root.title("Accurate Car Colour Detection")

btn = Button(root, text="Upload Image", command=select_image)
btn.pack()

panel = Label(root)
panel.pack()

root.mainloop()
