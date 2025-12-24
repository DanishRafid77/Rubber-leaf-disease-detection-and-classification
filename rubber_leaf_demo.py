import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms

# =========================
# CONFIGURATION
# =========================
class_names = ['Anthracnose', 'Dry_Leaf', 'Healthy', 'Leaf_Spot']
MODEL_PATH = "rubber_leaf_disease_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return image, class_names[predicted.item()], confidence.item() * 100

# =========================
# GUI FUNCTION
# =========================
def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )

    if not file_path:
        return

    image, label, confidence = predict_image(file_path)

    # Display image
    display_img = image.resize((300, 300))
    img_tk = ImageTk.PhotoImage(display_img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    result_label.config(
        text=f"Prediction: {label}\nConfidence: {confidence:.2f}%",
        fg="green" if label == "Healthy" else "red"
    )

# =========================
# GUI WINDOW
# =========================
root = tk.Tk()
root.title("Rubber Leaf Disease Detection")
root.geometry("400x500")

title_label = Label(root, text="Rubber Leaf Disease Detection",
                    font=("Arial", 14, "bold"))
title_label.pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

select_button = Button(root, text="Select Leaf Image",
                       command=open_image,
                       font=("Arial", 12),
                       bg="#4CAF50", fg="white")
select_button.pack(pady=20)

root.mainloop()
