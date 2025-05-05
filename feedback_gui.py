import tkinter as tk
from PIL import Image, ImageTk
import torch
import torchvision
import torchvision.transforms as transforms
import json

# Load the trained CNN
from base_model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('base_cnn.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# GUI for feedback
class FeedbackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classification Feedback")
        self.index = 0
        self.feedback_data = []

        # Load first image
        self.image, self.label = testset[self.index]
        self.image_pil = transforms.ToPILImage()(self.image * 0.5 + 0.5)  # Denormalize
        self.photo = ImageTk.PhotoImage(self.image_pil.resize((420, 360)))

        # GUI elements
        self.canvas = tk.Canvas(root, width=420, height=360)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.pred_label = tk.Label(root, text="Predicted: ")
        self.pred_label.pack()
        self.true_label = tk.Label(root, text=f"True: {classes[self.label]}")
        self.true_label.pack()

        self.correct_btn = tk.Button(root, text="Correct", command=self.correct)
        self.correct_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.incorrect_btn = tk.Button(root, text="Incorrect", command=self.incorrect)
        self.incorrect_btn.pack(side=tk.LEFT, padx=5, pady=5)
        # Done button to save and exit
        self.done_btn = tk.Button(root, text="Done", command=self.done)
        self.done_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.update_prediction()

    def update_prediction(self):
        with torch.no_grad():
            output = model(self.image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
        self.pred_label.config(text=f"Predicted: {classes[pred]}")

    def correct(self):
        self.feedback_data.append({"index": self.index, "correct": True})
        self.next_image()

    def incorrect(self):
        self.feedback_data.append({"index": self.index, "correct": False})
        self.next_image()

    def next_image(self):
        self.index += 1
        if self.index >= len(testset):
            self.save_feedback()
            self.root.quit()
            return
        self.image, self.label = testset[self.index]
        self.image_pil = transforms.ToPILImage()(self.image * 0.5 + 0.5)
        self.photo = ImageTk.PhotoImage(self.image_pil.resize((420, 360)))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.true_label.config(text=f"True: {classes[self.label]}")
        self.update_prediction()

    def done(self):
        # Save feedback and exit early
        self.save_feedback()
        self.root.quit()

    def save_feedback(self):
        with open('feedback.json', 'w') as f:
            json.dump(self.feedback_data, f)
        print("Feedback saved to feedback.json")

if __name__ == "__main__":
    root = tk.Tk()
    app = FeedbackGUI(root)
    root.mainloop()