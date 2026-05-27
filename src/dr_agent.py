import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Import the new Heatmap tools
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class DRAgent:
    def __init__(self, model_path="dr_model_weights.pth"):
        self.device = torch.device("cpu")
        self.target_size = (224, 224)
        
        print("Waking up Agent... Rebuilding the brain architecture.")
        self.model = models.resnet50(weights=None) 
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 5)
        
        print("Injecting learned memories...")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 
        
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ])
        
        # --- NEW: Setup the Heatmap Generator ---
        # We hook into the final visual layer of ResNet50
        self.target_layers = [self.model.layer4[-1]]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        
        print("Agent is fully loaded and ready to diagnose!")

    def preprocess_image(self, image_path):
        """Loads the image and returns both the tensor for the AI and the raw image for the heatmap."""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)
        return input_tensor.to(self.device), image

    def run_analysis(self, image_path):
        """The main control center."""
        # Note: We now catch two variables from preprocessing
        input_tensor, original_image = self.preprocess_image(image_path)
        
        # 1. Get the Prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)
            
            dr_class = predicted_class.item()
            conf_score = confidence.item()

        # --- NEW: Generate the Heatmap ---
        # 2. Run the Grad-CAM algorithm to see what the AI looked at
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :] # Extract the 2D map
        
        # 3. Resize original image to match the tensor and convert it to float (0 to 1) for overlay
        img_resized = np.array(original_image.resize(self.target_size)) / 255.0
        
        # 4. Create the visual overlay 
        visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)
        
        # 5. Save the heatmap to the data folder
        heatmap_path = "../data/heatmap_output.jpg"
        # OpenCV uses BGR instead of RGB, so we flip the colors before saving
        cv2.imwrite(heatmap_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        # --- Final Output Formatting ---
        explanation = "Heatmap generated! Red areas show the lesions the AI focused on."
        difficulty = "None. High confidence." if conf_score > 0.60 else f"Struggling. Only {conf_score*100:.1f}% sure."
        
        return {
            "DR_Class": dr_class,
            "Explanation": explanation,
            "Confidence": f"{conf_score * 100:.2f}%",
            "Agent_Struggles": difficulty,
            "Heatmap_Saved_At": heatmap_path
        }

# --- Testing the Agent ---
if __name__ == "__main__":
    import os
    weights_path = "dr_model_weights.pth" 
    
    if not os.path.exists(weights_path):
        print(f"ERROR: Cannot find the brain file at {weights_path}!")
        exit()

    agent = DRAgent(model_path=weights_path)
    test_image = "../data/test_scan.png" 
    
    try:
        print(f"\nAnalyzing image: {test_image}...")
        results = agent.run_analysis(test_image)
        
        print("\n=== AI DIAGNOSIS REPORT ===")
        for key, value in results.items():
            print(f"{key}: {value}")
        print("===========================\n")
        
    except FileNotFoundError:
        print(f"\nERROR: Could not find an image at '{test_image}'.")