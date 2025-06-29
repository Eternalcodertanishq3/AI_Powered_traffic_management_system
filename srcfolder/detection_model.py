import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
from typing import Tuple, Dict, List, Any

# Add the srcfolder to the Python path to allow importing modules correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import constants from the central constants.py, including scene classes and model path
from .constants import SCENE_CLASSES, SCENE_CLASSIFIER_MODEL_PATH 

# --- DEVICE CONFIGURATION - MOVED TO TOP FOR IMMEDIATE AVAILABILITY ---
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[JARVIS-LOG] Scene classification inference device: {_device}")

# Define the SceneClassifier model architecture
class SceneClassifier(nn.Module):
    def __init__(self, num_classes: int):
        """
        Initializes the SceneClassifier model.

        Args:
            num_classes (int): The number of unique scene categories to classify.
        """
        super(SceneClassifier, self).__init__()
        
        # Load a pre-trained ResNet-50 model from torchvision.
        # 'weights=models.ResNet50_Weights.IMAGENET1K_V1' loads the weights
        # trained on the ImageNet dataset, which provides a strong starting point.
        # IMPORTANT: Naming this 'resnet' to match your EXISTING 'scene_classifier.pth' file.
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all parameters in the ResNet backbone
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer (classifier head) of ResNet.
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        return self.resnet(x)

# Global model instances and status flag
# These are initialized at the module level.
_scene_model = None 
_model_loaded_successfully = False 


def load_scene_model() -> SceneClassifier:
    """
    Loads the scene classification model and its trained weights.
    This function is designed to be called once, making subsequent predictions faster.
    """
    # Declare that we intend to modify the global variables _scene_model and _model_loaded_successfully
    global _scene_model, _model_loaded_successfully 

    if _scene_model is None:
        try:
            model_instance = SceneClassifier(num_classes=len(SCENE_CLASSES))
            
            if os.path.exists(SCENE_CLASSIFIER_MODEL_PATH):
                print(f"[JARVIS-LOG] Attempting to retrieve trained model from: {SCENE_CLASSIFIER_MODEL_PATH}")
                state_dict = torch.load(SCENE_CLASSIFIER_MODEL_PATH, map_location=_device)
                
                model_instance.load_state_dict(state_dict, strict=True) 
                
                model_instance.to(_device) # Move model to the selected device
                model_instance.eval() # Set model to evaluation mode (IMPORTANT)
                _scene_model = model_instance # Assign to the global variable
                _model_loaded_successfully = True # Assign to the global variable
                print(f"[JARVIS-LOG] Scene classifier core module online and calibrated. Status: Optimal.")
            else:
                print(f"[JARVIS-WARNING] Trained scene classifier model file not found at {SCENE_CLASSIFIER_MODEL_PATH}.")
                print("[JARVIS-WARNING] Operating with randomly initialized/ImageNet pre-trained backbone. Predictions will be conceptual, not highly calibrated. Prioritize training.")
                _scene_model = model_instance.to(_device) # Assign to the global variable
                _scene_model.eval()
                _model_loaded_successfully = False # Mark as not fully loaded
        except Exception as e:
            print(f"[JARVIS-ERROR] Core module initialization failed: {e}")
            print("[JARVIS-ERROR] Scene classification services are degraded. Please inspect model integrity and dependencies.")
            _scene_model = None # Ensure model is None if loading fails completely
            _model_loaded_successfully = False
    
    return _scene_model

# Define the image transforms for inference (resize, center crop, to tensor, normalize)
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_scene_prediction(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Analyzes the input image and provides a detailed scene classification report.
    This function aims to mimic a more 'intelligent' AI output, providing
    not just the top prediction but also confidence assessment and suggested actions.
    """
    if not isinstance(pil_image, Image.Image):
        print(f"[JARVIS-ERROR] Input for visual analysis is not a valid image format. Received: {type(pil_image)}")
        return {
            "main_prediction": "INVALID_INPUT",
            "main_confidence": 0.0,
            "confidence_assessment": "Critical Error",
            "suggested_action": "System integrity check required. Input data is malformed.",
            "all_predictions": [],
            "diagnostic_notes": f"Expected PIL.Image.Image, got {type(pil_image)}.",
            "status": "ERROR"
        }

    model = load_scene_model() # Ensure model is loaded (or attempted)
    if model is None or not _model_loaded_successfully:
        # If model couldn't load properly, return a default/error report
        return {
            "main_prediction": "UNAVAILABLE",
            "main_confidence": 0.0,
            "confidence_assessment": "Offline",
            "suggested_action": "Core visual intelligence module not operational. Please initiate retraining sequence.",
            "all_predictions": [],
            "diagnostic_notes": "Model file missing or failed to load. Predictions are not possible.",
            "status": "MODEL_ERROR"
        } 

    try:
        with torch.no_grad(): # Disable gradient calculations for inference (saves memory, faster)
            # Apply transformations and add a batch dimension (unsqueeze(0))
            input_tensor = inference_transforms(pil_image).unsqueeze(0).to(_device)
            
            # Get raw model outputs (logits)
            outputs = model(input_tensor)
            
            # Convert logits to probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze(0) # Remove batch dimension
            
            # Get the top prediction
            confidence, predicted_idx = torch.max(probabilities, 0) # 0 for single image prediction
            main_prediction = SCENE_CLASSES[predicted_idx.item()]
            main_confidence = confidence.item()

            # Get top N predictions for detailed report
            top_n = min(3, len(SCENE_CLASSES)) # Get top 3, or fewer if less classes
            top_confidences, top_indices = torch.topk(probabilities, top_n)
            all_predictions_list = [
                {"label": SCENE_CLASSES[idx.item()], "confidence": conf.item()}
                for idx, conf in zip(top_indices, top_confidences)
            ]

            # Assess confidence qualitatively
            confidence_assessment = "Unknown"
            if main_confidence >= 0.95:
                confidence_assessment = "High Confidence"
            elif main_confidence >= 0.75:
                confidence_assessment = "Moderate Confidence"
            elif main_confidence >= 0.50:
                confidence_assessment = "Low Confidence"
            else:
                confidence_assessment = "Very Low Confidence - Manual Review Advised"

            # Suggest actions based on prediction (JARVIS-like awareness)
            suggested_action = "Monitoring traffic flow."
            if main_prediction == "accident":
                if main_confidence >= 0.90:
                    suggested_action = "CRITICAL INCIDENT DETECTED. Immediate alert to emergency services and traffic control."
                else:
                    suggested_action = "Potential incident identified. Initiate close-up visual verification and prepare for response."
            elif main_prediction == "dense_traffic":
                suggested_action = "High congestion detected. Suggest optimizing traffic light cycles and evaluating alternative routes."
            elif main_prediction == "stalled_vehicle":
                suggested_action = "Stalled vehicle detected. Dispatch roadside assistance. Advise caution to approaching vehicles."
            elif main_prediction == "emergency_vehicle_passing":
                suggested_action = "Emergency vehicle in transit. Initiate green-wave protocol for its trajectory. Advise clearing pathways."
            elif main_prediction == "adverse_weather_rain":
                suggested_action = "Heavy rain detected. Advise reduced speed limits and increased caution for drivers."
            # ... add more sophisticated actions for other SCENE_CLASSES

            return {
                "main_prediction": main_prediction,
                "main_confidence": main_confidence,
                "confidence_assessment": confidence_assessment,
                "suggested_action": suggested_action,
                "all_predictions": all_predictions_list,
                "diagnostic_notes": "Visual analysis complete. Integration with external sensor data pending for comprehensive assessment.",
                "status": "OK"
            }

    except Exception as e:
        print(f"[JARVIS-ERROR] An error occurred during visual analysis: {e}")
        return {
            "main_prediction": "SYSTEM_ERROR",
            "main_confidence": 0.0,
            "confidence_assessment": "Critical Failure",
            "suggested_action": "Urgent diagnostic required. Core prediction module encountered an unhandled exception.",
            "all_predictions": [],
            "diagnostic_notes": f"Exception: {e}. Please report to engineering.",
            "status": "ERROR"
        }

if __name__ == "__main__":
    # Example usage for testing detection_model.py directly
    print("\n--- Initializing JARVIS-Level AI Detection Model Self-Test ---")
    
    # Test case 1: Model file missing (simulate by temporarily renaming model)
    original_model_path = SCENE_CLASSIFIER_MODEL_PATH
    temp_model_path = original_model_path + ".bak"
    # ONLY attempt to move if the file exists
    if os.path.exists(original_model_path):
        try:
            os.rename(original_model_path, temp_model_path)
            print(f"[JARVIS-TEST] Temporarily moved {original_model_path} to {temp_model_path} for missing file test.")
        except OSError as e:
            print(f"[JARVIS-TEST] Could not temporarily move model file: {e}. Skipping missing file test.")
            temp_model_path = None # Indicate it wasn't moved
    
    print("\n[JARVIS-TEST] Test Case 1: Model file missing.")
    dummy_image = Image.new('RGB', (224, 224), color='grey')
    report = get_scene_prediction(dummy_image)
    print(f"Report Status: {report['status']}")
    print(f"Report Main Prediction: {report['main_prediction']}")
    print(f"Report Suggested Action: {report['suggested_action']}")
    
    # Restore model file if it was moved
    if temp_model_path and os.path.exists(temp_model_path):
        os.rename(temp_model_path, original_model_path)
        print(f"[JARVIS-TEST] Restored {original_model_path}.")

    # Re-load model after restoration for subsequent tests
    # Reset global variables directly, no 'global' keyword needed in this script-level context
    _scene_model = None 
    _model_loaded_successfully = False
    load_scene_model()

    # Test Case 2: Invalid input type
    print("\n[JARVIS-TEST] Test Case 2: Invalid input type (string instead of PIL Image).")
    report = get_scene_prediction("this is not an image")
    print(f"Report Status: {report['status']}")
    print(f"Report Main Prediction: {report['main_prediction']}")
    print(f"Report Suggested Action: {report['suggested_action']}")

    # Test Case 3: Dummy image prediction (with a loaded model, even if untrained or newly loaded)
    print("\n[JARVIS-TEST] Test Case 3: Dummy image prediction (assuming model loaded).")
    dummy_image_blue = Image.new('RGB', (224, 224), color='blue')
    report = get_scene_prediction(dummy_image_blue)
    print(f"Report Status: {report['status']}")
    print(f"Report Main Prediction: {report['main_prediction']}")
    print(f"Report Main Confidence: {report['main_confidence']:.2f}")
    print(f"Report Confidence Assessment: {report['confidence_assessment']}")
    print(f"Report Suggested Action: {report['suggested_action']}")
    print(f"Report All Predictions (Top 3): {report['all_predictions']}")
    print(f"Report Diagnostic Notes: {report['diagnostic_notes']}")

    # Test Case 4: Real image prediction (Requires an actual image file in your system)
    print("\n[JARVIS-TEST] Test Case 4: Real image prediction (requires existing image).")
    test_real_image_path = "C:/Personal Projects/AI_Powered_traffic_management_system/data/Traf_Acci_Data/image_2.1/Accident1 (1).jpg" 
    if os.path.exists(test_real_image_path):
        try:
            real_image = Image.open(test_real_image_path).convert("RGB")
            report = get_scene_prediction(real_image)
            print(f"Report Status: {report['status']}")
            print(f"Report Main Prediction: {report['main_prediction']}")
            print(f"Report Main Confidence: {report['main_confidence']:.2f}")
            print(f"Report Confidence Assessment: {report['confidence_assessment']}")
            print(f"Report Suggested Action: {report['suggested_action']}")
            print(f"Report All Predictions (Top 3): {report['all_predictions']}")
            print(f"Report Diagnostic Notes: {report['diagnostic_notes']}")
        except Exception as e:
            print(f"[JARVIS-TEST] Failed to load/process test image at {test_real_image_path}: {e}")
    else:
        print(f"[JARVIS-TEST] Test image not found at {test_real_image_path}. Skipping real image test.")

    print("\n--- JARVIS-Level AI Detection Model Self-Test Complete ---")
