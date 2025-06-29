import os
import glob
from PIL import Image

class ImageDataset:
    def __init__(self, root_dir, subfolder="image_2", extension="jpg"):
        """
        Initialize the dataset by recursively finding all images with the given extension in 
        the specified subfolder under the root directory.
        
        Parameters:
          root_dir (str): The dataset path e.g. 
              "C:/Personal Projects/AI_Powered_traffic_management_system/data/trafficnet_dataset_v1/trafficnet_dataset_v1/train"
          subfolder (str): The subfolder under which images live (e.g. "image_2"). If images
              reside directly under root_dir, pass an empty string ("").
          extension (str): File extension for image files (e.g. "jpg" or "png").
        """
        if subfolder:
            pattern = os.path.join(root_dir, "**", subfolder, f"*.{extension}")
        else:
            pattern = os.path.join(root_dir, "**", f"*.{extension}")
        self.paths = sorted(glob.glob(pattern, recursive=True))
        if not self.paths:
            raise FileNotFoundError(
                f"No images found under {root_dir} in subfolder '{subfolder}' with extension {extension}"
            )
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return img, {}  # In this demo, we don’t provide labels automatically

def load_dataset(root_dir, subfolder="image_2", extension="jpg"):
    """
    A helper function that creates an instance of ImageDataset.
    
    For training data:
        dataset = load_dataset(
            root_dir="C:/Personal Projects/AI_Powered_traffic_management_system/data/trafficnet_dataset_v1/trafficnet_dataset_v1/train",
            subfolder="image_2",
            extension="jpg"
        )
    
    For test data, change the root_dir accordingly.
    
    Returns:
        An instance of ImageDataset containing your images.
    """
    return ImageDataset(root_dir, subfolder, extension)