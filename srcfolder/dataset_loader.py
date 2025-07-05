import os
import glob
from PIL import Image

class ImageDataset:
    """
    A custom dataset class to load images from a specified directory structure.
    It recursively searches for images within a given subfolder and extension.
    This design allows for easy switching between datasets like KITTI and Traffic-Net
    by adjusting the `root_dir` and `subfolder` parameters.
    """
    def __init__(self, root_dir, subfolder="image_2", extension="png"):
        """
        Initializes the dataset by recursively finding all images with
        the given extension inside the specified subfolder under the root.

        Parameters:
          root_dir (str): The dataset folder path (e.g., the train or test folder).
                          This is the base directory where your images reside.
          subfolder (str): The specific subfolder within the dataset structure
                           where images are directly located (e.g., "image_2" for KITTI-like structures).
                           If images reside directly under `root_dir` or its subdirectories
                           without a specific named subfolder like "image_2", use an empty string ("").
          extension (str): File extension to look for (e.g., "jpg", "jpeg", "png").
        """
        # Construct the search pattern based on whether a subfolder is specified.
        # "**" allows for recursive search through any number of subdirectories.
        if subfolder:
            pattern = os.path.join(root_dir, "**", subfolder, f"*.{extension}")
        else:
            pattern = os.path.join(root_dir, "**", f"*.{extension}")

        # Use glob.glob with recursive=True to find all matching files.
        # sorted() ensures a consistent order of images across runs.
        self.paths = sorted(glob.glob(pattern, recursive=True))

        if not self.paths:
            # Raise a clear error if no images are found, indicating a potential path issue.
            raise FileNotFoundError(
                f"No images found matching pattern '{pattern}'. "
                f"Please check the 'root_dir', 'subfolder', and 'extension' parameters."
            )
        print(f"Dataset Loader: Found {len(self.paths)} images matching pattern: {pattern}")

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Retrieves an image at the given index.

        Parameters:
          idx (int): The zero-based index of the image to retrieve.

        Returns:
          PIL.Image: The loaded image in RGB format. PIL is used for its compatibility
                     with various image formats and ease of use with torchvision transforms.
          dict: An empty dictionary for compatibility with standard PyTorch dataset
                conventions (this loader focuses solely on image loading, not labels).
        """
        # Open the image using PIL and convert it to RGB to ensure a consistent 3-channel format,
        # which is typically required by deep learning models.
        img = Image.open(self.paths[idx]).convert("RGB")
        return img, {}  # Return the image and an empty dictionary (for labels)

def load_dataset(root_dir, subfolder="image_2", extension="png"):
    """
    Helper function to easily create an ImageDataset instance.
    This function acts as a convenient wrapper around the ImageDataset class.

    Example Usage for Traffic-Net (assuming images are in 'train/image_2/'):
      # dataset = load_dataset(
      #     root_dir="C:/Projects/AI_Traffic/data/trafficnet_dataset_v1/trafficnet_dataset_v1/train",
      #     subfolder="image_2",
      #     extension="jpg"
      # )

    Example Usage for KITTI (assuming images are in 'training/image_2/'):
      # dataset = load_dataset(
      #     root_dir="C:/Datasets/KITTI/training", # KITTI's training folder contains image_2, image_3, etc.
      #     subfolder="image_2",
      #     extension="png" # KITTI typically uses PNG
      # )

    Example Usage for a simple flat directory of images:
      # dataset = load_dataset(
      #     root_dir="C:/MyImages/SceneDataset",
      #     subfolder="", # No specific subfolder, images are directly here or in sub-subfolders
      #     extension="jpg"
      # )

    Returns:
      ImageDataset: An instance of the ImageDataset object, ready to be iterated over.
    """
    return ImageDataset(root_dir, subfolder, extension)
