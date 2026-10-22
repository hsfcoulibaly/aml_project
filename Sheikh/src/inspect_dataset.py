
from pathlib import Path

# The root folder of our project
PROJECT_DIR = Path(__file__).resolve().parent.parent

# The folder containing the training images
TRAIN_DIR = PROJECT_DIR / "data" / "train"
IMAGE_DIR = TRAIN_DIR / "train_images"

# Find the annotation JSON file
json_files = list(TRAIN_DIR.glob("*.json"))

print("Project directory:", PROJECT_DIR)
print("Training directory exists:", TRAIN_DIR.exists())
print("Image directory exists:", IMAGE_DIR.exists())

# Count the image files
if IMAGE_DIR.exists():
    image_extensions = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}

    image_paths = [
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in image_extensions
    ]

    print("Number of training images:", len(image_paths))

    # Show the first five filenames
    for path in image_paths[:5]:
        print("Image:", path.name)
else:
    print("Please check the location of your training images.")

# List the annotation JSON files
print("Annotation JSON files:", len(json_files))

for path in json_files:
    print("Annotation file:", path.name)
