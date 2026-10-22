
import json
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"

json_files = list(TRAIN_DIR.glob("*.json"))

if not json_files:
    raise FileNotFoundError(
        f"No annotation JSON file found in {TRAIN_DIR}"
    )

annotation_path = json_files[0]

# Load the JSON file into a Python dictionary
with annotation_path.open("r", encoding="utf-8") as file:
    data = json.load(file)

print("JSON file:", annotation_path.name)
print("Top-level keys:", list(data.keys()))

images = data.get("images", [])
annotations = data.get("annotations", [])
categories = data.get("categories", [])

print("\nNumber of image records:", len(images))
print("Number of annotations:", len(annotations))
print("Number of categories:", len(categories))

print("\nCategories:")
for category in categories:
    print(category)

print("\nFirst image record:")
if images:
    print(json.dumps(images[0], indent=2))

print("\nFirst annotation record:")
if annotations:
    print(json.dumps(annotations[0], indent=2))

# Count annotations per image record
annotation_counts = Counter(
    annotation["image_id"]
    for annotation in annotations
)

print("\nAnnotation counts for the first five image IDs:")
for image_id, count in annotation_counts.most_common(5):
    print(image_id, "->", count, "filaments")
