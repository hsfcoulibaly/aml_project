
import json
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"
IMAGE_DIR = TRAIN_DIR / "train_images"

json_path = next(TRAIN_DIR.glob("*.json"))

with json_path.open("r", encoding="utf-8") as file:
    data = json.load(file)

# Map category IDs to category names.
categories = {
    category["id"]: category["name"]
    for category in data["categories"]
}

# Group records that refer to the same underlying filename.
records_by_filename = defaultdict(list)

for image in data["images"]:
    filename = Path(image["file_name"]).stem
    records_by_filename[filename].append(image)

# Group annotations by annotation image ID.
annotations_by_image = defaultdict(list)

for annotation in data["annotations"]:
    annotations_by_image[annotation["image_id"]].append(annotation)

# Count image files.
image_files = [
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
]

# Count records and category frequencies.
records_per_filename = Counter(
    len(records) for records in records_by_filename.values()
)

category_counts = Counter(
    categories.get(annotation["category_id"], "Unknown")
    for annotation in data["annotations"]
)

print("DATASET SUMMARY")
print("=" * 40)
print("Physical image files:", len(image_files))
print("Unique image filenames:", len(records_by_filename))
print("Annotation records:", len(data["images"]))
print("Total filament annotations:", len(data["annotations"]))
print("Category counts:", dict(category_counts))

print("\nANNOTATION RECORDS PER PHYSICAL IMAGE")
for number_of_records, number_of_images in sorted(records_per_filename.items()):
    print(
        f"{number_of_records} record(s): "
        f"{number_of_images} image(s)"
    )

# Report the number of unique observations with each category.
images_with_category = Counter()

for image in data["images"]:
    image_id = image["id"]
    filename = Path(image["file_name"]).stem

    present_categories = {
        categories.get(annotation["category_id"], "Unknown")
        for annotation in annotations_by_image[image_id]
    }

    for category in present_categories:
        images_with_category[category] += 1

print("\nANNOTATION RECORDS CONTAINING EACH CATEGORY")
for category, count in sorted(images_with_category.items()):
    print(f"{category}: {count} records")
