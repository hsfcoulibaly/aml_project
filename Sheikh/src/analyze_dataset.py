
import json
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"
IMAGE_DIR = TRAIN_DIR / "train_images"

json_files = list(TRAIN_DIR.glob("*.json"))

if not json_files:
    raise FileNotFoundError("No annotation JSON file found.")

with json_files[0].open("r", encoding="utf-8") as file:
    data = json.load(file)

images = data["images"]
annotations = data["annotations"]
categories = {
    category["id"]: category["name"]
    for category in data["categories"]
}

# Count actual image files by filename stem.
image_files = [
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
]

available_stems = {path.stem for path in image_files}

# Group annotation image records by filename.
records_by_filename = defaultdict(list)

for image_info in images:
    stem = Path(image_info["file_name"]).stem
    records_by_filename[stem].append(image_info)

print("Actual image files:", len(image_files))
print("Annotation image records:", len(images))
print("Unique annotation filenames:", len(records_by_filename))

matched = sum(
    stem in available_stems
    for stem in records_by_filename
)

print("Unique annotation filenames with files:", matched)

# Find filenames with multiple annotation records.
duplicates = {
    filename: records
    for filename, records in records_by_filename.items()
    if len(records) > 1
}

print("Filenames with multiple annotation records:", len(duplicates))

for filename, records in list(duplicates.items())[:5]:
    print("\nFilename:", filename)
    for record in records:
        print("  Image ID:", record["id"])

# Count annotations by image ID and category.
counts_by_image = Counter(
    annotation["image_id"] for annotation in annotations
)

counts_by_category = Counter(
    categories.get(annotation["category_id"], "Unknown")
    for annotation in annotations
)

print("\nAnnotations by category:")
for category, count in counts_by_category.items():
    print(f"  {category}: {count}")

print("\nAnnotations per image record:")
print("  Records with annotations:", len(counts_by_image))
print("  Maximum annotations on one record:", max(counts_by_image.values(), default=0))
print("  Average annotations per annotated record:",
      round(sum(counts_by_image.values()) / max(len(counts_by_image), 1), 2))
