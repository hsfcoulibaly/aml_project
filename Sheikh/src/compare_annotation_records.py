
import json
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"

json_path = next(TRAIN_DIR.glob("*.json"))

with json_path.open("r", encoding="utf-8") as file:
    data = json.load(file)

# Translate category IDs into readable names.
category_names = {
    category["id"]: category["name"]
    for category in data["categories"]
}

# Group annotation image records by filename.
records_by_filename = defaultdict(list)

for image in data["images"]:
    filename = Path(image["file_name"]).stem
    records_by_filename[filename].append(image)

# Count annotations belonging to each image ID.
annotations_by_image_id = defaultdict(list)

for annotation in data["annotations"]:
    annotations_by_image_id[annotation["image_id"]].append(annotation)

# Investigate one filename with three annotation records.
target_filename = "20111116063134Lh"

records = records_by_filename[target_filename]

print("Filename:", target_filename)
print("Number of records:", len(records))

for record in records:
    image_id = record["id"]
    annotations = annotations_by_image_id[image_id]

    category_counts = Counter(
        category_names.get(annotation["category_id"], "Unknown")
        for annotation in annotations
    )

    print("\nImage ID:", image_id)
    print("Dimensions:", record["width"], "x", record["height"])
    print("Number of annotations:", len(annotations))
    print("Category counts:", dict(category_counts))

    # Inspect the annotation metadata.
    if annotations:
        annotation = annotations[0]
        print("First annotation ID:", annotation["id"])
        print("First annotation area:", annotation.get("area"))
        print("First annotation bounding box:", annotation.get("bbox"))
        print("First annotation category:",
              category_names.get(annotation["category_id"]))
