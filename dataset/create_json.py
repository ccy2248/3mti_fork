import json

data = {
    "train": {
        "target_image": "/tmp/train/rgb",
        "image": "/tmp/train/cloudy",
        "ref_image": "/tmp/train/vh",
        "prompt": "remove cloud"
    },
    "test": {
        "target_image": "/tmp/val/rgb",
        "image": "/tmp/val/cloudy",
        "ref_image": "/tmp/val/vh",
        "prompt": "remove cloud"
    }
}

with open("./your_dataset.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

print("JSON file created successfully!")
