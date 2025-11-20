import os
import replicate

# Assumes REPLICATE_API_TOKEN is set in your env
# export REPLICATE_API_TOKEN=...
# Or pass it explicitly with replicate.Client(auth=...)

# 1. Public URL to your zipped training data
INPUT_ZIP_URL = "https://raw.githubusercontent.com/rafanime/hair_ai/refs/heads/main/data.zip"
TRAINER_VERSION = "replicate/fast-flux-trainer:8b10794665aed907bb98a1a5324cd1d3a8bea0e9b31e65210967fb9c9e2e08ed"
DESTINATION_MODEL = "rafanime/hair-face-lora"
INPUT_ZIP_URL = "https://github.com/rafanime/hair_ai/raw/main/data.zip"

training = replicate.trainings.create(
    version=TRAINER_VERSION,
    destination=DESTINATION_MODEL,
    input={
        "input_images": INPUT_ZIP_URL,
        "trigger_word": "rafanime",
        "lora_type": "subject",
        "steps": 800,  # good starting point
    },
)

print("✅ Training created")
print("ID:", training.id)
print("Web:", training.urls["web"])
