import os
import replicate

# Make sure you did:
# export REPLICATE_API_TOKEN=your_token_here

INPUT_ZIP_URL = "https://github.com/rafanime/hair_ai/raw/main/data.zip"

TRAINER_VERSION = (
    "stability-ai/sdxl:"
    "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
)

# IMPORTANT: create this empty model in the UI first:
# https://replicate.com/create  -> e.g. "rafanime/face-sdxl-lora"
DESTINATION_MODEL = "rafanime/face-sdxl"

training = replicate.trainings.create(
    version=TRAINER_VERSION,
    destination=DESTINATION_MODEL,
    input={
        "input_images": INPUT_ZIP_URL,
        # SDXL LoRA-related knobs:
        "is_lora": True,                  # default, but explicit is fine
        "max_train_steps": 800,           # similar to what you used before
        "unet_learning_rate": 1e-6,
        "lora_lr": 1e-4,
        # make the token your trigger word
        "token_string": "rafanime",
        "caption_prefix": "a photo of rafanime",
        # face-focused training
        "use_face_detection_instead": True,
    },
)

print("✅ Training started!")
print("ID:", training.id)
print("Web:", training.urls["web"])
