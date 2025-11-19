import replicate

training = replicate.trainings.create(
    version="replicate/fast-flux-trainer:56cb4a6447e586e40c6834a7a48b649336ade35325479817ada41cd3d8dcc175",
    input={
        "input_images": "./data",
        "txt": "portrait photo of rafanime",   # required caption
        "trigger_word": "rafanime",
        "training_steps": 500,
        "hardware": "gpu-l40s",                # cheaper + fast start
    },
    destination="rafanime/hair-face-lora"      # name your output model
)

print("✅ Training started!")
print("View progress at:", training.urls["web"])
