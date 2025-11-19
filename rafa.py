import replicate

model = replicate.models.get("rafanime/fast-flux-trainer")
version = model.versions.get("xxxx")  # your version ID

output = replicate.run(
    version,
    input={
        "prompt": "a photo of rafa with short curly hair, cinematic lighting",
        "image": open("selfie.png", "rb"),  # optional reference
    }
)

print("Output image URL:", output)