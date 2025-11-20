import os
import replicate

BASE_MODEL = "rafanime/hair-face-lora:d32fd7b78da8ef64a00910d15c0a04f89d2fa97f67263bfe346f6afe6e207ea8"
LORA_WEIGHTS = BASE_MODEL
TRIGGER_WORD = "rafanime"

reference_image_url = "https://media.vanityfair.com/photos/671aac828f6fb0005dcec934/master/w_1600%2Cc_limit/VF1224-VF-Hollywood-Portfolio-07.jpg"

beard_level = 0      # trying to remove beard
fade_level = 5

def fade_prompt(level):
    guide = {
        0: "no fade, uniform length sides",
        1: "low taper fade, subtle gradient near ears",
        2: "low fade, light skin fade around ears, soft transition",
        3: "mid fade, clean gradient from temples to ears",
        4: "high fade, sharp gradient, exposed lower sides",
        5: "skin fade, very tight sides, high contrast, ultra-clean transition"
    }
    return guide.get(level, "low fade")

def beard_prompt(level):
    guide = {
        0: "clean-shaven, completely shaved face, zero stubble, smooth bare skin",
        1: "very light stubble, 1-day growth",
        2: "light stubble, 2–3 day growth",
        3: "short beard, neatly trimmed",
        4: "medium beard, fuller cheek coverage",
        5: "full beard, thick beard growth",
    }
    return guide.get(level, "clean-shaven")

# ❗ STRONG beard removal here
extra_no_beard = (
    "clean-shaven, no beard, smooth bare skin, no mustache, "
    "freshly shaved face, zero facial hair, "
    "no stubble, no goatee, no shadow under lips"
)

prompt = (
    f"profile portrait of {TRIGGER_WORD}, adult male, masculine facial structure, "
    f"{beard_prompt(beard_level)}, {extra_no_beard}, "
    f"{fade_prompt(fade_level)}, "
    "unchanged real hairstyle, realistic lighting, studio background, "
    "accurate facial proportions, true likeness, no feminization, strict identity preservation, "
    "with ryan gosling haircut"
)

# ❗ STRONG anti-beard negative prompt
negative_prompt = (
    "beard, stubble, mustache, goatee, heavy beard, thick beard, "
    "facial hair, beard shadow, beard growth"
)

def main():

    image_input = reference_image_url

    input_payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,      # <-- NEW
        "image": image_input,
        "lora_weights": LORA_WEIGHTS,
        "lora_scale": 0.7,                       # <-- lower LoRA influence
        "go_fast": True,
        "guidance": 3,
        "megapixels": "1",
        "num_outputs": 2,
        "aspect_ratio": "3:4",
        "prompt_strength": 0.80,                 # <-- stronger prompt override
        "num_inference_steps": 28,
    }

    print("Running generation...")
    output = replicate.run(BASE_MODEL, input=input_payload)

    for i, url in enumerate(output):
        print(f"Image {i}: {url}")

if __name__ == "__main__":
    main()
