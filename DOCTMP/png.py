import os
from PIL import Image

input_folder = "/home/ilyassechaouki/DOCTMP/output_jpeg/esp auth"
output_folder = "/home/ilyassechaouki/DOCTMP/output_jpeg/esp jpg"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.lower().endswith(".png"):
        img = Image.open(os.path.join(input_folder, file)).convert("RGB")
        out_path = os.path.join(output_folder, file.replace(".png", ".jpg"))
        img.save(out_path, format="JPEG", quality=95, optimize=False)
