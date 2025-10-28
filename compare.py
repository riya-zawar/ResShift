from PIL import Image
import os

sinsr_folder = "SinSR/results"
resshift_folder = "ResShift/results"
comparison_folder = "comparison"
os.makedirs(comparison_folder, exist_ok=True)

# Loop through all images
for img_name in os.listdir(sinsr_folder):
    sin_img = Image.open(os.path.join(sinsr_folder, img_name))
    res_img = Image.open(os.path.join(resshift_folder, img_name))
    
    # Ensure same width
    if sin_img.width != res_img.width:
        new_width = min(sin_img.width, res_img.width)
        sin_img = sin_img.resize((new_width, int(sin_img.height * new_width / sin_img.width)))
        res_img = res_img.resize((new_width, int(res_img.height * new_width / res_img.width)))
    
    # Create a new image stacked vertically
    combined = Image.new('RGB', (sin_img.width, sin_img.height + res_img.height))
    combined.paste(sin_img, (0, 0))
    combined.paste(res_img, (0, sin_img.height))
    
    combined.save(os.path.join(comparison_folder, img_name))

print(f"All comparisons saved in {comparison_folder}")