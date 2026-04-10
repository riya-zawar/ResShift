from PIL import Image, ImageFilter
import numpy as np

def downgrade_image(input_path, output_path,
                    new_size=(256, 256),
                    quality=20,
                    blur_radius=1.5,
                    noise_level=15):
    """
    Downgrades an image by reducing resolution, quality, and adding blur + noise.
    
    Parameters:
    - input_path: Path to the input image.
    - output_path: Path to save the downgraded image.
    - new_size: (width, height) tuple for downscaling.
    - quality: JPEG quality level (1–100, lower = worse).
    - blur_radius: Gaussian blur intensity.
    - noise_level: Standard deviation of added Gaussian noise (0–50 typical).
    """
    # Load image
    img = Image.open(input_path).convert("RGB")

    # 2️⃣ Add blur
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    # 3️⃣ Add noise
    img_np = np.array(img)
    noise = np.random.normal(0, noise_level, img_np.shape).astype(np.int16)
    noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    img_noisy = Image.fromarray(noisy_img)

    # 4️⃣ Save with reduced quality
    img_noisy.save(output_path, quality=quality)

    print(f"✅ Downgraded image saved to: {output_path}")


# Example usage
downgrade_image(
    input_path="IMG_4706.JPG",
    output_path="downgraded.jpg",
    quality=50,           # reduce compression quality
    blur_radius=2.0,      # more blur
    noise_level=70        # add noise
)
