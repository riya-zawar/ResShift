import os
import torch
import pyiqa
from PIL import Image
from tqdm import tqdm
import pandas as pd

# -------------------------------
# Paths to result folders
# -------------------------------
sinsr_folder = "/home/zawar/SinSR/results"
resshift_folder = "/home/zawar/ResShift/results"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------
# Initialize all metrics
# -------------------------------
metrics = {
    "NIQE": pyiqa.create_metric('niqe', device=device),
    "BRISQUE": pyiqa.create_metric('brisque', device=device),
    "CLIP-IQA+": pyiqa.create_metric('clipiqa+_vitL14_512', device=device),
    "MUSIQ": pyiqa.create_metric('musiq-spaq', device=device)
}

# -------------------------------
# Compute metrics for all images in a folder
# -------------------------------
def compute_scores(folder, metrics):
    results = {name: [] for name in metrics.keys()}
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"\nEvaluating {len(image_files)} images in {folder}...\n")

    for img_name in tqdm(sorted(image_files)):
        img_path = os.path.join(folder, img_name)
        img = Image.open(img_path).convert("RGB")

        for name, metric in metrics.items():
            with torch.no_grad():
                score = metric(img)
            results[name].append(float(score))

    avg_scores = {name: sum(values) / len(values) for name, values in results.items()}
    return results, avg_scores, image_files

# -------------------------------
# Compute and compare
# -------------------------------
sinsr_results, sinsr_avg, sinsr_images = compute_scores(sinsr_folder, metrics)
resshift_results, resshift_avg, resshift_images = compute_scores(resshift_folder, metrics)

# -------------------------------
# Print average results
# -------------------------------
print("\n📊 Average Image Quality Scores:")
print(f"{'Metric':<15} {'SinSR':>10} {'ResShift':>12}")
print("-" * 40)
for metric in metrics.keys():
    print(f"{metric:<15} {sinsr_avg[metric]:>10.4f} {resshift_avg[metric]:>12.4f}")

# -------------------------------
# Save detailed results to CSV
# -------------------------------
df = pd.DataFrame({
    "Image": sinsr_images,
    "SinSR_NIQE": sinsr_results["NIQE"],
    "SinSR_BRISQUE": sinsr_results["BRISQUE"],
    "SinSR_CLIP-IQA+": sinsr_results["CLIP-IQA+"],
    "SinSR_MUSIQ": sinsr_results["MUSIQ"],
    "ResShift_NIQE": resshift_results["NIQE"],
    "ResShift_BRISQUE": resshift_results["BRISQUE"],
    "ResShift_CLIP-IQA+": resshift_results["CLIP-IQA+"],
    "ResShift_MUSIQ": resshift_results["MUSIQ"]
})

df.to_csv("iqa_comparison_results.csv", index=False)
print("\n✅ Saved detailed results to iqa_comparison_results.csv")

print("\nHigher CLIP-IQA+ & MUSIQ → better perceptual quality.")
print("Lower NIQE & BRISQUE → better perceptual quality.\n")
