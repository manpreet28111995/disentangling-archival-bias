"""Run additional real-data VLM audits with the shared publication pipeline."""

import argparse
import io
import json
import os

import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor

from audit_statistics import run_publishable_analysis
from device_utils import get_device, move_inputs


PROMPT_SETS = {
    "set_1_masterpiece": ("an important masterpiece of fine art", "a minor, forgettable work of art"),
    "set_2_quality": ("a museum-quality masterwork", "an amateur painting"),
    "set_3_influence": ("a groundbreaking and influential artwork", "a decorative craft object"),
    "set_4_neutral_control": ("a museum-quality masterwork", "an artwork"),
}
PROMPTS = [p for pair in PROMPT_SETS.values() for p in pair]


def load_model(model_name):
    is_siglip = "siglip" in model_name.lower()
    if is_siglip:
        return SiglipModel.from_pretrained(model_name), SiglipProcessor.from_pretrained(model_name), True
    return CLIPModel.from_pretrained(model_name), CLIPProcessor.from_pretrained(model_name), False


def met_image(object_id, cache_dir=".met_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{object_id}.json")
    try:
        if os.path.exists(path):
            data = json.loads(open(path).read())
        else:
            response = requests.get(
                f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}",
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            with open(path, "w") as handle:
                json.dump(data, handle)
        return data.get("primaryImageSmall") or data.get("primaryImage")
    except Exception as exc:
        print(f"[WARN] Metadata lookup failed for {object_id}: {exc}")
    return None

def image_url(row):
    return row.get("primaryImageSmall") or row.get("primaryImage") or met_image(row.get("objectID"))


def download(url, object_id):
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        print(f"[WARN] Image download failed for {object_id}: {exc}")
        return None


def score(model, processor, images, is_siglip, device, batch_size=16):
    values = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = move_inputs(processor(text=PROMPTS, images=batch, return_tensors="pt", padding=True), device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits_per_image
        if not is_siglip:
            logits = logits.softmax(dim=1)
        values.extend(logits.cpu().numpy())
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="results/met_metadata_enriched.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--label", required=True, help="Filesystem-safe model label")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
    metadata = pd.read_csv(args.infile)
    model, processor, is_siglip = load_model(args.model_name)
    model.to(device).eval()

    images, rows, manifest = [], [], []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Loading {args.label}"):
        oid = row["objectID"]
        url = image_url(row)
        image = download(url, oid) if url else None
        manifest.append({
            "objectID": oid,
            "source_image_url": url,
            "image_status": "included" if image is not None else "excluded_image_unavailable",
            "model": args.label,
        })
        if image is not None:
            images.append(image)
            rows.append(row)

    probs = score(model, processor, images, is_siglip, device)
    records = []
    for row, image, values in zip(rows, images, probs):
        record = {
            "objectID": row["objectID"],
            "department": row.get("department"),
            "title": row.get("title"),
            "artistDisplayName": row.get("artistDisplayName"),
            "inferred_gender": row.get("inferred_gender"),
            "century": row.get("century", "unknown"),
            "medium_category": row.get("medium_category", "other"),
            "img_width": image.width,
            "img_height": image.height,
            "aspect_ratio": round(image.width / image.height, 2),
        }
        for index, name in enumerate(PROMPT_SETS):
            high, low = PROMPT_SETS[name]
            high_index, low_index = PROMPTS.index(high), PROMPTS.index(low)
            record[f"score_{name}"] = float(values[high_index] - values[low_index])
        record["mean_value_score"] = sum(record[f"score_{name}"] for name in PROMPT_SETS) / len(PROMPT_SETS)
        records.append(record)

    results = pd.DataFrame(records)
    slug = args.label.lower()
    results.to_csv(os.path.join(args.out_dir, f"{slug}_scores.csv"), index=False)
    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, f"{slug}_image_manifest.csv"), index=False)
    run_publishable_analysis(results, args.label, args.out_dir)
    print(f"Saved {len(results)} real image scores for {args.label}.")


if __name__ == "__main__":
    main()
