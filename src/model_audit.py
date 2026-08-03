"""Run one configurable real-image VLM audit."""

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
PROMPTS = list(dict.fromkeys(prompt for pair in PROMPT_SETS.values() for prompt in pair))
PROMPT_INDEX = {prompt: i for i, prompt in enumerate(PROMPTS)}


def load_model(model_name, device):
    siglip = "siglip" in model_name.lower()
    cls, processor_cls = (SiglipModel, SiglipProcessor) if siglip else (CLIPModel, CLIPProcessor)
    model = cls.from_pretrained(model_name)
    processor = processor_cls.from_pretrained(model_name)
    model.to(device).eval()
    return model, processor, siglip


def image_url(row, cache_dir=".met_cache"):
    url = row.get("primaryImageSmall") or row.get("primaryImage")
    if url:
        return url
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{row.get('objectID')}.json")
    try:
        if os.path.exists(path):
            data = json.loads(open(path).read())
        else:
            response = requests.get(
                f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{row.get('objectID')}",
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            with open(path, "w") as handle:
                json.dump(data, handle)
        return data.get("primaryImageSmall") or data.get("primaryImage")
    except Exception as exc:
        print(f"[WARN] Metadata lookup failed for {row.get('objectID')}: {exc}")
        return None


def download(url, object_id):
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        print(f"[WARN] Image download failed for {object_id}: {exc}")
        return None


def score(model, processor, images, siglip, device, batch_size=16):
    values = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        inputs = move_inputs(processor(text=PROMPTS, images=batch, return_tensors="pt", padding=True), device)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image
        values.extend((logits if siglip else logits.softmax(dim=1)).cpu().numpy())
    return values


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    metadata = pd.read_csv(args.infile)
    if args.n_female is not None and args.n_male is not None:
        parts = []
        for gender, count in (("female", args.n_female), ("male", args.n_male)):
            group = metadata[metadata.inferred_gender == gender]
            parts.append(group.sample(n=min(count, len(group)), random_state=42))
        metadata = pd.concat(parts, ignore_index=True)

    model, processor, siglip = load_model(args.model_name, device)
    images, rows, manifest = [], [], []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=f"Loading {args.label}"):
        oid = row["objectID"]
        url = image_url(row)
        image = download(url, oid) if url else None
        manifest.append({"objectID": oid, "source_image_url": url,
                         "image_status": "included" if image is not None else "excluded_image_unavailable",
                         "model": args.label})
        if image is not None:
            images.append(image)
            rows.append(row)

    values = score(model, processor, images, siglip, device)
    records = []
    for row, image, scores in zip(rows, images, values):
        record = {"objectID": row["objectID"], "department": row.get("department"),
                  "title": row.get("title"), "artistDisplayName": row.get("artistDisplayName"),
                  "inferred_gender": row.get("inferred_gender"), "century": row.get("century", "unknown"),
                  "medium_category": row.get("medium_category", "other"), "img_width": image.width,
                  "img_height": image.height, "aspect_ratio": round(image.width / image.height, 2)}
        contrasts = []
        for name, (high, low) in PROMPT_SETS.items():
            contrast = float(scores[PROMPT_INDEX[high]] - scores[PROMPT_INDEX[low]])
            record[f"score_{name}"] = contrast
            contrasts.append(contrast)
        record["mean_value_score"] = sum(contrasts) / len(contrasts)
        records.append(record)

    slug = args.label.lower()
    results = pd.DataFrame(records)
    results.to_csv(os.path.join(args.out_dir, f"{slug}_scores.csv"), index=False)
    pd.DataFrame(manifest).to_csv(os.path.join(args.out_dir, f"{slug}_image_manifest.csv"), index=False)
    run_publishable_analysis(results, args.label, args.out_dir)
    print(f"Saved {len(results)} real image scores for {args.label}.")


def main(default_model=None, default_label=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="results/met_metadata_enriched.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--model-name", default=default_model, required=default_model is None)
    parser.add_argument("--label", default=default_label, required=default_label is None)
    parser.add_argument("--n-female", type=int)
    parser.add_argument("--n-male", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
