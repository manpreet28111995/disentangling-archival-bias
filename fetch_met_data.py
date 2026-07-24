"""
fetch_met_data.py

Pulls object metadata from the Metropolitan Museum of Art's Open Access API
(no API key required) across multiple departments & queries, saving to CSV.

Docs: https://metmuseum.github.io/

Usage:
    python fetch_met_data.py --departments 1 6 8 9 11 12 15 19 21 --max-objects 4000 --out met_metadata.csv
"""

import argparse
import time
import json
import os
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
CACHE_DIR = ".met_cache"

SEARCH_QUERIES = ["painting", "drawing", "print", "photograph", "sculpture", "woman", "female", "portrait", "art", "artist"]


def get_object_ids(department_ids, queries=SEARCH_QUERIES, has_images_only=True):
    """Return a list of unique objectIDs across departments and queries."""
    all_ids = set()
    session = requests.Session()
    for dept in department_ids:
        for q in queries:
            params = {"departmentId": dept, "q": q}
            if has_images_only:
                params["hasImages"] = "true"
            try:
                resp = session.get(f"{BASE}/search", params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    ids = data.get("objectIDs") or []
                    all_ids.update(ids)
            except Exception as e:
                print(f"Error searching dept {dept} with query '{q}': {e}")
    print(f"Total unique candidate objects found across {len(department_ids)} departments: {len(all_ids)}")
    return list(all_ids)


def fetch_object(object_id, cache_dir=CACHE_DIR):
    """Fetch a single object's metadata with local disk caching."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{object_id}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    try:
        resp = requests.get(f"{BASE}/objects/{object_id}", timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
        with open(cache_path, "w") as f:
            json.dump(data, f)
        return data
    except Exception:
        return None


def extract_row(obj):
    if not obj or not obj.get("primaryImage"):
        return None
    return {
        "objectID": obj.get("objectID"),
        "title": obj.get("title"),
        "department": obj.get("department"),
        "classification": obj.get("classification"),
        "culture": obj.get("culture"),
        "medium": obj.get("medium"),
        "objectDate": obj.get("objectDate"),
        "objectBeginDate": obj.get("objectBeginDate"),
        "objectEndDate": obj.get("objectEndDate"),
        "artistDisplayName": obj.get("artistDisplayName"),
        "artistNationality": obj.get("artistNationality"),
        "artistGender": obj.get("artistGender"),
        "artistBeginDate": obj.get("artistBeginDate"),
        "artistEndDate": obj.get("artistEndDate"),
        "primaryImage": obj.get("primaryImage"),
        "primaryImageSmall": obj.get("primaryImageSmall"),
        "tags": ";".join(t["term"] for t in (obj.get("tags") or []) if "term" in t),
        "creditLine": obj.get("creditLine"),
        "isHighlight": obj.get("isHighlight"),
    }


def build_dataframe(object_ids, max_objects=4000, num_workers=10):
    if max_objects:
        object_ids = object_ids[:max_objects]

    rows = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fetch_object, oid): oid for oid in object_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching object metadata"):
            obj = future.result()
            row = extract_row(obj)
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--departments", nargs="+", type=int, default=[1, 6, 8, 9, 11, 12, 15, 19, 21],
                         help="Met department IDs to pull from")
    parser.add_argument("--max-objects", type=int, default=4000,
                         help="Cap on number of candidate objects to check")
    parser.add_argument("--out", type=str, default="met_metadata.csv")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    object_ids = get_object_ids(args.departments)
    df = build_dataframe(object_ids, max_objects=args.max_objects, num_workers=args.workers)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} objects with images to {args.out}")


if __name__ == "__main__":
    main()

