import os
import sys
import io
import zipfile
import random
import urllib.request
from PIL import Image

import shutil

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset', 'liver'))
SPLITS = ['train', 'validation', 'test']
CLASSES = ['normal', 'abnormal']

def setup_directories():
    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT, ignore_errors=True)
    for split in SPLITS:
        for cls in CLASSES:
            p = os.path.join(DATASET_ROOT, split, cls)
            os.makedirs(p, exist_ok=True)
    print(f"[INFO] Initialized clean liver directory structure at {DATASET_ROOT}")

def download_and_extract_all():
    setup_directories()
    random.seed(42)

    print("[INFO] Downloading Liver ultrasound dataset archive (xmidy/CNN_LIVER_FIBROSIS)...")
    url = 'https://github.com/xmidy/CNN_LIVER_FIBROSIS/archive/refs/heads/main.zip'
    req = urllib.request.Request(url, headers={'User-Agent': 'MultiOrganAI-Liver'})
    
    with urllib.request.urlopen(req, timeout=60) as response:
        archive_data = response.read()
        print(f"[INFO] Downloaded archive: {len(archive_data)} bytes. Extracting images...")
        z = zipfile.ZipFile(io.BytesIO(archive_data))

    all_names = z.namelist()
    
    # Filter F0 (Normal) and F4 (Abnormal / Severe Fibrosis)
    f0_files = [n for n in all_names if ('/F0/' in n or '\\F0\\' in n) and n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    f4_files = [n for n in all_names if ('/F4/' in n or '\\F4\\' in n) and n.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"[INFO] Found {len(f0_files)} F0 (Normal) images and {len(f4_files)} F4 (Abnormal) images in archive.")

    random.shuffle(f0_files)
    random.shuffle(f4_files)

    # 80 train, 10 validation, 10 test per class (200 total verified images)
    target_count = 100
    selected_f0 = f0_files[:target_count]
    selected_f4 = f4_files[:target_count]

    # Partition F0 (Normal)
    f0_train = selected_f0[:80]
    f0_val = selected_f0[80:90]
    f0_test = selected_f0[90:100]

    for name in selected_f0:
        split = 'train' if name in f0_train else ('validation' if name in f0_val else 'test')
        base = os.path.basename(name).lower()
        if not base.endswith('.jpg'):
            base = f"{os.path.splitext(base)[0]}.jpg"
        base = f"normal_liver_{base}"
        target = os.path.join(DATASET_ROOT, split, 'normal', base)
        with z.open(name) as f:
            img = Image.open(f).convert('RGB')
            img.save(target, 'JPEG', quality=95)

    print(f"  [+] Saved Normal liver images: {len(f0_train)} train, {len(f0_val)} val, {len(f0_test)} test")

    # Partition F4 (Abnormal / Fibrosis)
    f4_train = selected_f4[:80]
    f4_val = selected_f4[80:90]
    f4_test = selected_f4[90:100]

    for name in selected_f4:
        split = 'train' if name in f4_train else ('validation' if name in f4_val else 'test')
        base = os.path.basename(name).lower()
        if not base.endswith('.jpg'):
            base = f"{os.path.splitext(base)[0]}.jpg"
        base = f"abnormal_liver_{base}"
        target = os.path.join(DATASET_ROOT, split, 'abnormal', base)
        with z.open(name) as f:
            img = Image.open(f).convert('RGB')
            img.save(target, 'JPEG', quality=95)

    print(f"  [+] Saved Abnormal liver images: {len(f4_train)} train, {len(f4_val)} val, {len(f4_test)} test")

    print("\n[INFO] Final Liver Dataset Partitioning:")
    total = 0
    for split in SPLITS:
        for cls in CLASSES:
            p = os.path.join(DATASET_ROOT, split, cls)
            cnt = len(os.listdir(p))
            total += cnt
            print(f"  - {split}/{cls}: {cnt} images")
    print(f"[INFO] Total verified liver ultrasound images: {total}")

if __name__ == '__main__':
    download_and_extract_all()
