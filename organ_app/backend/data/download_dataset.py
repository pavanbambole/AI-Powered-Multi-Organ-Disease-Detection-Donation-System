import os
import sys
import io
import shutil
import hashlib
import zipfile
import random
import urllib.request
from PIL import Image

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset', 'kidney'))
SPLITS = ['train', 'validation', 'test']
CLASSES = ['normal', 'stone']

def setup_directories():
    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)
    for split in SPLITS:
        for cls in CLASSES:
            p = os.path.join(DATASET_ROOT, split, cls)
            os.makedirs(p, exist_ok=True)
    print(f"[INFO] Initialized clean directory structure at {DATASET_ROOT}")

def download_and_extract_all():
    setup_directories()
    random.seed(42)

    # 1. Download & Deduplicate Normal images from main branch zip
    print("[INFO] Downloading Normal kidney ultrasound images (main.zip)...")
    url_main = 'https://github.com/saraswathi05/kidney_dataset/archive/refs/heads/main.zip'
    req_main = urllib.request.Request(url_main, headers={'User-Agent': 'MultiOrganAI'})
    with urllib.request.urlopen(req_main, timeout=30) as r:
        z_main = zipfile.ZipFile(io.BytesIO(r.read()))

    normal_files = [n for n in z_main.namelist() if n.upper().endswith('.JPG') and 'NORMAL' in n.upper()]
    
    # Deduplicate normal by MD5
    unique_normal = []
    normal_hashes = set()
    for n in sorted(normal_files):
        content = z_main.read(n)
        h = hashlib.md5(content).hexdigest()
        if h not in normal_hashes:
            normal_hashes.add(h)
            unique_normal.append(n)
    
    print(f"[INFO] Total Normal images: {len(normal_files)} | Unique Normal: {len(unique_normal)}")
    random.shuffle(unique_normal)
    selected_normal = unique_normal[:45]

    n_train = selected_normal[:32]
    n_val = selected_normal[32:38]
    n_test = selected_normal[38:45]

    for name in selected_normal:
        split = 'train' if name in n_train else ('validation' if name in n_val else 'test')
        base = os.path.basename(name).lower()
        target = os.path.join(DATASET_ROOT, split, 'normal', base)
        with z_main.open(name) as f:
            img = Image.open(f).convert('RGB')
            img.save(target, 'JPEG')

    print(f"  [+] Saved {len(selected_normal)} unique Normal images: {len(n_train)} train, {len(n_val)} val, {len(n_test)} test")

    # 2. Download & Deduplicate Stone images from abnormal branch zip
    print("[INFO] Downloading Stone kidney ultrasound images (abnormal.zip)...")
    url_abn = 'https://github.com/saraswathi05/kidney_dataset/archive/refs/heads/abnormal.zip'
    req_abn = urllib.request.Request(url_abn, headers={'User-Agent': 'MultiOrganAI'})
    with urllib.request.urlopen(req_abn, timeout=30) as r:
        z_abn = zipfile.ZipFile(io.BytesIO(r.read()))

    stone_files = [n for n in z_abn.namelist() if n.upper().endswith('.JPG') and 'STONE' in n.upper()]
    
    # Deduplicate stone by MD5
    unique_stone = []
    stone_hashes = set()
    for n in sorted(stone_files):
        content = z_abn.read(n)
        h = hashlib.md5(content).hexdigest()
        if h not in stone_hashes:
            stone_hashes.add(h)
            unique_stone.append(n)

    print(f"[INFO] Total Stone images: {len(stone_files)} | Unique Stone: {len(unique_stone)}")
    random.shuffle(unique_stone)
    selected_stone = unique_stone[:45]

    s_train = selected_stone[:32]
    s_val = selected_stone[32:38]
    s_test = selected_stone[38:45]

    for name in selected_stone:
        split = 'train' if name in s_train else ('validation' if name in s_val else 'test')
        base = os.path.basename(name).lower()
        target = os.path.join(DATASET_ROOT, split, 'stone', base)
        with z_abn.open(name) as f:
            img = Image.open(f).convert('RGB')
            img.save(target, 'JPEG')

    print(f"  [+] Saved {len(selected_stone)} unique Stone images: {len(s_train)} train, {len(s_val)} val, {len(s_test)} test")

    print("\n[INFO] Final Deduplicated Balanced Dataset Partitioning:")
    total = 0
    for split in SPLITS:
        for cls in CLASSES:
            p = os.path.join(DATASET_ROOT, split, cls)
            cnt = len(os.listdir(p))
            total += cnt
            print(f"  - {split}/{cls}: {cnt} images")
    print(f"[INFO] Total verified unique images: {total}")

if __name__ == '__main__':
    download_and_extract_all()

