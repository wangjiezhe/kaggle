## Generated by ChatGPT 3.5 and modified

import hashlib
import os
from collections import defaultdict

import imagehash
from PIL import Image


def calculate_file_hashes(directory):
    file_hashes = defaultdict(list)

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                file_path = os.path.join(root, filename)
                hash_value = calculate_md5(file_path)
                file_hashes[hash_value].append(file_path)

    return file_hashes


def calculate_image_hashes(directory):
    image_hashes = defaultdict(list)

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                file_path = os.path.join(root, filename)
                image = Image.open(file_path)
                image_hash = imagehash.average_hash(image)
                image_hashes[image_hash].append(file_path)

    return image_hashes


def calculate_md5(file_path):
    md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def find_duplicate_files(directory):
    image_hashes = calculate_file_hashes(directory)
    duplicate_images = []

    for _, file_paths in image_hashes.items():
        if len(file_paths) > 1:
            duplicate_images.append(file_paths)

    return duplicate_images


def find_duplicate_images(directory):
    image_hashes = calculate_image_hashes(directory)
    duplicate_images = []

    for _, file_paths in image_hashes.items():
        if len(file_paths) > 1:
            duplicate_images.append(file_paths)

    return duplicate_images


if __name__ == "__main__":
    # 替换为你的图片文件夹的父目录路径
    parent_directory = "../data/images_cleaned_train"

    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.join(current_directory, parent_directory)

    duplicate_images = find_duplicate_images(parent_directory)

    if duplicate_images:
        print("重复图片发现:")
        for image_path_list in duplicate_images:
            for image_path in image_path_list:
                print(image_path)
            print()
    else:
        print("未发现重复图片。")