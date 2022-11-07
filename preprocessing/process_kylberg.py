import argparse
import os
import zipfile

import imageio.v2 as io
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_output_path(output_path):
    """Checks if output path exists and if not creates output folders"""
    complete_path = ""
    for folder in output_path.split('/'):
        complete_path = os.path.join(complete_path, folder)
        if not os.path.exists(complete_path):
            os.mkdir(complete_path)


def extract_zip_data(input_path='zip_data', output_path='extracted_data'):
    """Extracts zip archives with data"""

    create_output_path(output_path)
    for file in tqdm(os.listdir(input_path)):
        with zipfile.ZipFile(os.path.join(input_path, file), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(output_path, file[:-4]))


def crop_images(data_path='extracted_data', output_path='cropped_data'):
    """Transforms 576x576 images to 4x256x256 images"""

    create_output_path(output_path)
    for folder in os.listdir(data_path):
        if not os.path.exists(os.path.join(output_path, folder)):
            os.mkdir(os.path.join(output_path, folder))
        for file in tqdm(os.listdir(os.path.join(data_path, folder))):
            image = np.array(io.imread(os.path.join(data_path, folder, file)))
            side = int(image.shape[0] // 2)

            image1 = image[:side, :side][16:-16, 16:-16]
            image2 = image[:side, side:][16:-16, 16:-16]
            image3 = image[side:, :side][16:-16, 16:-16]
            image4 = image[side:, side:][16:-16, 16:-16]

            Image.fromarray(image1).save(os.path.join(output_path, folder, f"{file[:-8]}1-{file[-8:]}"))
            Image.fromarray(image2).save(os.path.join(output_path, folder, f"{file[:-8]}2-{file[-8:]}"))
            Image.fromarray(image3).save(os.path.join(output_path, folder, f"{file[:-8]}3-{file[-8:]}"))
            Image.fromarray(image4).save(os.path.join(output_path, folder, f"{file[:-8]}4-{file[-8:]}"))

            os.remove(os.path.join(data_path, folder, file))
        os.rmdir(os.path.join(data_path, folder))
    os.rmdir(data_path)


def rename(data_path='cropped_data', output_path='data/dataset'):
    """
    Merges different target folders.
    Renames files into template <target_label>-<file_id>
    """

    counter = 0
    max_counter = 0
    for folder in os.listdir(data_path):
        max_counter += len(os.listdir(os.path.join(data_path, folder)))
    max_counter_len = len(str(max_counter))

    create_output_path(output_path)

    for folder in tqdm(os.listdir(data_path)):
        for file in os.listdir(os.path.join(data_path, folder)):
            new_name = f"{file.split('-')[0]}-{counter:0{max_counter_len}}.png"
            os.rename(os.path.join(data_path, folder, file), os.path.join(output_path, new_name))
            counter += 1
        os.rmdir(os.path.join(data_path, folder))
    os.rmdir(data_path)


def make_dataset_sheet(data_path='data/dataset', output_path='data'):
    """makes dataset.txt with a list of filenames in dataset"""

    with open(os.path.join(output_path, "dataset.txt"), 'w') as f:
        for filename in sorted(os.listdir(data_path)):
            f.write(filename + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        required=False, default='zip_data', help="path for downloaded data")
    parser.add_argument("--output_path", type=str,
                        required=False, default='data', help="path for downloaded data")
    return parser.parse_args()


def main():
    args = parse_args()

    extract_zip_data(input_path=args.input_path)
    crop_images()
    rename()
    make_dataset_sheet(output_path=args.output_path)


if __name__ == "__main__":
    main()
