import os
import zipfile
import numpy as np
import imageio.v2 as io
from PIL import Image

from tqdm import tqdm


def extract_zip_data(input_path='zip_data/', output_path='data_extracted/'):
    # extracts zip archives with data
    for file in os.listdir(input_path):
        print(file[:-4])
        with zipfile.ZipFile(input_path + file, 'r') as zip_ref:
            zip_ref.extractall(output_path + file[:-4])


def crop_images(extracted_data_path='data_extracted', output_data_path='data'):
    # transforms 576x576 images to 4x256x256 images
    for folder in os.listdir(extracted_data_path):
        target = folder.split('-')[0]
        if not os.path.exists(output_data_path + '/' + folder):
            os.mkdir(output_data_path + '/' + folder)
        for file in tqdm(os.listdir(extracted_data_path + '/' + folder)):
            image = np.array(io.imread('/'.join([extracted_data_path, folder, file])))
            side = int(image.shape[0] // 2)
            image1 = image[:side, :side][16:-16, 16:-16]
            image2 = image[:side, side:][16:-16, 16:-16]
            image3 = image[side:, :side][16:-16, 16:-16]
            image4 = image[side:, side:][16:-16, 16:-16]
            Image.fromarray(image1).save('/'.join([output_data_path, folder, file[:-8] + '1-' + file[-8:]]))
            Image.fromarray(image2).save('/'.join([output_data_path, folder, file[:-8] + '2-' + file[-8:]]))
            Image.fromarray(image3).save('/'.join([output_data_path, folder, file[:-8] + '3-' + file[-8:]]))
            Image.fromarray(image4).save('/'.join([output_data_path, folder, file[:-8] + '4-' + file[-8:]]))


def main():
    extract_zip_data()
    crop_images()


if __name__ == "__main__":
    main()