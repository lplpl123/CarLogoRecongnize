import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import torch


def ReadData():

    # read csv
    structure_df = pd.read_csv("./data/vehicle-logos/vehicle-logos-dataset-master/structure.csv")

    brand_list = structure_df["Template Name"].tolist()
    file_list = structure_df["Mask"].tolist()

    only_one_brand_list = list(set(brand_list))

    # print(brand_list)
    # print(only_one_brand_list)
    # print(file_list)

    # end read csv
    # read image and transfer to matrix

    file_folder = "./data/vehicle-logos/vehicle-logos-dataset-master"

    X_original = []
    Y_original = []
    count = 0

    for brand in brand_list:
        index = only_one_brand_list.index(brand)

        Y_original.append(index)

    for file_name in file_list:

        file_path = os.path.join(file_folder, file_name)

        image = Image.open(file_path)

        image = image.resize((32, 32))
        image = image.convert('RGB')


        matrix = np.array(image).tolist()
        fixed_matrix = []
        for i in range(3):
            first_matrix = []
            for index_0 in matrix:
                second_matrix = []
                for index_1 in index_0:
                    second_matrix.append(float(index_1[i]))
                first_matrix.append(second_matrix)
            fixed_matrix.append(first_matrix)

        tensor_matrix = torch.tensor(fixed_matrix)

        single_data = (tensor_matrix, Y_original[count])
        count += 1
        X_original.append(single_data)


    # print("X")
    # print(X_original.shape)
    #
    # print("Y")
    # print(Y_original.shape)

    return X_original, only_one_brand_list
