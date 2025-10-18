import csv
import cv2
import os
import numpy as np

def compute_mean_var(directory):
    means = []
    variances = []
    count = 0
    with open(directory, "r") as file:
        root_path = "/media/mprl2/Hard Disk/zwl/gazedata"
        reader = csv.reader(file)
        for row in reader:
            folder_path = row[0]
            subpath = os.path.join(root_path, folder_path)

            for root, dirs, files in os.walk(subpath):
                for file in files:
                    if (file.endswith(".jpg") or file.endswith(".png")) and "cali" not in file:
                        print(file)
                        img = cv2.imread(os.path.join(root, file))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img / 255.0  # Normalize the image
                        means.append(np.mean(img, axis=(0, 1)))
                        variances.append(np.std(img, axis=(0, 1)))
                        count += 1
    print("Number of images: ", count)
    return np.mean(means, axis=0), np.mean(variances, axis=0)


if __name__ == "__main__":
    directory = "/media/mprl2/Hard Disk/zwl/gazedata/9_fold_heat1_u.csv"
    means, variances = compute_mean_var(directory)
    print("Means: ", means)
    print("Variances: ", variances)
