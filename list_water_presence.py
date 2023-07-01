import os
import numpy as np
import glob

def scan_occurance(image: np.ndarray):
    return image.any()

if __name__ == "__main__":
    images = "./pub/Pytorch-UNet/data/test/masks/*.npy"

    names = []
    results = []

    for file in glob.glob(images):
        result = scan_occurance(np.load(file))
        names.append(os.path.basename(file))
        results.append(result)

    names = np.expand_dims(np.array(names), 1)
    results = np.expand_dims(np.array(results), 1)

    np.savetxt("./csv/water-presence.csv", np.hstack((names, results)), fmt='%s', delimiter=',', header="name,contains_water", comments='')


