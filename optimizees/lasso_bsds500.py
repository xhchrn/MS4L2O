'''
Please download BSDS500 dataset from:
https://github.com/BIDS/BSDS500.git
to the folder "./BSDS500/", then run this script.
Then you may find 1000 LASSO instances in "./matdata/lasso-real/"

To run this script, we have to install:
pip install ksvd
conda install -c anaconda scikit-image
'''
from ksvd import ApproximateKSVD
from sklearn.preprocessing import normalize
from skimage import color
from skimage import io
from os import listdir, makedirs
import random
import numpy as np
import scipy.io as sio

raw_image_pth = './BSDS500/BSDS500/data/images/test/'
imgpaths = []
for f in listdir(raw_image_pth):
	if f[-3:] == '.db':
		continue
	imgpaths.append(raw_image_pth + f)
generated_data_pth = './matdata/lasso-real/'
makedirs(generated_data_pth, exist_ok=True)

'''
Each patch is of 8*8.
After flattening, each patch is a 64-dim signal.
We use 128 kernels in KSVD, then the dictionary is of size 64*128.
'''
test_size = 1000
batch_size = 50
patch_size = 8
num_dict = 128

random.seed(0)

## Randomly sample patches from the test set and flatten those patches
patches = np.zeros((test_size, patch_size**2))
print(patches.shape)

for i in range(test_size):
    imgpath = random.sample(imgpaths, 1)[0]
    img = color.rgb2gray(io.imread(imgpath))

    x_max = img.shape[0] - patch_size
    y_max = img.shape[1] - patch_size
    x_start = random.sample([ii for ii in range(x_max)], 1)[0]
    y_start = random.sample([ii for ii in range(y_max)], 1)[0]
    x_end = x_start + patch_size
    y_end = y_start + patch_size

    patch = img[x_start:x_end, y_start:y_end]
    vmax = np.max(patch)
    vmin = np.min(patch)
    patch = (patch - vmin) / (vmax -vmin + 1e-3)
    patch = patch - np.mean(patch)

    patches[i] = patch.reshape(-1)

## Call KSVD to obtain a dictionary for sparse coding
aksvd = ApproximateKSVD(n_components=num_dict)
dictionary = aksvd.fit(patches).components_
x = aksvd.transform(patches)
print(dictionary.shape, x.shape, patches.shape)

## Process the dictionary and generate 1000 LASSO instances for testing
dictionary = normalize(dictionary, axis=1, norm = 'l2')
W = np.expand_dims(dictionary.T, 0).repeat(batch_size, axis=0)
Y = np.expand_dims(patches, -1)
rho = 0.5
num_batches = Y.shape[0] // batch_size
for b in range(num_batches):
    sio.savemat(generated_data_pth + str(b) + ".mat", {'W':W,'Y':Y[b*batch_size:(b+1)*batch_size],'rho':rho})
