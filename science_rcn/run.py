"""
Reference implementation of a two-level RCN model for MNIST classification experiments.

Examples:
- To run a small unit test that trains and tests on 20 images using one CPU 
  (takes ~2 minutes, accuracy is ~60%):
python science_rcn/run.py

- To run a slightly more interesting experiment that trains on 100 images and tests on 20 
  images using multiple CPUs (takes <1 min using 7 CPUs, accuracy is ~90%):
python science_rcn/run.py --train_size 100 --test_size 20 --parallel

- To test on the full 10k MNIST test set, training on 1000 examples 
(could take hours depending on the number of available CPUs, average accuracy is ~97.7+%):
python science_rcn/run.py --full_test_set --train_size 1000 --parallel --pool_shape 25 --perturb_factor 2.0
"""

import argparse
import logging
import numpy as np
import os
from multiprocessing import Pool
from functools import partial
from scipy.misc import imresize
from scipy.ndimage import imread

from science_rcn.inference import test_image
from science_rcn.learning import train_image

from PIL import Image

LOG = logging.getLogger(__name__)


def run_experiment(data_dir='data/botdetect',
                   pool_shape=(25, 25),
                   perturb_factor=2.,
                   parallel=True,
                   verbose=False,):
    """Run MNIST experiments and evaluate results. 

    Parameters
    ----------
    data_dir : string
        Dataset directory.
    pool_shape : (int, int)
        Vertical and horizontal pool shapes.
    perturb_factor : float
        How much two points are allowed to vary on average given the distance
        between them. See Sec S2.3.2 for details.
    parallel : bool
        Parallelize over multiple CPUs.
    verbose : bool
        Higher verbosity level.
    
    Returns
    -------
    model_factors : ([numpy.ndarray], [numpy.ndarray], [networkx.Graph])
        ([frcs], [edge_factors], [graphs]), outputs of train_image in learning.py.
    test_results : [(int, float)]
        List of (winner_idx, winner_score), outputs of test_image in inference.py.
    """
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    # Multiprocessing set up
    num_workers = None if parallel else 1
    pool = Pool(num_workers)

    train_data, test_data = get_botdetect_data_iters(data_dir)

    LOG.info("Training on {} images...".format(len(train_data)))
    train_partial = partial(train_image,
                            perturb_factor=perturb_factor)
    train_results = pool.map_async(train_partial, [d[0] for d in train_data]).get(9999999)
    all_model_factors = zip(*train_results)

    LOG.info("Testing on {} images...".format(len(test_data)))
    test_partial = partial(test_image, model_factors=all_model_factors,
                           pool_shape=pool_shape)
    test_results = pool.map_async(test_partial, [d[0] for d in test_data]).get(9999999)

    # Evaluate result
    correct = 0
    for test_idx, (winner_idx, _) in enumerate(test_results):
        correct += int(test_data[test_idx][1]) == winner_idx // train_size
    print "Total test accuracy = {}".format(float(correct) / len(test_results))

    return all_model_factors, test_results


def get_mnist_data_iters(data_dir, train_size, test_size,
                         full_test_set=False, seed=5):
    """
    Load MNIST data.

    Assumed data directory structure:
        training/
            0/
            1/
            2/
            ...
        testing/
            0/
            ...

    Parameters
    ----------
    train_size, test_size : int
        MNIST dataset sizes are in increments of 10
    full_test_set : bool
        Test on the full MNIST 10k test set.
    seed : int
        Random seed used by numpy.random for sampling training set.

    Returns
    -------
    train_data, train_data : [(numpy.ndarray, str)]
        Each item reps a data sample (2-tuple of image and label)
        Images are numpy.uint8 type [0,255]
    """
    if not os.path.isdir(data_dir):
        raise IOError("Can't find your data dir '{}'".format(data_dir))

    def _load_data(image_dir, num_per_class, get_filenames=False):
        loaded_data = []
        for category in sorted(os.listdir(image_dir)):
            cat_path = os.path.join(image_dir, category)
            if not os.path.isdir(cat_path) or category.startswith('.'):
                continue
            if num_per_class is None:
                samples = sorted(os.listdir(cat_path))
            else:
                samples = np.random.choice(sorted(os.listdir(cat_path)), num_per_class)

            for fname in samples:
                filepath = os.path.join(cat_path, fname)
                # Resize and pad the images to (200, 200)
                image_arr = imresize(imread(filepath), (112, 112))
                img = np.pad(image_arr,
                             pad_width=tuple([(p, p) for p in (44, 44)]),
                             mode='constant', constant_values=0)
                loaded_data.append((img, category))
        return loaded_data

    np.random.seed(seed)
    train_set = _load_data(os.path.join(data_dir, 'training'),
                           num_per_class=train_size // 10)
    test_set = _load_data(os.path.join(data_dir, 'testing'),
                          num_per_class=None if full_test_set else test_size // 10)
    return train_set, test_set

# returns (traindata, testdata) tuple
# only works for our cuts and shadowcross folders for now
def get_botdetect_data_iters(data_dir):
    if not os.path.isdir(data_dir):
        raise IOError("Can't find your data dir '{}'".format(data_dir))

    # returns list of (symbolImg: nparray, symbol: string) tuples, imgages are black and white
    def _clean_and_split_captcha(path, fileName):
        captchaPath = os.path.join(path, fileName)
        img     = Image.open(captchaPath)
        img     = img.convert('L')                              # grayscale
        img     = img.point(lambda x: 0 if x>200 else 255, '1') # map grayscale to black and white
        img.save("./test.bmp")
        pixels  = img.load()
        # crop image to text
        xStart  = img.size[0]
        yStart  = None
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                if (pixels[x, y] == 0):
                    xStart = x if x < xStart else xStart
                    yStart = y if yStart is None else yStart
                    break
        xEnd = 0
        yEnd = None
        for y in range(img.size[1]-1, -1, -1):
            for x in range(img.size[0]-1, -1, -1):
                if (pixels[x, y] == 0):
                    xEnd = x if x > xEnd else xEnd
                    yEnd = y if yEnd is None else yEnd
                    break
        img = img.crop((xStart, yStart, xEnd, yEnd))
        # split symbols
        res = []
        # +1 pixel width for tolerance
        symbolWidth = int(img.size[0]/6)
        # our captchas contain 6 letters
        for i in range(6):
            # match dimensions of training data symbols
            sImg = img.crop((symbolWidth*i, 0, symbolWidth*(i+1), img.size[1])).resize((112, 112))
            bg = Image.new('L', (200, 200), 0)
            bg.paste(sImg, (int((bg.size[0]-sImg.size[0])/2), int((bg.size[1]-sImg.size[1])/2))) # insert into center
            res.append((np.array(bg), fileName[i]))
        return res

    def _load_train_data(image_dir, get_filenames=False):
        loaded_data = []
        for category in sorted(os.listdir(image_dir)):  # category => symbol folder name => symbol
            cat_path = os.path.join(image_dir, category)
            if not os.path.isdir(cat_path) or category.startswith('.'):
                continue
            
            samples = sorted(os.listdir(cat_path))

            for fname in samples:
                filepath = os.path.join(cat_path, fname)
                # Resize and pad the images to (200, 200)
                image_arr = imresize(imread(filepath), (112, 112))
                img = np.pad(image_arr,
                             pad_width=tuple([(p, p) for p in (44, 44)]),
                             mode='constant', constant_values=0)
                loaded_data.append((img, category))
        return loaded_data

    def _load_test_data(image_dir, get_filenames=False):
        loaded_data = []
        for captchaFile in os.listdir(image_dir):
            samples = _clean_and_split_captcha(image_dir, captchaFile)

            for t in sorted(samples, key=lambda tup: tup[1]):   # sort tuples after symbol/category
                loaded_data.append(t)
        return sorted(loaded_data, key=lambda tup: tup[1])

    train_set = _load_train_data(os.path.join(data_dir, 'training'))
    test_set = _load_test_data(os.path.join(data_dir, 'testing/cut'))
    return train_set, test_set

if __name__ == '__main__':
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_size',
        dest='train_size',
        type=int,
        default=20,
        help="Number of training examples.",
    )
    parser.add_argument(
        '--test_size',
        dest='test_size',
        type=int,
        default=20,
        help="Number of testing examples.",
    )
    parser.add_argument(
        '--full_test_set',
        dest='full_test_set',
        action='store_true',
        default=False,
        help="Test on full MNIST test set.",
    )
    parser.add_argument(
        '--pool_shapes',
        dest='pool_shape',
        type=int,
        default=25,
        help="Pool shape.",
    )
    parser.add_argument(
        '--perturb_factor',
        dest='perturb_factor',
        type=float,
        default=2.,
        help="Perturbation factor.",
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        default=5,
        help="Seed for numpy.random to sample training and testing dataset split.",
    )
    parser.add_argument(
        '--parallel',
        dest='parallel',
        default=False,
        action='store_true',
        help="Parallelize over multi-CPUs if True.",
    )
    parser.add_argument(
        '--verbose',
        dest='verbose',
        action='store_true',
        default=False,
        help="Verbosity level.",
    )
    options = parser.parse_args()
    run_experiment(pool_shape=(options.pool_shape, options.pool_shape),
                   perturb_factor=options.perturb_factor,
                   verbose=options.verbose,
                   parallel=options.parallel)
