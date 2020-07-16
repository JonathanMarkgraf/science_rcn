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
import time
import pickle
import shutil


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

    if not os.path.exists("./trained.pickle"):
        LOG.info("Training on {} images...".format(len(train_data)))
        train_partial = partial(train_image,
                                perturb_factor=perturb_factor)
        train_results = pool.map_async(train_partial, [d[0] for d in train_data]).get(9999999)
        all_model_factors = zip(*train_results)
        with open('./trained.pickle', 'wb') as f:
            pickle.dump(all_model_factors, f)
    else:
        LOG.info("trained.pickle detected, skipping training")
        with open('./trained.pickle', 'rb') as f:
            all_model_factors = pickle.load(f)

    LOG.info("Testing on {} images...".format(len(test_data)))
    t0 = time.time()
    test_partial = partial(test_image, model_factors=all_model_factors,
                           pool_shape=pool_shape)
    test_results = pool.map_async(test_partial, [d[0] for d in test_data]).get(9999999)
    t1 = time.time() - t0

    # Evaluate result
    correctDigits   = 0
    captcha         = []    # [(guessed digit, real digit)]
    captchaList     = []
    for test_i, (winner_i, confidence) in enumerate(test_results):
        captcha.append((train_data[winner_i][1], (test_data[test_i][1]).upper()))
        if len(captcha) == 6:
            captchaList.append(captcha)
            captcha = []
        print "guess:\t{}\nscore:\t{}\nreal:\t{}\n".format(train_data[winner_i][1], confidence, (test_data[test_i][1]).upper())
        if train_data[winner_i][1] == (test_data[test_i][1]).upper():
            correctDigits = correctDigits+1
    correctCaptchas = 0
    for cap in captchaList:
        match = True
        for t in cap:
            if t[0] == t[1]:
                continue
            else:
                match = False
                break
        correctCaptchas = correctCaptchas+1 if match else correctCaptchas
    # print test_data
    print "Tested captchas: {}".format(len(test_data)/6)
    print "Time per captcha: {0:0.2f} seconds".format(t1/(len(test_data)/6))
    print "Digits guessed correct: {0:0.2f}%".format(float(correctDigits)/len(test_data)*100)
    print "Captchas guessed correct: {}".format(correctCaptchas)

    return all_model_factors, test_results


# returns (traindata, testdata) tuple
# only works for our cuts and shadowcross folders for now
def get_botdetect_data_iters(data_dir):
    if not os.path.isdir(data_dir):
        raise IOError("Can't find your data dir '{}'".format(data_dir))

    # returns list of (symbolImg: nparray, symbol: string) tuples, imgages are black and white
    def _clean_and_split_captcha(path, fileName):
        captchaPath = os.path.join(path, fileName)
        fname       = os.path.splitext(fileName)[0]
        debugDir    = "./debug/{}".format(fname)
        if os.path.exists(debugDir):
            shutil.rmtree(debugDir)
        os.makedirs(debugDir)
        img         = Image.open(captchaPath)
        img         = img.convert('L')                              # grayscale
        img         = img.point(lambda x: 0 if x>200 else 255, 'L') # map grayscale to black and white
        # crop image to text
        pixels      = img.load()
        xStart      = img.size[0]
        yStart      = None
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                if (pixels[x, y] == 255):
                    xStart = x if x < xStart else xStart
                    yStart = y if yStart is None else yStart
                    break
        xEnd = 0
        yEnd = None
        for y in range(img.size[1]-1, -1, -1):
            for x in range(img.size[0]-1, -1, -1):
                if (pixels[x, y] == 255):
                    xEnd = x if x > xEnd else xEnd
                    yEnd = y if yEnd is None else yEnd
                    break
        img = img.crop((xStart, yStart, xEnd, yEnd))
        img.save("./debug/{}/captcha.bmp".format(fname))
        # split symbols
        res = []
        symbolWidth = int(img.size[0]/6)
        captchaLen = 6
        for i in range(captchaLen):
            if i == 0:
                sImg = img.crop((symbolWidth*i, 0, symbolWidth*(i+1)+6, img.size[1]))
            elif i < captchaLen:
                sImg = img.crop((symbolWidth*i-3, 0, symbolWidth*(i+1)+3, img.size[1]))
            else:
                sImg = img.crop((symbolWidth*i-6, 0, symbolWidth*(i+1), img.size[1]))
            
            sImg = sImg.resize((100, 100), Image.BILINEAR)
            padded = Image.new('L', (200, 200), 0)
            padded.paste(sImg, (int((padded.size[0]-sImg.size[0])/2), int((padded.size[1]-sImg.size[1])/2))) # insert into center
            padded.save("./debug/{}/{}_{}.bmp".format(fname, i, fileName[i]))
            res.append((np.asarray(padded), fileName[i]))
            sImg.close()
            padded.close()
        img.close()

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
                # image_arr = imresize(imread(filepath), (200, 200))
                # loaded_data.append((image_arr, category))
                img = Image.open(filepath)
                img = img.convert('L')
                img = img.point(lambda x: 0 if x>128 else 255, 'L')
                img = img.resize((150, 150), Image.BILINEAR)
                padded = Image.new('L', (200, 200), 0)
                padded.paste(img, (int((padded.size[0]-img.size[0])/2), int((padded.size[1]-img.size[1])/2))) # insert into center
                loaded_data.append((np.asarray(padded), category))
                padded.close()
                img.close()
        return loaded_data

    def _load_test_data(image_dir, get_filenames=False):
        loaded_data = []
        for captchaFile in os.listdir(image_dir):
            if len(loaded_data) > 10:
                break
            samples = _clean_and_split_captcha(image_dir, captchaFile)

            for t in samples:
                loaded_data.append(t)
        return loaded_data

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













#####################################################################################
def splitCaptcha(img):
    img         = img.convert('L')                              # grayscale
    img         = img.point(lambda x: 0 if x>200 else 255, 'L') # map grayscale to black and white
    # crop image to text
    pixels      = img.load()
    xStart      = img.size[0]
    yStart      = None
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if (pixels[x, y] == 255):
                xStart = x if x < xStart else xStart
                yStart = y if yStart is None else yStart
                break
    xEnd = 0
    yEnd = None
    for y in range(img.size[1]-1, -1, -1):
        for x in range(img.size[0]-1, -1, -1):
            if (pixels[x, y] == 255):
                xEnd = x if x > xEnd else xEnd
                yEnd = y if yEnd is None else yEnd
                break
    img = img.crop((xStart, yStart, xEnd, yEnd))
    # split symbols
    res = []
    symbolWidth = int(img.size[0]/6)
    captchaLen = 6
    for i in range(captchaLen):
        if i == 0:
            sImg = img.crop((symbolWidth*i, 0, symbolWidth*(i+1)+6, img.size[1]))
        elif i < captchaLen:
            sImg = img.crop((symbolWidth*i-3, 0, symbolWidth*(i+1)+3, img.size[1]))
        else:
            sImg = img.crop((symbolWidth*i-6, 0, symbolWidth*(i+1), img.size[1]))        
        sImg = sImg.resize((100, 100), Image.BILINEAR)
        padded = Image.new('L', (200, 200), 0)
        padded.paste(sImg, (int((padded.size[0]-sImg.size[0])/2), int((padded.size[1]-sImg.size[1])/2))) # insert into center
        res.append((np.asarray(padded), ''))
        sImg.close()
        padded.close()        
    img.close()
    return res


# img = PIL Image
# returns captcha guess as string
def readCaptcha(img):
    data = splitCaptcha(img)
    with open('./trained.pickle', 'rb') as f:
        all_model_factors = pickle.load(f)        
    test = partial(test_image, model_factors=all_model_factors)
    pool = Pool(None)
    testResults = pool.map_async(test, [d[0] for d in data]).get(9999999)
    pool.close()
    pool.join()

    # captchaAlphabet = "26abcdefghkmnprstuvxz"
    captchaAlphabet = "222222666666aaaaaabbbbbbccccccddddddeeeeeeffffffgggggghhhhhhkkkkkkmmmmmmnnnnnnpppppprrrrrrssssssttttttuuuuuuvvvvvvxxxxxxzzzzzz"
    result = ""
    for test_i, (winner_i, confidence) in enumerate(testResults):
        result = result + captchaAlphabet[winner_i]
    return result