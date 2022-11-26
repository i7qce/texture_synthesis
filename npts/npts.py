import numpy as np
import matplotlib.pyplot as plt
import argparse

import cv2

# reference for debugging?
# https://github.com/goldbema/TextureSynthesis/blob/master/synthesis.py

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--input', help='input file path')
    parser.add_argument('-o', '--output', help='output file destination')
    parser.add_argument('-w', '--winsize', help='window size')
    parser.add_argument('-sm', '--seedmode', help='seed mode')
    parser.add_argument('-os', '--outputsize', help='output file size')

    args = parser.parse_args()

    return args

def prep_img(fil, outsize, seed_type):
    a = cv2.imread(fil)
    a = a/255
    b = np.zeros((outsize, outsize, 3))
    c = np.zeros((outsize, outsize, 1))


    if seed_type == 0:
        b[0:a.shape[0], 0:a.shape[1],:] = a
        c[0:a.shape[0], 0:a.shape[1]] = 1
    elif seed_type == 1:
        offset = int(outsize/2) - int(a.shape[0]/2) 
        b[offset:a.shape[0]+offset, offset:a.shape[1]+offset,:] = a
        c[offset:a.shape[0]+offset, offset:a.shape[1]+offset] = 1
    elif seed_type > 2:
        offset = int(outsize/2) - int((seed_type-1)/2)
        b[offset:seed_type+offset, offset:seed_type+offset,:] = a[0:seed_type, 0:seed_type]
        c[offset:seed_type+offset, offset:seed_type+offset] = 1

    return a, b, c

def get_unfilled_neighbors(mask):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(mask, kernel, iterations=1)

    res = np.nonzero(dilate-mask[:,:,0])


    return list(zip(*res))

def get_neighborhood_window(img, mask, winsize, pix):
    extent = int((winsize-1)/2)

    template = np.zeros((winsize, winsize, 3))
    template_mask = np.zeros((winsize, winsize, 1))

    i = pix[0]
    j = pix[1]

    npad = ((winsize, winsize), (winsize, winsize), (0, 0))

    padded = np.pad(img, npad)
    padded_mask = np.pad(mask, npad)

    return padded[winsize + i - extent : winsize + i + extent + 1,  winsize + j - extent : winsize + j + extent + 1 ], padded_mask[winsize + i - extent : winsize + i + extent + 1,  winsize + j - extent : winsize + j + extent + 1 ]  

def find_matches(template, template_mask, sample):

    err_thresh = 0.1

    sigma = template.shape[0] / 6.4
    kernel = cv2.getGaussianKernel(ksize=template.shape[0], sigma=sigma)
    kernel_2d = np.outer(kernel,kernel)

    gauss_mask = kernel_2d*template_mask[:,:,0]

    extent = int((template.shape[0]-1)/2)
    npad = ((extent, extent), (extent, extent), (0, 0))


    SSD = cv2.matchTemplate(np.pad(sample, npad).astype(np.float32), template.astype(np.float32), cv2.TM_SQDIFF, None, gauss_mask.astype(np.float32))
    SSD = SSD / gauss_mask.sum()

    print(f'S{sample.shape}')
    print(f'S{template.shape}')
    print(f'S{gauss_mask.shape}')
    print(f'S{SSD.shape}')


    pixel_list = np.where((SSD <= SSD.min()*(1 + err_thresh)))

    tmp = [SSD[x,y] for x in pixel_list[0] for y in pixel_list[1]]

    res = [pixel_list[0], pixel_list[1], tmp]


    return list(zip(*res))

def grow_image(fil, outsize, seed_mode, winsize):

    max_error_thresh = 0.3

    sample, img, mask = prep_img(fil, outsize, seed_mode)
    plt.imsave('sample.png', sample[:,:,::-1])
    plt.imsave('image.png', img[:,:,::-1])
    #plt.imsave('mask.png', mask[:,:])

    init_unfilled = np.count_nonzero(1-mask)
    
    while np.count_nonzero(1-mask) > 0:
        progress = 0
        pixels = get_unfilled_neighbors(mask) # each is (i, j)

        for pixel in pixels:
            template, template_mask = get_neighborhood_window(img, mask, winsize, pixel)

            best_matches = find_matches(template, template_mask, sample)
            print(f'Found {len(best_matches)} matches')

            if len(best_matches):
                best_match_choice = np.random.choice(range(len(best_matches)))
                best_match = best_matches[best_match_choice]

                if best_match[2] < max_error_thresh:
                    img[pixel[0], pixel[1]] = sample[best_match[0], best_match[1]]
                    mask[pixel[0], pixel[1]] = 1
                    progress = 1
                    print('Replacing Pixel, outputting...')

                    #plt.imshow(img)
                    plt.imsave('temp.png', img[:,:,::-1])
            
        if progress == 0:
            max_error_thresh *= 1.1

        print(f'{np.count_nonzero(1-mask)}/{init_unfilled} Pixels need to be filled ... ')
        
    return img

if __name__ == '__main__':
    # python3 npts.py -i ../el_pattern.png -o nan -w 5 -sm 1 -os 94
    args = parse_args()
    img = grow_image(args.input, int(args.outputsize), int(args.seedmode), int(args.winsize))