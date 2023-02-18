import numpy as np
from math import ceil
from PIL import Image

def find_coeffs(pa, pb):
    '''
    Distortion
    '''
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float32)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)
    
def distort(img, distortrnd):
    '''
    img -- PIL image
    distortrnd -- random array size 8
    Returns distorted image (random perspective)
    '''
    ns = img.size[0]
    m = distortrnd * 0.25 * ns

    x1, y1 = -m[0:2]
    x2, y2 = ns+m[2], m[3]

    x3, y3 = ns+m[4], ns-m[5]
    x4, y4 = -m[6], ns+m[7]

    xmin = min(x1, x4)
    if (xmin < 0):
        x1 -= xmin
        x2 -= xmin
        x3 -= xmin
        x4 -= xmin

    ymin = min(y1, y2)
    if (ymin < 0):
        y1 -= ymin
        y2 -= ymin
        y3 -= ymin
        y4 -= ymin

    new_width = int(ceil(max(x2,x3,ns)))
    new_height = int(ceil(max(y3,y4,ns)))

    coeffs = find_coeffs(
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
        [(0, 0), (ns, 0), (ns, ns), (0, ns)])
    img = img.transform((new_width, new_height), 
                        Image.Transform.PERSPECTIVE,
                        coeffs,
                        Image.Resampling.BICUBIC,
                        fillcolor = 0)
    return img
    
def moirebackground(rndarr, imgsize):
    '''
    rndarr -- random array size 8
    Returns numpy array shape (imgsize,imgsize) with moire pattern
    '''
    ns = 2*imgsize

    blank = np.zeros((ns,ns), np.uint8)
    for i in range(ns):
        if i % 3 != 0:
            continue
        blank[i] = 255*np.ones(ns, np.uint8)
    img = Image.fromarray(blank)

    rndxxx = rndarr*np.array(4*[1,-1])
    
    im1 = distort(img, rndarr)
    im2 = distort(img, rndxxx)
    
    xs = max(im1.size[0], im2.size[0])
    ys = max(im1.size[1], im2.size[1])
    
    im1 = im1.resize((xs,ys)) 
    im2 = im2.resize((xs,ys)) 
    
    re1 = np.asarray(im1)
    re2 = np.asarray(im2)
    
    ims = Image.fromarray(np.maximum(re1,re2))
    ims = ims.crop((imgsize//2, imgsize//2, imgsize//2+imgsize, imgsize//2+imgsize))
    ims = ims.resize((120,120))
    ims = ims.resize((imgsize,imgsize))
    return np.expand_dims(ims, axis=-1) / 255
