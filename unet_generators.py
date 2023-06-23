import os
import numpy as np
import cv2 as cv

import zipfile
from zipfile import ZipFile

from common_utils.noisyutils import moirebackground
from common_utils.picutils import get_properbb, compute_iou

class UnetMaskTestGenerator:

    def __init__(self, iterator, random_gen, max_num, noise_level, size_in, size_out):
        self.iterator = iterator
        self.imgsize = size_out
        self.mnistsize = size_in
        self.noise = noise_level
        self.randgen = random_gen
        self.maxnum = max_num

    def __iter__(self):
        return self

    def __next__(self):        
        
        randoms_here = self.randgen.rand(10+self.maxnum)
        blanks = 1 - self.noise * randoms_here[8]*moirebackground(randoms_here[0:8], self.imgsize)
        masks = np.zeros(shape=[self.imgsize, self.imgsize,1], dtype=np.float32)
        npics = 1 + int(randoms_here[9]*self.maxnum)
        
        boxes = []        
        ip = 0
        while ip < npics:
            a = next(self.iterator)
        
            image = a['image'].astype(np.float32)[::-1]
            labl = a['label'] + 1
            ratio = 1 + 2 * randoms_here[10+ip]
            w, h = int(self.mnistsize*ratio), int(self.mnistsize*ratio)
            image = cv.resize(image, (w, h))
            boundbox = get_properbb(image)
            shapebox = np.array([0, 0, w, h], dtype=np.int32)
            
            numsafe = 0
            while True:
                randoms_box = self.randgen.rand(2)
                
                xmin = int((self.imgsize-w)*randoms_box[0])
                ymin = int(self.imgsize*randoms_box[1])

                box = np.array([xmin, ymin, 0, 0], dtype=np.int32) + boundbox
                sox = np.array([xmin, ymin, 0, 0], dtype=np.int32) + shapebox
                
                iou = [compute_iou(sox, b, self.imgsize) for b in boxes]
                if len(iou)==0 or max(iou) < 0.02:
                    break
                numsafe += 1
                if (numsafe == 100):
                    break                    
                    
            ip += 1
            if (numsafe == 100):
                continue
                
            boxes.append(sox)
            for j in range(h):
                y = (ymin + j) % self.imgsize
                blanks[y,xmin:xmin+w,0] *= 1 - image[j,:] / 255
                
            for j in range(box[3]):
                y = (box[1] + j) % self.imgsize
                masks[y,box[0]:box[0]+box[2],0] = labl

        return 1 - blanks, masks
    
    
class UnetMaskTrainGenerator:

    def __init__(self, iterator, random_gen, max_num, noisefiles, noise_level, size_in, size_out):
        self.iterator = iterator
        self.imgsize = size_out
        self.mnistsize = size_in
        self.noise = noise_level
        self.randgen = random_gen
        self.maxnum = max_num
        self.moirefiles = noisefiles
        self.shindex = np.arange(len(self.moirefiles))
        np.random.shuffle(self.shindex)
        self.count = -1
        self.numread = -1
        self.numfile = -1
        self.randoms_pool = None
        self.moiredat = None

    def __iter__(self):
        self.shindex = np.arange(len(self.moirefiles))
        np.random.shuffle(self.shindex)
        self.count = -1
        self.numread = -1
        self.numfile = -1
        self.randoms_pool = None
        self.moiredat = None
        return self

    def __next__(self):        
        
        if self.count == self.numread:
            self.numfile = (self.numfile+1) % len(self.moirefiles)
            fileix = self.shindex[self.numfile]
            filename = self.moirefiles[fileix]
            with zipfile.ZipFile(filename, 'r') as thezip:
                contents = thezip.namelist()
                with thezip.open(contents[0]) as thefile:
                    self.moiredat = np.frombuffer(thefile.read(), dtype=np.uint8).reshape((-1,500,500,1))  
            self.numread = self.moiredat.shape[0]
            self.randoms_pool = np.random.rand(self.numread,5+self.maxnum)
            self.count = 0
        
        randoms_here = self.randoms_pool[self.count]
        
        xs = int((500-self.imgsize) * randoms_here[0])
        ys = int((500-self.imgsize) * randoms_here[1])
        flp = int(4 * randoms_here[2])
        if flp == 0:
            datapiece = self.moiredat[self.count,ys:ys+self.imgsize,xs:xs+self.imgsize]
        elif flp == 1:
            datapiece = self.moiredat[self.count,ys+self.imgsize:ys:-1,xs:xs+self.imgsize]
        elif flp == 2:
            datapiece = self.moiredat[self.count,ys:ys+self.imgsize,xs+self.imgsize:xs:-1]
        else:
            datapiece = self.moiredat[self.count,ys+self.imgsize:ys:-1,xs+self.imgsize:xs:-1]
        blanks = 1 - self.noise * randoms_here[3]*datapiece/255
        masks = np.zeros(shape=[self.imgsize, self.imgsize, 1], dtype=np.float32)
        npics = 1 + int(randoms_here[4]*self.maxnum)
        
        boxes = []
        
        ip = 0
        while ip < npics:
            a = next(self.iterator)
        
            image = a['image'].astype(np.float32)[::-1]
            labl = a['label'] + 1
            ratio = 1 + 2 * randoms_here[ip+5]
            w, h = int(self.mnistsize*ratio), int(self.mnistsize*ratio)
            image = cv.resize(image, (w, h))
            boundbox = get_properbb(image)
            shapebox = np.array([0, 0, w, h], dtype=np.int32)
            
            numsafe = 0
            while True:
                randoms_box = self.randgen.rand(2)

                xmin = int((self.imgsize-w)*randoms_box[0])
                ymin = int(self.imgsize*randoms_box[1])
                
                box = np.array([xmin, ymin, 0, 0], dtype=np.int32) + boundbox
                sox = np.array([xmin, ymin, 0, 0], dtype=np.int32) + shapebox
                
                iou = [compute_iou(sox, b, self.imgsize) for b in boxes]
                if len(iou)==0 or max(iou) < 0.02:
                    break
                numsafe += 1
                if (numsafe == 100):
                    break                    
                    
            ip += 1
            if (numsafe == 100):
                continue
                
            boxes.append(sox)
            for j in range(h):
                y = (ymin + j) % self.imgsize
                blanks[y,xmin:xmin+w,0] *= 1 - image[j,:] / 255
                
            for j in range(box[3]):
                y = (box[1] + j) % self.imgsize
                masks[y,box[0]:box[0]+box[2],0] = labl
                
        self.count += 1
        return 1 - blanks, masks
