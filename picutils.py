import numpy as np

def get_properbb(img):
    '''
    Returns box that contains not-zero values
    '''
    bycol = np.where(np.squeeze(np.any(img > 0, axis=1)))[0]
    byrow = np.where(np.squeeze(np.any(img > 0, axis=0)))[0]
    ymin, ymax = bycol[0], bycol[-1]
    xmin, xmax = byrow[0], byrow[-1]
    return np.array([xmin, ymin, xmax-xmin, ymax-ymin])
    
    
def intersection(seg1Start, seg1End, seg2Start, seg2End):
    '''
    Intersection of two intervals: true if intersection takes place
    '''
    return seg2Start <= seg1End and seg1Start <= seg2End
def intersectionp(seg1Start, seg1End, seg2Start, seg2End, period):
    '''
    Intersection of two intervals with periodic boundary conditions
    '''
    l0 = intersection(seg1Start, seg1End, seg2Start, seg2End)
    l1 = intersection(seg1Start, seg1End, seg2Start + period, seg2End + period)
    l2 = intersection(seg1Start, seg1End, seg2Start - period, seg2End - period)
    return l0 or l1 or l2    
    
    
def intersectlen(seg1Start, seg1End, seg2Start, seg2End):
    '''
    Intersection length of two intervals
    '''
    if not intersection(seg1Start, seg1End, seg2Start, seg2End):
        return 0
    mn = np.minimum(seg1End, seg2End)
    mx = np.maximum(seg1Start, seg2Start)
    return mn - mx
def intersectlenp(seg1Start, seg1End, seg2Start, seg2End, period):
    '''
    Intersection length of two intervals with periodic boundary conditions
    '''
    l0 = intersectlen(seg1Start, seg1End, seg2Start, seg2End)
    l1 = intersectlen(seg1Start, seg1End, seg2Start + period, seg2End + period)
    l2 = intersectlen(seg1Start + period, seg1End + period, seg2Start, seg2End)
    return max(l0,l1,l2)
def compute_iou(box1, box2, period):
    '''
    IoU of two boxes with periodic boundary conditions on Y dimension
    '''
    A1 = box1[2] * box1[3]
    A2 = box2[2] * box2[3]
    P1 = intersectlen(box1[0], box1[0]+box1[2], box2[0], box2[0]+box2[2])
    P2 = intersectlenp(box1[1], box1[1]+box1[3], box2[1], box2[1]+box2[3], period)
    AA = P1 * P2
    return  AA / min(A1, A2)
    

