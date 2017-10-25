import numpy as np
from os import path

import MCPM


class TpfRectangles(object):
    """
    Keeps information on rectangles that are used in TPF files.
    Note that there may be multple rectangles for a single TPF,
    some rectangles are not full, 
    and some may have width or height of 1 pixel
    """
    
    directory = path.join(MCPM.MODULE_PATH, 'data', 'K2C9', 'tpf_rectangles') 
    
    def __init__(self, campaign, channel):
        self.campaign = campaign
        self.channel = channel
        
        file_name = "tpf_rectangles_{:}_{:}.data".format(campaign, channel)
        path_ = path.join(self.directory, file_name)
        load = np.loadtxt(path_, unpack=True, dtype=int)
        (epic, self.min_x, self.max_x, self.min_y, self.max_y) = load
        self.epic = np.array(epic, dtype=str)
        
        self.center_x = (self.max_x + self.min_x) / 2.
        self.center_y = (self.max_y + self.min_y) / 2.
        
        self.half_size_x = self.max_x - self.center_x
        self.half_size_y = self.max_y - self.center_y
        
    def point_distances(self, x, y):
        """get distance from given (x,y) point to all the rectangles"""
        dx = np.maximum(np.abs(x - self.center_x) - self.half_size_x, 0.)
        dy = np.maximum(np.abs(y - self.center_y) - self.half_size_y, 0.)
        return np.sqrt(dx**2 + dy**2)
        
    def closest_epics(self, x, y):
        """for given point (x,y), calculate all the distances and then sort 
        them so that sorted list of epics is returned; takes into account 
        only the closest rectangle from given epic"""
        distances = self.point_distances(x=x, y=y)
        indexes = np.argsort(distances)
        out_epic = []
        out_distance = []
        for i in indexes:
            if self.epic[i] in out_epic:
                continue
            out_epic.append(self.epic[i])
            out_distance.append(distances[i])
        return (out_epic, out_distance)
        
if __name__ == '__main__':
    channel=52 
    campaign=92
    (x, y) = (883, 670)
    
    rectangle = TpfRectangles(campaign, channel)
    (out_1, out_2) = rectangle.closest_epics(y, x)
    print(out_1[:15])
    print(out_2[:15])
    print(len(out_2), len(rectangle.epic))