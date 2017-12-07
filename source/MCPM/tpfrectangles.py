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
    
    def __init__(self, campaign, channel):
        self.campaign = campaign
        self.channel = channel
        
        file_name = "tpf_rectangles_{:}_{:}.data".format(campaign, channel)
        if int(self.campaign) in [91, 92]:
            subdir = 'K2C9'
        elif int(self.campaign) in [111, 112]:
            subdir = 'K2C11'
        else:
            msg = 'expected campaigns: 91, 92, 111, or 112; got {:}'
            raise ValueError(msg.format(self.campaign))
        directory = path.join(MCPM.MODULE_PATH, 'data', subdir) 
        path_ = path.join(directory, 'tpf_rectangles', file_name)
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
        return (np.array(out_epic), np.array(out_distance))
   
    def get_epic_id_for_pixel(self, x, y):
        """find in which TPF given pixel lies"""
        distances = self.point_distances(x=x, y=y)
        selection = (distances == 0.)
        if not selection.any():
            return None
        else:
            return self.epic[np.argmax(selection)]
           
    def get_nearest_epic(self, x, y):
        """find the nearest epic; return its id and distance 
        (0 if point is inside it)"""
        distances = self.point_distances(x=x, y=y)
        index = np.argmin(distances)
        return (self.epic[index], distances[index])
        