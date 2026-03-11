import itertools
import numpy as np

class Joint:
    newid = itertools.count()
    def __init__(self, beams = None, joint_points = None):
        self.id = next(self.newid)
        self.beams = beams
        self.joint_points = joint_points
