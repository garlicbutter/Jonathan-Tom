import numpy as np
from robosuite.models.objects import MujocoXMLObject

class GMCPlateObject(MujocoXMLObject):
    '''
    GMC Plate object
    '''
    def __init__(self, name):
        super().__init__("./assets/XML_files/GMC_Plate.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)