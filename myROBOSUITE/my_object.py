import numpy as np
from robosuite.models.objects import MujocoXMLObject

class GMC_Plate_Object(MujocoXMLObject):
    '''
    GMC Plate object
    '''
    def __init__(self, name):
        super().__init__("./assets/XML_files/GMC_Plate.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class GMC_Assembly_Object(MujocoXMLObject):
    '''
    GMC Plate object
    '''
    def __init__(self, name):
        super().__init__("./assets/XML_files/GMC_Assembly.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class Round_peg_16mm_Object(MujocoXMLObject):
    '''
    Round_peg_16mm
    '''
    def __init__(self, name):
        super().__init__("./assets/XML_files/Round_peg_16mm.xml",
                         name=name, joints=[dict(type="free", damping="0.0001")],
                         obj_type="all", duplicate_collision_geoms=True)

class Round_peg_12mm_Object(MujocoXMLObject):
    '''
    Round_peg_12mm
    '''
    def __init__(self, name):
        super().__init__("./assets/XML_files/Round_peg_12mm.xml",
                         name=name, joints=[dict(type="free", damping="0.0001")],
                         obj_type="all", duplicate_collision_geoms=True)

class Custom_Hole_18mm_Object(MujocoXMLObject):
    '''
    Round_peg_12mm
    '''
    def __init__(self, name):
        super().__init__("./assets/XML_files/Custom_Hole_18mm.xml",
                         name=name, joints=[dict(type="free", damping="0.0001")],
                         obj_type="all", duplicate_collision_geoms=True)
