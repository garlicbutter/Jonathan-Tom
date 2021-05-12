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

class GMC_Competition_Task_Board_Virtual_Object(MujocoXMLObject):
    '''
    GMC_Competition_Task_Board_Virtual
    '''
    def __init__(self, name):
        super().__init__("./assets/XML_files/GMC_Competition_Task_Board_Virtual.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class Plate_course_Object(MujocoXMLObject):
    '''
    Plate_course
    '''
    def __init__(self, name):
        super().__init__("./assets/XML_files/Plate_course.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class GMC_Laser_Plate_fine_mm_Object(MujocoXMLObject):
    '''
    GMC_Laser_Plate_fine_mm
    '''
    def __init__(self, name):
        super().__init__("./assets/XML_files/GMC_Laser_Plate_fine_mm.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class Gear_Assembly_Object(MujocoXMLObject):
    '''
    Gear_Assembly
    '''
    def __init__(self, name):
        super().__init__("./assets/XML_files/Gear_Assembly.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class Round_peg_16mm_Object(MujocoXMLObject):
    '''
    Round_peg_16mm
    '''
    def __init__(self, name):
        super().__init__("./assets/XML_files/Round_peg_16mm.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

