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

class GMC_Laser_Plate_Virtual_Object(MujocoXMLObject):
    '''
    GMC_Laser_Plate_Virtual
    '''
    def __init__(self, name):
        super().__init__("./assets/XML_files/GMC_Laser_Plate_Virtual.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)