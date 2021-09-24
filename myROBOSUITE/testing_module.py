from matplotlib.pyplot import legend
import numpy as np
from main_osc import main_osc

class things_to_test():
    '''
    Dataclass for things to test iteratively.
    the iterator returns the evenly spaced value within the range.
    '''
    def __init__(self,
                name:str,
                testing_min,
                testing_max,
                amount_of_tests:int) -> None:
    
        assert all(testing_max>=testing_min)  # check if all the max values are larger than min values
        assert amount_of_tests >=2
        self.name = name
        self.min  = testing_min
        self.max  = testing_max
        self.num  = amount_of_tests
        self.interval = (self.max - self.min)/(self.num-1)
    
    def __iter__(self):
        self.n  = 0
        self.result = self.min - self.interval
        return self

    def __next__(self):
        if self.n < self.num:
            self.result = self.result + self.interval
            self.n += 1
            return np.round( self.result, 3)
        else:
            raise StopIteration

def run_test(kp, kd, percetion_error):
    eeff_record, eeft_record, eefd_record, t_record = main_osc(kp,
                                                               kd, 
                                                               perception_error=percetion_error,
                                                               offscreen=True)
    
    if isinstance(t_record, int):
        result = {'success':False,
                    'run_time':0,
                    'inserting_eeff_xy_max':0,
                    'inserting_eeff_z_max':0}
    else:
        inserting_eeff_x_max = np.amax(eeff_record[0][-50:-1])
        inserting_eeff_y_max = np.amax(eeff_record[1][-50:-1])
        inserting_eeff_z_max = np.amax(eeff_record[2][-50:-1])
        result = {'success':True,
                    'run_time':t_record,
                    'inserting_eeff_xy_max':np.sqrt(inserting_eeff_x_max**2 + inserting_eeff_y_max**2),
                    'inserting_eeff_z_max':inserting_eeff_z_max}
    
    return result