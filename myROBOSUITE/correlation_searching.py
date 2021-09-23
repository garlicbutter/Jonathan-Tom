import numpy as np
from numpy.core.fromnumeric import reshape

class things_to_test():
    '''
    Dataclass for things to test iteratively.
    the iterator returns the evenly spaced value within the range.
    '''
    def __init__(self,
                name:str,
                testing_min,
                testing_max,
                amount_of_test:int) -> None:
    
        self.name = name
        self.min  = testing_min
        self.max  = testing_max
        self.num  = amount_of_test
        self.interval = (self.max - self.min)/(amount_of_test-1)
    
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

def main():
    kp_test = things_to_test('controller_stiffness',
                            testing_min = np.array([100, 100, 50, 50, 50, 100]),
                            testing_max = np.array([1500, 1500, 50, 150, 150, 150]),
                            amount_of_test = 10)

    kd_test = things_to_test('damping_ratio',
                            testing_min = np.array([1, 1, 1, 10, 10, 10]),
                            testing_max = np.array([3, 3, 1, 10, 10, 10]),
                            amount_of_test = 10)

    test_results = np.array([])

    for kp in iter(kp_test):
        for kd in iter(kd_test):
            result = run_test(kp, kd)
            test_results = np.append(test_results, result)
    
    print(test_results)

def run_test(kp, kd, **kwargs):
    return kp[0]*kd[1]

if __name__ == "__main__":
    main()
