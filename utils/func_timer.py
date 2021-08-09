from time import time
from functools import wraps

def timing(f):
    '''
    Function timer wrapping
    '''
    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print('Elapsed time for %r : %2.3f ms' % (f.__name__, (te - ts) * 1e03))
        return result

    return wrapper