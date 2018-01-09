import numpy as np


def debug(obj, name="<noname>", plimit=250):
    
    typ = type(obj)
    typstr = str(typ)
    
    # for debugging ndarray shape (large use case)
    if typ == np.ndarray:
        shape = obj.shape
        amax = np.amax(obj)
        amin = np.amin(obj)
        diagnostics = "arrmax = %s, arrmin = %s" % (str(amax), str(amin))
    else:
        diagnostics = ''
        try:
            shape = len(obj)
        except TypeError:
            shape = "<n/a>"
    shapestr = str(shape)

    objstr = str(obj)
    objout = objstr if len(objstr) < plimit else objstr[:plimit] + "[...%s output truncated]" % name
    
    header = ' '.join([typstr, shapestr, name + ':', diagnostics])
    print ("\n".join([header, objout]))
