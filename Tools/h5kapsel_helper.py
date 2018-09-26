import h5py
import numpy as np
import itertools

#Map function as np array over arbitrary lists
def npiter(func, *iterables):
    return np.array(list(itertools.imap(func, *iterables)))

#Normalize list of vectors
def normalize(r):
    return npiter(lambda x: x / np.sqrt(x.dot(x)), r)

# get number of particles and number of frames
def h5_conf(fname):
    f = h5py.File(fname, "r")
    nump = int(f.attrs['nump'][0])
    numo = int(f.attrs['nump_obs'][0])
    numf = len(f['trajectory_data'].items())
    f.close()
    return nump, numo, numf

# Read kapsel h5 particle trajectory data for property 'tag'
# valid tags: 
#             pos       (position)
#             pos_raw   (position with no pbc)
#             QR        (orientation matrix)
#                          x_body = QR.x_lab
#                          x_lab  = tr(QR).x_body
#             vel       (velocity)
#             omega     (angular velocity)
#             force_h   (hydrodynamic force)
#             torque_h  (hydrodynamic torque)
#             force_r   (non-hydro force: external, LJ, etc.)
def h5_pdata_trj(fname, tag):
    valid_tag = ['pos', 'pos_raw', 'QR', 'vel', 'omega', 'force_h', 'force_r', 'torque_h']
    if tag not in valid_tag:
        print "Invalid Tag"
        print "Valid tags are: ", valid_tag
        return
    
    nump, numo, numf = h5_conf(fname)

    f = h5py.File(fname, "r")
    loc = 'trajectory_data/frame_0/particle/'+tag
    x0  = f[loc].value
    shape = list(x0.shape)
    shape.insert(0, numf)
    x = np.zeros(tuple(shape), dtype=x0.dtype)
    for i in range(numf):
        loc = 'trajectory_data/frame_'+str(i)+'/particle/'+tag
        x[i] = f[loc].value
    f.close()
    return x
# Read kapsel h5 particle frame data for property 'tag'
def h5_pdata(f, tag, tid):
    valid_tag = ['pos', 'pos_raw', 'QR', 'vel', 'omega', 'force_h', 'force_r', 'torque_h']
    if tag not in valid_tag:
        print "Invalid Tag"
        print "Valid tags are: ", valid_tag
        return
    
    loc = 'trajectory_data/frame_'+str(tid) + '/particle/'+tag
    return f[loc].value
    

# Read kapsel h5 obstacle trajectory data for property 'tag'
# valid tags: 
#             vel       (velocity)
#             omega     (angular velocity)
#             force_h   (hydrodynamic force)
#             torque_h  (hydrodynamic torque)
#             force_r   (non-hydro force: external, LJ, etc.)
def h5_odata_trj(fname, tag):
    valid_tag = ['force_h', 'force_r']
    if tag not in valid_tag:
        print "Invalid Tag"
        print "Valid tags are: ", valid_tag
        return
    
    nump, numo, numf = h5_conf(fname)

    f = h5py.File(fname, "r")
    loc = 'trajectory_data/frame_0/obstacle/'+tag
    x0  = f[loc].value
    shape = list(x0.shape)
    shape.insert(0, numf)
    x = np.zeros(tuple(shape), dtype=x0.dtype)
    for i in range(numf):
        loc = 'trajectory_data/frame_'+str(i)+'/obstacle/'+tag
        x[i] = f[loc].value
    f.close()
    return x

# Read kapsel h5 obstacle frame data for property 'tag'
def h5_odata(f, tag, tid):
    valid_tag = ['force_h', 'force_r']
    if tag not in valid_tag:
        print "Invalid Tag"
        print "Valid tags are: ", valid_tag
        return
    
    loc = 'trajectory_data/frame_'+str(tid)+'/obstacle/'+tag
    return f[loc].value


# get trajectory time data
def h5_time(fname):
    nump, numo, numf = h5_conf(fname)

    f = h5py.File(fname, "r")
    t0 = f['trajectory_data/frame_0'].attrs['time']
    t = np.zeros(numf, dtype=t0.dtype)
    for i in range(numf):
        loc = 'trajectory_data/frame_'+str(i)
        t[i] = f[loc].attrs['time']
    f.close()
    return t
