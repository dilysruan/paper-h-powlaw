import numpy as np

def get_info(files): 

    peakmag = []
    Rij = []
    totalmag=[]
    tau12 = []
    tau13 = []
    tau14 = []
    tau23 = []
    tau24 = []
    tau34 = []
    time_info = []
    time_max=[]
    time_min=[]
    radius_arr = []
    annulus_arr = []

    for n in range(0,len(files)):
        filein = files[n]

        data = np.loadtxt(filein,skiprows=7)
        #print(data)

        data = np.transpose(data)
        distance = np.sqrt(np.array(data[0])**2 + np.array(data[1])**2)
        Rij_index = np.argsort(distance)
        Rij.append((distance[Rij_index[-1]] - distance[Rij_index[0]])/(distance[Rij_index[-1]] + distance[Rij_index[0]]))
        peakmag.append(np.max(data[2]))
        totalmag.append(np.sum(data[2]))
        timedelays = data[5]
        timedelays = np.sort(timedelays)
        tau12.append(timedelays[1])
        tau13.append(timedelays[2])
        tau14.append(timedelays[3])
        tau23.append(timedelays[2]-timedelays[1])
        tau24.append(timedelays[3]-timedelays[1])
        tau34.append(timedelays[3]-timedelays[2])
        time_max.append(np.max([tau12[n], tau13[n], tau14[n], tau23[n], tau24[n], tau34[n]]))
        time_min.append(np.min([tau12[n], tau13[n], tau14[n], tau23[n], tau24[n], tau34[n]]))
        
        data = np.transpose(data)
        radius_arr.append(np.sqrt(data[:,0]**2 + data[:,1]**2))
        annulus_arr.append((np.min(radius_arr[n]), np.max(radius_arr[n])))

    radius_arr = (np.array(radius_arr))
    annulus_arr = (np.array(annulus_arr))
    annulus_length = (np.abs(annulus_arr[:,0] - annulus_arr[:,1]))
    
    print('totalmag, peakmag, Rij, tau12, tau13, tau14, tau23, tau24, tau34, time_max, time_min, annulus_length')
    return totalmag, peakmag, Rij, tau12, tau13, tau14, tau23, tau24, tau34, time_max, time_min, annulus_length
