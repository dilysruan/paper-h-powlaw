import numpy as np

def get_info(files): 

    imgpos = []
    drarr  = []
    taumat = []

    for n in range(0,len(files)):
        filein = files[n]

        image_data = np.loadtxt(filein,skiprows=7)
        #print(data)

        # sort in time delay order, and arrange so image index is last
        tindex = np.argsort(image_data[:,5])
        data = np.transpose(image_data[tindex])
        # store image positions
        imgpos.append(data[:2])
        
        # image radii
        rtmp = np.sqrt(data[0]**2 + data[1]**2)
        # store Delta r relative to leading image
        drarr.append(rtmp[0] - rtmp)
        
        # store differential time delays
        tdel = data[5]
        tautmp = np.zeros((4,4))
        for i in range(4):
            for j in range(i+1,4):
                tautmp[i,j] = tdel[j] - tdel[i]
        taumat.append(tautmp)
        
    imgpos = np.array(imgpos)
    drarr  = np.array(drarr)
    taumat = np.array(taumat)
    
    print('imgpos, drarr, taumat')
    return imgpos, drarr, taumat
