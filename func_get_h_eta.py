import numpy as np

def get_info(files, eta_tracker, increment):

    h_med=[]
    h_err=[]
    eta_med=[]
    eta_err=[]

    index=0
    for n in range(0,len(files)):
        data = np.loadtxt(files[n])
        data = np.transpose(data)

        h = data[-1]
        eta = data[-2]

        h_med.append(np.quantile(h, q=0.5))
        h_err.append( [(h_med[index]-np.quantile(h,q=0.16)),(np.quantile(h,q=0.84) - h_med[index])] )

        eta_med.append(np.quantile(eta, q=0.5))
        eta_err.append( [(eta_med[index]-np.quantile(eta,q=0.16)),(np.quantile(eta,q=0.84) - eta_med[index])] )

        h_eta_array = [h_med[index], h_err[index][0], h_err[index][1], eta_med[index], eta_err[index][0], eta_err[index][1]]

        index+=1
        
    eta_err = np.transpose(eta_err)
    h_err = np.transpose(h_err)

    print('h_med, h_err, eta_med, eta_err')
    return h_med, h_err, eta_med, eta_err
