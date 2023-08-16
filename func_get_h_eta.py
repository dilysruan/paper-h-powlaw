import numpy as np

def get_info(files, bests, eta_tracker, increment):

    #eta_tracker=1.0   #whatever the power law index starts with
    #increment=0 #the step size between power law values

#file_root = 'mock_quad_shear_eta'

    step=0.000001 #slight horizontal offset so figure more readable

    h_med=[]
    h_err=[]
    eta_med=[]
    eta_err=[]

    eta_true=[]

    eta_best = []
    h_best = []

    index=0
    for n in range(0,len(files)):
        data = np.loadtxt(files[n])
        data = np.transpose(data)
    
        best = np.loadtxt(bests[n])
        eta_best.append(best[-2])#[5])
        h_best.append(best[-1])#[6])

        h = data[-1]#[6]
        eta = data[-2]#[5]

        h_med.append(np.quantile(h, q=0.5))
        h_err.append( [(h_med[index]-np.quantile(h,q=0.16)),(np.quantile(h,q=0.84) - h_med[index])] )

        eta_med.append(np.quantile(eta, q=0.5))
        eta_err.append( [(eta_med[index]-np.quantile(eta,q=0.16)),(np.quantile(eta,q=0.84) - eta_med[index])] )

        eta_true.append(eta_tracker+n*step)

        h_eta_array = [h_med[index], h_err[index][0], h_err[index][1], eta_med[index], eta_err[index][0], eta_err[index][1]]

        index+=1
    
        eta_tracker+=step
    
    eta_err = np.transpose(eta_err)
    h_err = np.transpose(h_err)

    print('h_med, h_err, eta_true, eta_med, eta_err, h_best, eta_best')
    return h_med, h_err, eta_true, eta_med, eta_err, h_best, eta_best
