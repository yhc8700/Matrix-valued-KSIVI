
import torch
import numpy as np

def gaussian_kernel(samples_x, samples_y, h = -1, get_width = False, detach=False):
    '''
    samples_x: shape = [n,d]
    samples_y: shape = [n,d]
    '''
    pairwise_dists = ((samples_x[:,None,:] - samples_y[None,:,:])**2).sum(-1)
    if h < 0: # use the median trick
        if detach:
            h = torch.median(pairwise_dists).detach()
        else:
            h = torch.median(pairwise_dists)
        h = torch.sqrt(0.5 * h /np.log(samples_x.shape[0] + 1))
    kxy = torch.exp(- pairwise_dists/ h**2 / 2)
    if get_width:
        return kxy, h 
    else:
        return kxy

def matrix_kernel(samples_x, samples_y, anchor = torch.tensor([0]), h = -1, get_width = False, detach=False):
    '''
    samples_x: shape = [n,d]
    samples_y: shape = [n,d]
    '''
    n = samples_x.shape[0]
    d = samples_x.shape[1]
    m = anchor.shape[0] # size of anchor points

    with torch.autograd.set_detect_anomaly(True):
        def Hessian(X): # X with shape [n,d]; return the negative Hessian of p(x) for STAR_SHAPED
            det = 3.8 * 0.2
            sigmasqinv_0 = torch.tensor([[2., 1.8], [1.8, 2.]]).unsqueeze(0) / det
            sigmasqinv_1 = torch.tensor([[3.8, 0], [0, 0.2]]).unsqueeze(0) / det
            sigmasqinv_2 = torch.tensor([[2., -1.8], [-1.8, 2.]]).unsqueeze(0) / det
            sigmasqinv_3 = torch.tensor([[0.2, 0], [0, 3.8]]).unsqueeze(0) / det # shape = [1, d, d]

            Y = torch.nn.functional.softmax(torch.stack(
                (-1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_0), X[:,:,None]).squeeze(-1),
                -1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_1), X[:,:,None]).squeeze(-1),
                -1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_2), X[:,:,None]).squeeze(-1),
                -1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_3), X[:,:,None]).squeeze(-1)),1
                ), dim = 1).squeeze()
            
            p10 = torch.matmul(sigmasqinv_1 - sigmasqinv_0, torch.matmul(torch.matmul(X.unsqueeze(2), X.unsqueeze(1)), sigmasqinv_1 - sigmasqinv_0)) # shape = [n, d, d]
            p20 = torch.matmul(sigmasqinv_2 - sigmasqinv_0, torch.matmul(torch.matmul(X.unsqueeze(2), X.unsqueeze(1)), sigmasqinv_2 - sigmasqinv_0))
            p30 = torch.matmul(sigmasqinv_3 - sigmasqinv_0, torch.matmul(torch.matmul(X.unsqueeze(2), X.unsqueeze(1)), sigmasqinv_3 - sigmasqinv_0))
            p21 = torch.matmul(sigmasqinv_2 - sigmasqinv_1, torch.matmul(torch.matmul(X.unsqueeze(2), X.unsqueeze(1)), sigmasqinv_2 - sigmasqinv_1))
            p31 = torch.matmul(sigmasqinv_3 - sigmasqinv_1, torch.matmul(torch.matmul(X.unsqueeze(2), X.unsqueeze(1)), sigmasqinv_3 - sigmasqinv_1))
            p32 = torch.matmul(sigmasqinv_3 - sigmasqinv_2, torch.matmul(torch.matmul(X.unsqueeze(2), X.unsqueeze(1)), sigmasqinv_3 - sigmasqinv_2))

            w10 = (Y[:, 1] * Y[:, 0]).unsqueeze(1).unsqueeze(2) # shape = [n, 1, 1]
            w20 = (Y[:, 2] * Y[:, 0]).unsqueeze(1).unsqueeze(2)
            w30 = (Y[:, 3] * Y[:, 0]).unsqueeze(1).unsqueeze(2)
            w21 = (Y[:, 2] * Y[:, 1]).unsqueeze(1).unsqueeze(2)
            w31 = (Y[:, 3] * Y[:, 1]).unsqueeze(1).unsqueeze(2)
            w32 = (Y[:, 3] * Y[:, 2]).unsqueeze(1).unsqueeze(2)

            b = w10*p10 + w20*p20 + w30*p30 + w21*p21 + w31*p31 + w32*p32

            a = Y[:,0].unsqueeze(1).unsqueeze(2) * sigmasqinv_0 + Y[:,1].unsqueeze(1).unsqueeze(2) * sigmasqinv_1 + Y[:,2].unsqueeze(1).unsqueeze(2) * sigmasqinv_2 + Y[:,3].unsqueeze(1).unsqueeze(2) * sigmasqinv_3 # shape [n, d, d]
            
            return a+b

        def x_Hessian(X): # X with shape [n,d]; return the negative Hessian of p(x) for X_SHAPED
            det = 0.76
            sigmasqinv_0 = torch.tensor([[2., -1.8], [-1.8, 2.]]).unsqueeze(0) / det
            sigmasqinv_1 = torch.tensor([[2., 1.8], [1.8, 2.]]).unsqueeze(0) / det # shape = [1, d, d]

            Y = torch.nn.functional.softmax(torch.stack(
                (-1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_0), X[:,:,None]).squeeze(-1),
                -1/2 * torch.matmul(torch.matmul(X[:,None,:],sigmasqinv_1), X[:,:,None]).squeeze(-1),),1
                ), dim = 1).squeeze() # shape = [n, d]

            # print(Y)
            p1 = torch.matmul(X.unsqueeze(2), X.unsqueeze(1)) # shape = [n, d, d]
            p2 = torch.matmul(sigmasqinv_1 - sigmasqinv_0, torch.matmul(p1, sigmasqinv_1 - sigmasqinv_0)) # shape = [n, d, d]
            p3 = Y[:, 0] * Y[:, 1] # shape = [n]
            p4 = p3.unsqueeze(1).unsqueeze(2) * p2 # shape = [n, d, d]
            p5 = -Y[:, 0].unsqueeze(1).unsqueeze(2) * sigmasqinv_0 - Y[:, 1].unsqueeze(1).unsqueeze(2) * sigmasqinv_1 # shape = [n, d, d]
            
            # Y with shape [n,d,d]
            return p4-p5
    
        # print(x_Hessian(samples_x)[0])
        # print(anchor)

# -------------------------------------------- for-loop code (unfinished)---------------------------------------------------------
        '''
        kxy = np.zeros((n, n, d, d))
        Q = x_Hessian(anchor) # shape = [m, d, d]

        for l in range(m):
            Ql = Q[l].unsqueeze(0).unsqueeze(1)
            tmp = samples_x[:, None, :] - samples_y[None, :, :]
            # tmp2 = torch.matmul(tmp.unsqueeze(2), tmp.unsqueeze(3)) # same as gaussian kernel
            tmp2 = torch.matmul(tmp.unsqueeze(2), torch.matmul(Ql, tmp.unsqueeze(3))) # shape = [n, n, 1, 1]
            
            h = torch.median(tmp2)
            h = torch.sqrt(0.5 * h /np.log(n + 1))

            kxy_l = torch.exp(- tmp2/ h**2 / 2) * torch.eye(d)
            kxy_l = torch.matmul(torch.inverse(Q).unsqueeze(0).unsqueeze(1), kxy_l) # shape = [n, n, d, d]

            
            pairwise_x = anchor[:, None, :] - samples_x[None, :, :] # shape = [m, n, d]       
            aux = -torch.matmul(pairwise_x.unsqueeze(2), torch.matmul(Q_list.unsqueeze(1), pairwise_x.unsqueeze(3))).squeeze()/2 + torch.log(torch.det(Q_list)).unsqueeze(1) # shape = [m, n]
            w_list = torch.nn.functional.softmax(aux, dim=0) # shape = [m, n]; meaning w_l(x_j) for l,j in 1st, 2nd component 
        
            pairwise_y = anchor[:, None, :] - samples_y[None, :, :] # shape = [m, n, d]       
            aux_y = -torch.matmul(pairwise_y.unsqueeze(2), torch.matmul(Q_list.unsqueeze(1), pairwise_y.unsqueeze(3))).squeeze()/2 + torch.log(torch.det(Q_list)).unsqueeze(1) # shape = [m, n]
            w_list_y = torch.nn.functional.softmax(aux_y, dim=0) # shape = [m, n]; meaning w_l(y_j) for l,j in 1st, 2nd component 

            

        return kxy
        '''
# -------------------------------------------- for-loop code (unfinished)---------------------------------------------------------


        pairwise = samples_x[:, None, :] - samples_y[None, :, :] # shape = [n, n, d]
        Q_list = Hessian(anchor).detach() # shape = [m, d, d]
        # Q_list = (Hessian(samples_x)/2 + Hessian(samples_y)/2).detach() 
    
        # Calculate w_list
        pairwise_x = anchor[:, None, :] - samples_x[None, :, :] # shape = [m, n, d]       
        aux = -torch.matmul(pairwise_x.unsqueeze(2), torch.matmul(Q_list.unsqueeze(1), pairwise_x.unsqueeze(3))).squeeze()/2 + torch.log(torch.det(Q_list)).unsqueeze(1) # shape = [m, n]
        w_list = torch.nn.functional.softmax(aux, dim=0) # shape = [m, n]; meaning w_l(x_j) for l,j in 1st, 2nd component 
        
        pairwise_y = anchor[:, None, :] - samples_y[None, :, :] # shape = [m, n, d]       
        aux_y = -torch.matmul(pairwise_y.unsqueeze(2), torch.matmul(Q_list.unsqueeze(1), pairwise_y.unsqueeze(3))).squeeze()/2 + torch.log(torch.det(Q_list)).unsqueeze(1) # shape = [m, n]
        w_list_y = torch.nn.functional.softmax(aux_y, dim=0) # shape = [m, n]; meaning w_l(y_j) for l,j in 1st, 2nd component 

        # Calculate K_QL
        kernelL = torch.matmul(pairwise.unsqueeze(0).unsqueeze(3), torch.matmul(Q_list.unsqueeze(1).unsqueeze(2), pairwise.unsqueeze(0).unsqueeze(4))).squeeze() # shape = [m, n, n]
        if h < 0: # use the median trick
            if detach:
                h = torch.median(kernelL.view((m, n*n)), dim=1)[0].detach()
            else:
                h = torch.median(kernelL.view((m, n*n)), dim=1)[0]

        with torch.no_grad():
            h = h / np.log(n+1) / 2 # shape = [m]
            h = h.clone().detach().unsqueeze(1).unsqueeze(2)

        u = torch.exp(- kernelL / (h) / 2).unsqueeze(3).unsqueeze(4) # shape = [m, n, n, 1, 1]
        kxy_tmp = u * torch.eye(d) # shape = [m, n, n, d, d]
        kxy_i = torch.matmul(torch.linalg.inv(Q_list).unsqueeze(1).unsqueeze(2), kxy_tmp) # shape = [m, n, n, d, d]

        # Final calculation
        wL_i = w_list.unsqueeze(2) * w_list_y.unsqueeze(1) # shape = [m, n, n]        
        wL_i = wL_i.unsqueeze(3).unsqueeze(4) * torch.eye(d) # shape = [m, n, n, 1, 1] -> [m, n, n, d, d] after multiplying eye(d)
        kxy = torch.matmul(kxy_i, wL_i) # shape = [m, n, n, d, d]
        kxy = torch.sum(kxy, dim=0).squeeze()
        return kxy # shape = [n, n, d, d]
    
# ------------------old codes (constant pre-conditioning)-----------------------------------------------------------------

    Q1 = Hessian(samples_x).mean(dim=0).squeeze() #.detach()
    Q2 = Hessian(samples_y).mean(dim=0).squeeze() #.detach()
    Q = (Q1/2 + Q2/2).unsqueeze(0).unsqueeze(1)
    # print(Q)

    
    tmp = samples_x[:, None, :] - samples_y[None, :, :]
    # tmp2 = torch.matmul(tmp.unsqueeze(2), tmp.unsqueeze(3)) # same as gaussian kernel
    tmp2 = torch.matmul(tmp.unsqueeze(2), torch.matmul(Q, tmp.unsqueeze(3)))
    # tmp2 = torch.matmul(tmp.unsqueeze(3), tmp.unsqueeze(2)) # matrix-valued

    # pairwise_dists = ((samples_x[:,None,:] - samples_y[None,:,:])**2).sum(-1)
    if h < 0: # use the median trick
        if detach:
            h = torch.median(tmp2).detach()
        else:
            h = torch.median(tmp2)
        h = torch.sqrt(0.5 * h /np.log(samples_x.shape[0] + 1))

    kxy = torch.exp(- tmp2/ h**2 / 2)
    kxy = kxy * torch.eye(d)
    kxy = torch.matmul(torch.inverse(Q1/2+Q2/2).unsqueeze(0).unsqueeze(1), kxy)
    
    if get_width:
        return kxy, h 
    else:
        return kxy # shape (n,n,d,d)



def laplace_kernel(samples_x, samples_y, h = -1,get_width = False, detach=False):
    '''
    samples_x: shape = [n,d]
    samples_y: shape = [n,d]
    '''
    pairwise_dists = torch.abs((samples_x[:,None,:] - samples_y[None,:,:])).sum(-1)
    if h < 0: # use the median trick
        h = torch.median(pairwise_dists).detach()
        h = h /np.log(samples_x.shape[0] + 1)
    kxy = torch.exp(- pairwise_dists/ h )
    if get_width:
        return kxy, h 
    else:
        return kxy


def IMQ_kernel(samples_x, samples_y, h = -1, get_width = False, detach=False):
    '''
    samples_x: shape = [n,d]
    samples_y: shape = [n,d]
    '''
    pairwise_dists = ((samples_x[:,None,:] - samples_y[None,:,:])**2).sum(-1)
    if h < 0: # use the median trick
        h = torch.median(pairwise_dists).detach()
        h = h /np.log(samples_x.shape[0] + 1)
    kxy = (1 + pairwise_dists/ h )**(-0.5)
    if get_width:
        return kxy, h 
    else:
        return kxy


def Riesz_kernel(samples_x, samples_y, h = -1, get_width = False, detach=False):
    '''
    samples_x: shape = [n,d]
    samples_y: shape = [n,d]
    '''
    pairwise_dists = (-(samples_x[:,None,:] - samples_y[None,:,:]).norm(1,dim=-1) 
                      + (samples_x[:,None,:]).norm(1,dim=-1) 
                      + (samples_y[None,:,:]).norm(1,dim=-1))
    if h < 0: # use the median trick
        h = torch.median(pairwise_dists).detach()
        h = h /np.log(samples_x.shape[0] + 1)
    kxy = pairwise_dists/ h
    if get_width:
        return kxy, h 
    else:
        return kxy
    
if __name__ == "__main__":
    test_x = torch.rand(2,2)
    test_y = torch.rand(2,2)
    rst1 = gaussian_kernel(test_x, test_y)
    rst2 = matrix_kernel(test_x, test_y)
    print(rst1)
    print(rst2)
