"""
Simple 2D Helmholtz solver
"""


import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def helmholtz_solver(nx, ny, dx, dy, w, c, f):

    # get sparse laplacian
    dx2 = dx ** 2
    dy2 = dy ** 2
    main_diag = (-2/dx2-2/dy2) * np.ones(nx*ny)
    off_diag = (1/dy2) * np.ones(nx*ny-1)
    off_diag2 = (1/dx2) * np.ones((nx-1)*ny)
    A = diags([main_diag, off_diag, off_diag, off_diag2, off_diag2], [0, -1, 1, -ny, ny])

    # add freq term
    A += diags(((w/c)**2).flatten())

    # define 0 boundary condition
    u_bc = np.zeros((nx,ny)).flatten()
    mask = np.zeros((nx,ny), dtype=int)
    mask[0,:] = mask[-1,:] = mask[:,0] = mask[:,-1] = 1
    mask = mask.flatten() == 1

    # solve constrained linear system
    b = f.flatten() - A.dot(u_bc)
    A_masked = A.tocsc()[~mask, :][:, ~mask]# cant crop a diag - becomes non-diag
    b_masked = b[~mask]
    u_free = spsolve(A_masked, b_masked)

    # output u
    u = u_bc.copy()
    u[~mask] = u_free
    u = u.reshape(nx,ny)

    return u



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from fbpinns.util.other import Timer

    #c = np.load("marmousi_crop.npy")
    #c = c/np.median(c)
    #nx, ny = c.shape

    nx, ny = 400, 401
    c = np.ones((nx, ny))

    w=2*np.pi/0.2
    sd=0.05

    x,dx = np.linspace(0,1,nx,retstep=True)
    y,dy = np.linspace(0,1,ny,retstep=True)
    xx,yy = np.meshgrid(x,y,indexing="ij")
    f = (1/np.sqrt(((2*np.pi)**2)*((sd**2)**2)))*np.exp(-0.5*(((xx-0.5)/sd)**2 + ((yy-0.5)/sd)**2))

    with Timer("solver"):
        u = helmholtz_solver(nx, ny, dx, dy, w, c, f)

    plt.figure()
    plt.imshow(f)
    plt.colorbar()
    plt.figure()
    plt.imshow(c)
    plt.colorbar()
    plt.show()
    plt.imshow(u)
    plt.colorbar()
    plt.show()

