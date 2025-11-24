"""
Defines various linear solvers for ELM-FBPINNs.
"""

import scipy
import scipy.sparse as sps
from jax import numpy as jnp
import numpy as np

from fbpinns.util.logger import logger

# NOTE

    # w is the number of models that overlap _at the center of the subdomain_ along 1 dimension
    # (w^d) is the number of models that overlap at the center of the domain
    # and ((2*w-1)^d) is the total number of models that contribute to the subdomain

    # using C collocation points per model,
        # Mj are size C * (w^d) * C
        # M and Q are size J * C * (w^d) * C
        # A is size J * ((2*w-1)^d) * C * C
        # Aj are size ((2*w-1)^d) * C * ((2*w-1)^d) * C
        # Ainv before sum is J * ((2*w-1)^d) * C * ((2*w-1)^d) * C
        # i.e. AS precondition has much worse memory performance


## not used for reference only
def _qr_inverse(A):
    "Computes inverse of square matrix A using QR decomposition"
    Q, R = scipy.linalg.qr(A)# A = QR, A^{-1} = R^{-1}Q^T  # QR should remain stable, but R can be poorly conditioned if A is
    Ainv = scipy.linalg.solve_triangular(R, Q.T)# can be unstable if R is poorly conditioned
    return Ainv

def _truncated_svd_inverse(A, hermitian):
    """Computes truncated SVD inverse of A. Same as scipy.linalg.pinv / pinvh.
    This is a regularised inversion which is more stable for high condition numbers"""

    U, s, Vh = np.linalg.svd(A, hermitian=hermitian, full_matrices=False)
    eps = np.finfo(A.dtype).eps
    tol = eps*np.max(s)*A.shape[0]# rough tolerance for float64
    mask = s > tol
    logger.info(f"{1-np.sum(mask)/len(mask):.1%} singular values dropped")
    s_inv = np.array([1/si if si > tol else 0.0 for si in s])
    Ainv = Vh.T @ np.diag(s_inv) @ U.T
    return Ainv
##


def txt_matrix(filename, M):
    "Save matrix to text file"

    # Set the desired precision for formatting
    width = 10  # Total width for each element (including the decimal point and the number)
    fmt = lambda x: f"{x:.2f}" if abs(x) < 1e6 else f"{x:1.2e}"

    # Open a text file for writing
    with open(filename, mode='w') as file:
        for row in M:
            # Format each element with the specified precision and a consistent width
            formatted_row = ' '.join([fmt(elem).rjust(width) for elem in row])
            # Write the formatted row to the text file
            file.write(formatted_row + '\n')

def report_inverse_error(A, Ainv):
    "Report the residual error of an inverse matrix"

    I = np.eye(A.shape[0])
    res1 = np.linalg.norm(I - A @ Ainv)
    res2 = np.linalg.norm(I - Ainv @ A)
    logger.info(f"‖I - A·Ainv‖ = {res1:1.1e}")
    logger.info(f"‖I - Ainv·A‖ = {res2:1.1e}")

def report_qr_error(A, Q, R):
    """Report quality metrics for QR decomposition A ≈ Q R"""

    I = np.eye(Q.shape[1])  # Q is usually (m x n), Q.T Q is (n x n)
    orthogonality_error = np.linalg.norm(I - Q.T @ Q)
    reconstruction_error = np.linalg.norm(A - Q @ R) / np.linalg.norm(A)
    logger.info(f"‖I - Q^T Q‖ = {orthogonality_error:1.1e}")
    logger.info(f"‖A - Q R‖/||A|| = {reconstruction_error:1.1e}")

def solve_error(A, b, x_approx):
    "Return the error of the solve for Ax = b"

    r = b - (A@x_approx)
    r1 = np.linalg.norm(r)
    r2 = np.linalg.norm(b)
    return r1/r2

_matrix_report_max_size = 128

def build_system(terms_left, terms_right, takess, constraints_left, J, C, J_active, a0_fixed,
                 build_normal):
    """Build M and f in the (active parameters) least squares system ||Ma - f||^2.
    Optionally also build A and b in the associated normal equation Aa = b"""

    matrices = []
    for ic, (term_left, term_right, takes, constraint_left) in enumerate(
            zip(terms_left, terms_right, takess, constraints_left)):
        m_take, n_take = takes[:2]
        # m_take is model relative to all_ims
        # n_take is point relative to each constraint

        # check shapes
        S = m_take.shape[0]# nonzero values
        N = constraint_left[0].shape[0]
        logger.info(f"Constraint {ic} sparsity factor: {S/((N+1e-10)*J):.2%}")
        assert term_left[0].shape == (S, 1, C)
        assert term_left[1].shape == (N, 1)
        assert term_right[0].shape == (N, 1)

        # build sparse constraint matrix - vector
        values = term_left[0].flatten()
        j = m_take.reshape((S,1))*C + jnp.arange(C, dtype=int).reshape((1,C))
        j = j.flatten()
        i = n_take.reshape((S,1)) + jnp.zeros(C, dtype=int).reshape((1,C))
        i = i.flatten()
        M = sps.coo_array((values, (i, j)), shape=(N, J*C))
        M = M.tocsr()
        # TODO: write my own COO->CSR->gram->addition JAX operators
        # M, A should be able to be computed efficiently using segment_sum / scatter / gather .at index ops
        #M = js.COO((values, jnp.stack([i, j], axis=1)), shape=(N, J*C))
        f = term_left[1]-term_right[0]
        if N>0:# normalise terms over batch
            M, f = M/np.sqrt(N), f/np.sqrt(N)

        # reduce linear system if some parameters are fixed, convert to normal form
        Mall = M.copy()
        fall = f.copy()
        if J_active < J:
            Ma, Mf = M[:,:J_active*C], M[:,J_active*C:]
            # get normal equation terms
            if build_normal:
                A11 = Ma.T @ Ma
                A12 = Ma.T @ Mf
                A = A11
                b = Ma.T @ f - A12 @ a0_fixed
            M = Ma
            f = f - Mf @ a0_fixed
        else:
            if build_normal:
                A = M.T @ M
                b = M.T @ f
        if not build_normal:
            A = sps.coo_matrix(([0.0], ([0], [0])), shape=(1, 1))# empty placeholder
            b = jnp.array([[0.]])# empty placeholder

        matrices.append((M,f,A,b,Mall,fall))

    # convert to single linear system
    M,f,A,b,Mall,fall = matrices[0]
    for Mp,fp,Ap,bp,_,_ in matrices[1:]:
        M = sps.vstack((M,Mp))# sparse
        f = jnp.vstack((f,fp))# dense
        A += Ap# sparse
        b += bp# dense
        Mall = sps.vstack((M,Mall))# sparse
        fall = jnp.vstack((f,fall))# dense
    #txt_matrix("M.txt", M.toarray())
    #txt_matrix("A.txt", A.toarray())

    if M.shape[1] > M.shape[0]:
        logger.warn(f"M matrix is under determined: {M.shape}")

    return M,f,A,b,Mall,fall


def rrqr(A, sigma):
    """Rank revealing QR decomposition of matrix A.
    Computes A[:, p] == Q @ R where p is chosen such that the diagonal of R is non-increasing,
    and rank = np.sum(np.abs(np.diag(R)) > np.abs(R[0,0])*sigma).
    """

    # in general A = QR  with A = (M, N), Q = (M, M), R = (M, N), R upper triangular
    # the rank reduced form is A_reduced = A[:p[:r]] = Q R[:r] = Q[:,:r] R[:r,:r] (as R is upper triangular) = (M, r)

    # in scipy economic mode:
    # for N <= M: Qj has shape (M, N), Rj has shape (N, N), p has shape (N,), diag R has shape (N,)
    # for N >  M: Qj has shape (M, M), Rj has shape (M, N), p has shape (N,), diag R has shape (M,) (and rank can only be <=M < N)

    # but for both cases, Q[:,:r] has rank reduced shape of (M, k) and R[:r,:r] has rank reduced shape of (k, k)
    # i.e. we can treat both cases the same when populating the global Q and R matrices below

    assert sigma <= 1

    #Q, R, p = jax.scipy.linalg.qr(A, mode="economic", pivoting=True)
    Q, R, p = scipy.linalg.qr(A, mode="economic", pivoting=True)# QR should remain stable, but R can be poorly conditioned if A is?
    # computes A[:, P] == Q @ R
    # where P is chosen such that the diagonal of R is non-increasing

    # Estimate rank based on tolerance
    #print(np.diag(R))
    diag = np.abs(np.diag(R))
    rank = np.sum(diag > diag[0]*sigma)# define tolerance relative to leading diagonal value

    return [Q, R, p, rank]

def block_rrqr_preconditioner(terms_left, takess, constraints_left, J_active, C, sigma):
    """Applies block rrqr preconditioning to M.
    Takes the rank revealing qr decomposition of each block in M, and defines the global MR^{-1} = Q
    matrix as the preconditioned system.
    """

    Qjs, Rjs, pjs, Ijs, Jjs, rankjs = [],[],[],[],[],[]
    # TODO: potentially vmap rrqr: would expect most subdomains to have similar numbers of points
    # unfortunately Mjs are not regular (variable nrows), but perhaps eventually we could vmap rrqr over rows..
    for j in range(J_active):

        # get subdomain block matrix
        Mj, Ij,Jj = [], [],[]
        N_offset = 0
        for term_left, takes, constraint_left in zip(terms_left, takess, constraints_left):
            m_take, n_take = takes[:2]
            N = constraint_left[0].shape[0]

            mask = (m_take == j)# all points-models in subdomain
            Sj = int(jnp.sum(mask))

            # matrix values
            value_rows = term_left[0][mask][:,0,:]
            if N>0:
                value_rows = value_rows/jnp.sqrt(N)

            # value indices in global stacked M matrix
            j_rows = m_take[mask].reshape((Sj,1))*C + jnp.arange(C, dtype=int).reshape((1,C))
            i_rows = n_take[mask].reshape((Sj,1)) + jnp.zeros(C, dtype=int).reshape((1,C)) + N_offset

            Mj.append(value_rows), Ij.append(i_rows), Jj.append(j_rows)
            N_offset += N

        Mj = jnp.concatenate(Mj)
        Jj = jnp.concatenate(Jj)
        Ij = jnp.concatenate(Ij)
        assert Mj.shape == Jj.shape == Ij.shape
        assert Mj.shape[0] > 0 and Mj.shape[1] == C

        # do rrqr decomposition on Mj (numpy from here onwards)
        Qj, Rj, pj, rankj = rrqr(Mj, sigma)
        #print(Qj.shape, Rj.shape, pj.shape, rankj)

        Qjs.append(Qj), Rjs.append(Rj), pjs.append(pj), Ijs.append(Ij), Jjs.append(Jj), rankjs.append(rankj)
        #txt_matrix(f"Qj_{j}.txt", Qj)
        #txt_matrix(f"Rj_{j}.txt", Rj)
        #txt_matrix(f"Jj_{j}.txt", Jj)

        if j == int(J_active)//2:
            logger.info(f"Mj={j} {Mj.shape}, {Mj.dtype}, {np.linalg.cond(Mj) if (Mj.shape[0]<_matrix_report_max_size and Mj.shape[1]<_matrix_report_max_size) else np.nan:1.1e}")
            report_qr_error(Mj[:,pj], Qj, Rj)

    # get all droped columns based on rank
    p = np.concatenate([j*C+pj[:rankj] for j,(pj,rankj) in enumerate(zip(pjs, rankjs))])
    drop_pct = np.array([1-rankj/Rj.shape[1] for Rj,rankj in zip(Rjs,rankjs)])
    drop_pct_total = 1-len(p)/(J_active*C)
    logger.info(f"Dropped {drop_pct_total:.1%} columns during block_rrqr_precondition (sigma={sigma:.1e}, min/max drop %: {drop_pct.min():.1%}, {drop_pct.max():.1%})")
    j_map = np.zeros(J_active*C, dtype=int)
    j_map[p] = np.arange(len(p))

    # build global sparse rank-reduced Q (M, :r)
    values_Q = np.concatenate([Qj[:,:rankj].flatten() for Qj, pj, rankj in zip(Qjs, pjs, rankjs)])# get rank-reduced Qj
    logger.debug(values_Q.shape)
    i = np.concatenate([Ij[:,:rankj].flatten() for Ij, pj, rankj in zip(Ijs, pjs, rankjs)])
    j = np.concatenate([Jj[:,pj[:rankj]].flatten() for Jj, pj, rankj in zip(Jjs, pjs, rankjs)])# put Qj in global matrix, but map onto global pj-ordered columns
    j = j_map[j]# convert global pj ordering of columns back to original Qj ordering, reduced on pj
    Q = sps.coo_matrix((values_Q, (i, j)), shape=(N_offset, len(p))).tocsc()

    # build global sparse rank-reduced R (:r, :r)
    values_R = np.concatenate([Rj[:,:rankj][:rankj,:].flatten() for Rj, pj, rankj in zip(Rjs, pjs, rankjs)])# get rank-reduced Rj
    logger.debug(values_R.shape)
    i = np.concatenate([((j*C+pj[:rankj]).reshape((-1,1))+np.zeros(rankj, dtype=int).reshape((1,rankj))).flatten()
                        for j,(pj,rankj) in enumerate(zip(pjs, rankjs))])# put Rj in global matrix, but map onto global pj-ordered columns
    j = np.concatenate([((j*C+pj[:rankj]).reshape((1,-1))+np.zeros(rankj, dtype=int).reshape((rankj,1))).flatten()
                        for j,(pj,rankj) in enumerate(zip(pjs, rankjs))])
    i, j = j_map[i], j_map[j]# convert global pj ordering of columns back to original Rj ordering, reduced on pj
    R = sps.coo_matrix((values_R, (i, j)), shape=(len(p), len(p))).tocsc()

    #txt_matrix("Q.txt", Q.toarray())
    #txt_matrix("R.txt", R.toarray())

    return Q, R, p, drop_pct_total


def additive_schwarz_preconditioner(terms_left, takess, constraints_left, J_active, C):
    """Applies additive Schwarz preconditioning to normal matrix A = M^T@M, as defined in Shang et al 2024.
    """

    # build and invert Aj matrices
    # Aj are of shape q x q, where q are _all_ the basis functions which are non-zero in j
    # note that global A contains off-diagonal blocks where subdomains overlap

    # get m_take, n_take indices for global M
    N_offset = 0
    m_take_all, n_take_all = [],[]
    for takes, constraint_left in zip(takess, constraints_left):
        m_take, n_take = takes[:2]
        N = constraint_left[0].shape[0]
        m_take_all.append(m_take)
        n_take_all.append(n_take + N_offset)
        N_offset += N
    m_take_all = jnp.concatenate(m_take_all)
    n_take_all = jnp.concatenate(n_take_all)
    N_total = N_offset

    Ainvjs, Ijs, Jjs = [], [], []
    # TODO: potentially vmap: would expect most subdomains to have similar numbers of points
    # unfortunately Ajs are not regular (variable nmodels), but could pad perhaps?
    for j in range(J_active):

        # find all (active) models which overlap j
        # find all colocation points which contribute to these models
        c = jnp.unique(n_take_all[(m_take_all == j)])# all unique points in subdomain
        mask = jnp.isin(n_take_all, c)# all points-models where point is in subdomain
        m_j = jnp.unique(m_take_all[mask & (m_take_all<J_active)])# all unique ACTIVE models which have a point in subdomain
        n_j = jnp.unique(n_take_all[jnp.isin(m_take_all, m_j)])# all unique points which cover all unique models
        #print(m_j)
        #print(n_j)

        # make index maps for reduced Mj matrix
        Mj = jnp.zeros((len(n_j), len(m_j)*C))
        j_index = jnp.zeros(J_active, dtype=int)
        j_index = j_index.at[m_j].set(jnp.arange(len(m_j), dtype=int))# reduced index
        i_index = jnp.zeros(N_total, dtype=int)
        i_index = i_index.at[n_j].set(jnp.arange(len(n_j), dtype=int))# reduced index
        #print(Mj.shape)

        N_offset = 0
        for term_left, takes, constraint_left in zip(terms_left, takess, constraints_left):
            m_take, n_take = takes[:2]
            N = constraint_left[0].shape[0]
            n_take += N_offset

            mask = jnp.isin(m_take, m_j)# all points-models in m_j
            Sj = int(jnp.sum(mask))

            # matrix values
            value_rows = term_left[0][mask][:,0,:]
            if N>0:
                value_rows = value_rows/jnp.sqrt(N)

            # build reduced dense Mj matrix
            #print(m_take[mask])
            #print(n_take[mask])
            #print(j_index[m_take[mask]])
            #print(i_index[n_take[mask]])
            j_rows = j_index[m_take[mask]].reshape((Sj,1))*C + jnp.arange(C, dtype=int).reshape((1,C))
            i_rows = i_index[n_take[mask]].reshape((Sj,1)) + jnp.zeros(C, dtype=int).reshape((1,C))
            Mj = Mj.at[i_rows.flatten(), j_rows.flatten()].set(value_rows.flatten())
            N_offset += N

        # create Aj
        Aj = Mj.T @ Mj

        # invert Aj (be careful - Aj can be highly ill-conditioned still)
        # QR / LU inversion is unstable for large condition number without regularisation
        #Ainvj = _qr_inverse(Aj)# TODO: convert to jax qr inverse + vmap (np from here)
        #Ainvj = scipy.linalg.inv(Aj)
        #Ainvj = _truncated_svd_inverse(Aj, hermitian=True)
        Ainvj, rank = scipy.linalg.pinvh(Aj, return_rank=True)# scipy version of my truncated_svd_inverse
        if j == int(J_active)//2:
            logger.info(f"Aj={j} {Aj.shape}, {Aj.dtype}, {np.linalg.cond(Aj) if (Aj.shape[0]<_matrix_report_max_size and Aj.shape[1]<_matrix_report_max_size) else np.nan:1.1e}")
            logger.info(f"{1-rank/Aj.shape[0]:.1%} singular values dropped")
            report_inverse_error(Aj, Ainvj)

        # get global coordinates of Ainvj elements
        L = C*len(m_j)
        rs = np.concatenate([j*C+np.arange(C, dtype=int) for j in m_j])
        Ij = rs.reshape((-1,1))+np.zeros(L, dtype=int).reshape((1,L))
        Jj = rs.reshape((1,-1))+np.zeros(L, dtype=int).reshape((L,1))

        Ijs.append(Ij)
        Jjs.append(Jj)
        Ainvjs.append(Ainvj)

        #if j in [0,1,2]:
            #txt_matrix(f"Mj_{j}.txt", Mj)
            #txt_matrix(f"Aj_{j}.txt", Aj)
            #txt_matrix(f"Ainvj_{j}.txt", Ainvj)
            #txt_matrix(f"Pj_{j}.txt", Ainvj @ Aj)

        #v, i, j = Ainvj.flatten(), Ij.flatten(), Jj.flatten()
        #Ainvj = sps.coo_matrix((v, (i, j)), shape=(J_active*C, J_active*C)).tocsc()

    # = Ainvjs[0]
    #for b in Ainvjs[1:]:
    #    Ainv += b

    # add sparse arrays together
    values_Ainv = np.concatenate([Ainvj.flatten() for Ainvj in Ainvjs])
    logger.debug(values_Ainv.shape)
    i = np.concatenate([Ij.flatten() for Ij in Ijs])
    j = np.concatenate([Jj.flatten() for Jj in Jjs])
    Ainv = sps.coo_matrix((values_Ainv, (i, j)), shape=(J_active*C, J_active*C)).tocsc()
    # By default when converting to CSR or CSC format, duplicate (i,j) entries will be summed

    #txt_matrix("Ainv.txt", Ainv.toarray())

    return Ainv



def report_matrices(matrices, save_results, outdir):
    "Prints sparse matrices properties, and saves matrices"

    for name, matrix in matrices:
        logger.info(f"{name} {matrix.shape}")
    for name, matrix in matrices:
        logger.info(f"{name} nnz: {matrix.nnz} ({matrix.nnz * matrix.dtype.itemsize / 1e9:.4f} GB)")
    for name, matrix in matrices:
        logger.info(f"{name} sparsity factor: {matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.2%}")
        #logger.info(f"{name} mean absolute value {np.mean(matrix.data)}")
    for name, matrix in matrices:
        if save_results:
            sps.save_npz(f"{outdir}/{name}.npz", matrix)
            if matrix.shape[0] < 8*_matrix_report_max_size and matrix.shape[1] < 8*_matrix_report_max_size:
                dense = matrix.toarray()
                svals = np.linalg.svd(dense, compute_uv=False)
                cn = svals[0] / svals[-1]
                logger.info(f'{name} SVD-based condition number: {cn:.2e}')
                #logger.info(f'{name} np.linalg condition number: {np.linalg.cond(dense):.2e}')


def _callback_factory(Mall, fall, a0_fixed, J_active, C, testing):
    _, _, test, active_params, all_test_inputs = testing
    def callback(i, xi):# standard callback when solver directly solves for a
        a = xi
        losses = test(i, Mall, fall, a, a0_fixed, J_active, C, active_params,
                      *all_test_inputs)
        return losses
    return callback

def linear_solver(terms_left, terms_right, takess, constraints_left, J, C, J_active, a0_fixed, kwargs,
                  testing):

    # build the system
    build_normal = kwargs["system"] == "normal"
    M,f,A,b,Mall,fall = build_system(terms_left, terms_right, takess, constraints_left, J, C, J_active, a0_fixed, build_normal)

    # report matrices
    matrices = [("M", M)]
    if build_normal: matrices += [("A", A)]
    report_matrices(matrices, kwargs["save_results"], testing[0])

    # solve system
    solver, solver_kwargs = kwargs["solver"], kwargs["solver_kwargs"]
    callback, test_freq = _callback_factory(Mall, fall, a0_fixed, J_active, C, testing), testing[1]
    if build_normal:
        a, info, _ = solver(A, b[:,0], callback, test_freq, solver_kwargs)
        logger.info(f"A a = b solve error: {solve_error(A, b[:,0], a):1.1e}")
    else:
        a, info, _ = solver(M, f[:,0], callback, test_freq, solver_kwargs)
        logger.info(f"M a = f solve error: {solve_error(M, f[:,0], a):1.1e}")

    return a, info, Mall, fall

def block_rrqr_solver(terms_left, terms_right, takess, constraints_left, J, C, J_active, a0_fixed, kwargs,
                      testing):

    # build the system
    build_normal = kwargs["system"] == "normal"
    M,f,A,b,Mall,fall = build_system(terms_left, terms_right, takess, constraints_left, J, C, J_active, a0_fixed, build_normal)

    # build preconditioner
    sigma = kwargs["sigma"]
    Q, R, p, drop_pct_total = block_rrqr_preconditioner(terms_left, takess, constraints_left, J_active, C, sigma)
    # test if decomposition is correct
    M_reduced = M.tocsr()[:, p]
    diff = Q @ R - M_reduced
    err = sps.linalg.norm(diff) / (sps.linalg.norm(M_reduced) + 1e-15)
    logger.info(f"Q @ R - M_reduced error: {err:1.1e}")
    assert err < 1e-6, "Q @ R - M_reduced error too large"

    # report matrices
    matrices = [("M", M), ("M_reduced", M_reduced), ("Q", Q), ("R", R)]
    if build_normal: matrices += [("A", A)]
    report_matrices(matrices, kwargs["save_results"], testing[0])

    # solve system
    solver, solver_kwargs = kwargs["solver"], kwargs["solver_kwargs"]
    _, test_freq, test, active_params, all_test_inputs = testing
    def callback(i, xi):# custom callback for this solver
        y = xi
        a_prec = sps.linalg.spsolve_triangular(R, y, lower=False)
        a = jnp.zeros(J_active*C)
        a = a.at[p].set(a_prec)
        losses = test(i, Mall, fall, a, a0_fixed, J_active, C, active_params,
                      *all_test_inputs)
        return losses
    # first, solve preconditioned system
    if build_normal:
        Q_gramian = Q.T @ Q
        f_gramian = Q.T @ f
        y, info, _ = solver(Q_gramian, f_gramian[:,0], callback, test_freq, solver_kwargs)
        logger.info(f"Q_gramian y = f_gramian solve error: {solve_error(Q_gramian, f_gramian[:,0], y):1.1e}")
    else:
        y, info, _ = solver(Q, f[:,0], callback, test_freq, solver_kwargs)
        logger.info(f"Q y = f solve error: {solve_error(Q, f[:,0], y):1.1e}")
    # then solve original system
    #txt_matrix("R.txt", R.toarray())
    #a_prec,_ = solver(R, y, **solver_kwargs)
    a_prec = sps.linalg.spsolve_triangular(R, y, lower=False)# can be unstable if R is poorly conditioned
    #a_prec = sps.linalg.spsolve(R, y)# can be unstable if R is poorly conditioned
    logger.info(f"R a = y solve error: {solve_error(R, y, a_prec):1.1e}")
    a = jnp.zeros(J_active*C)
    a = a.at[p].set(a_prec)# fill in columns that were not reduced

    info = info + (drop_pct_total,)

    return a, info, Mall, fall

def additive_schwarz_solver(terms_left, terms_right, takess, constraints_left, J, C, J_active, a0_fixed, kwargs,
                            testing):

    # build the system
    build_normal = True
    M,f,A,b,Mall,fall = build_system(terms_left, terms_right, takess, constraints_left, J, C, J_active, a0_fixed, build_normal)

    # build preconditioner
    Ainv = additive_schwarz_preconditioner(terms_left, takess, constraints_left, J_active, C)
    P = Ainv @ A
    #txt_matrix("P.txt", P.toarray())

    # report matrices
    matrices = [("M", M), ("A", A), ("Ainv", Ainv), ("Ainv A", P)]
    report_matrices(matrices, kwargs["save_results"], testing[0])

    # solve system
    solver, solver_kwargs = kwargs["solver"], kwargs["solver_kwargs"]
    callback, test_freq = _callback_factory(Mall, fall, a0_fixed, J_active, C, testing), testing[1]
    solver_kwargs["M"] = Ainv
    a, info, _ = solver(A, b[:,0], callback, test_freq, solver_kwargs)
    logger.info(f"A a = b solve error: {solve_error(A, b[:,0], a):1.1e}")

    return a, info, Mall, fall


class LinearSolverBase:
    "Base linear optimiser class"
    def __init__(self, **kwargs):
        self.update = kwargs
        self.update["solver_fn"] = None
        raise NotImplementedError
    def init(self, trainable_params):
        return []

class LinearSolver(LinearSolverBase):
    "Solve least squares system ||Ma - f||^2 without preconditioning"
    def __init__(self, **kwargs):
        self.update = kwargs
        self.update["solver_fn"] = linear_solver

class BlockRRQRLinearSolver(LinearSolverBase):
    "Solve least squares system ||Ma - f||^2 with block RRQR preconditioning"
    def __init__(self, **kwargs):
        self.update = kwargs
        self.update["solver_fn"] = block_rrqr_solver

class AdditiveSchwarzLinearSolver(LinearSolverBase):
    "Solve least squares system ||Ma - f||^2 with additive Schwarz preconditioning"
    def __init__(self, **kwargs):
        self.update = kwargs
        self.update["solver_fn"] = additive_schwarz_solver
