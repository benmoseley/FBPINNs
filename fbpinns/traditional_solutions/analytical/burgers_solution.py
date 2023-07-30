#! /usr/bin/env python3
#
def burgers_solution_test ( ):

#*****************************************************************************80
#
## BURGERS_SOLUTION_TEST tests the BURGERS_SOLUTION library.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    27 September 2015
#
#  Author:
#
#    John Burkardt
#
  import platform

  print ( '' )
  print ( 'BURGERS_SOLUTION_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  BURGERS_SOLUTION evaluates exact solutions of the Burgers equation.' )

  burgers_viscous_time_exact1_test01 ( )
  burgers_viscous_time_exact1_test02 ( )
  burgers_viscous_time_exact2_test01 ( )
  burgers_viscous_time_exact2_test02 ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'BURGERS_SOLUTION_TEST:' )
  print ( '  Normal end of execution.' )
  return

def burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt ):

#*****************************************************************************80
#
## BURGERS_VISCOUS_TIME_EXACT1 evaluates a solution to the Burgers equation.
#
#  Discussion:
#
#    The form of the Burgers equation considered here is
#
#      du       du        d^2 u
#      -- + u * -- = nu * -----
#      dt       dx        dx^2
#
#    for -1.0 < x < +1.0, and 0 < t.
#
#    Initial conditions are u(x,0) = - sin(pi*x).  Boundary conditions
#    are u(-1,t) = u(+1,t) = 0.  The viscosity parameter nu is taken
#    to be 0.01 / pi, although this is not essential.
#
#    The authors note an integral representation for the solution u(x,t),
#    and present a better version of the formula that is amenable to
#    approximation using Hermite quadrature.
#
#    This program library does little more than evaluate the exact solution
#    at a user-specified set of points, using the quadrature rule.
#    Internally, the order of this quadrature rule is set to 8, but the
#    user can easily modify this value if greater accuracy is desired.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    24 September 2015
#
#  Author:
#
#    John Burkardt.
#
#  Reference:
#
#    Claude Basdevant, Michel Deville, Pierre Haldenwang, J Lacroix,
#    J Ouazzani, Roger Peyret, Paolo Orlandi, Anthony Patera,
#    Spectral and finite difference solutions of the Burgers equation,
#    Computers and Fluids,
#    Volume 14, Number 1, 1986, pages 23-41.
#
#  Parameters:
#
#    Input, real NU, the viscosity.
#
#    Input, integer VXN, the number of spatial grid points.
#
#    Input, real VX(VXN), the spatial grid points.
#
#    Input, integer VTN, the number of time grid points.
#
#    Input, real VT(VTN), the time grid points.
#
#    Output, real VU(VXN,VTN), the solution of the Burgers
#    equation at each space and time grid point.
#
  import numpy as np

  qn = 60
#
#  Compute the rule.
#
  qx, qw = hermite_ek_compute ( qn )
#
#  Evaluate U(X,T) for later times.
#
  vu = np.zeros ( [ vxn, vtn ] )

  for vti in range ( 0, vtn ):

    if ( vt[vti] == 0.0 ):

      for i in range ( 0, vxn ):
        vu[i,vti] = - np.sin ( np.pi * vx[i] )

    else:

      for vxi in range ( 0, vxn ):

        top = 0.0
        bot = 0.0

        for qi in range ( 0, qn ):

          c = 2.0 * np.sqrt ( nu * vt[vti] )

          top = top - qw[qi] * c * np.sin ( np.pi * ( vx[vxi] - c * qx[qi] ) ) \
            * np.exp ( - np.cos ( np.pi * ( vx[vxi] - c * qx[qi]  ) ) \
            / ( 2.0 * np.pi * nu ) )

          bot = bot + qw[qi] * c \
            * np.exp ( - np.cos ( np.pi * ( vx[vxi] - c * qx[qi]  ) ) \
            / ( 2.0 * np.pi * nu ) )

          vu[vxi,vti] = top / bot

  return vu

def burgers_viscous_time_exact1_test01 ( ):

#*****************************************************************************80
#
##  BURGERS_VISCOUS_TIME_EXACT1_TEST01 tests sets up a small test case.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    24 September 2015
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  vtn = 11
  vxn = 11
  nu = 0.01 / np.pi

  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT1_TEST01' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  BURGERS_VISCOUS_TIME_EXACT1 evaluates solution #1' )
  print ( '  to the Burgers equation.' )
  print ( '' )
  print ( '  Viscosity NU = %g' % ( nu ) )
  print ( '  NX = %d' % ( vxn ) )
  print ( '  NT = %d' % ( vtn ) )

  xlo = -1.0
  xhi = +1.0
  vx = np.linspace ( xlo, xhi, vxn )
  r8vec_print ( vxn, vx, '  X grid points:' )

  tlo = 0.0
  thi = 3.0 / np.pi
  vt = np.linspace ( tlo, thi, vtn )
  r8vec_print ( vtn, vt, '  T grid points:' )

  vu = burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt )

  r8mat_print ( vxn, vtn, vu, '  U(X,T) at grid points:' )

  filename = 'burgers_solution_test01.txt'

  r8mat_write ( filename, vxn, vtn, vu )

  print ( '' )
  print ( '  Data written to file "%s"' % ( filename ) )
#
#  Terminate
#
  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT1_TEST01' )
  print ( '  Normal end of execution.' )
  return

def burgers_viscous_time_exact1_test02 ( ):

#*****************************************************************************80
#
## BURGERS_VISCOUS_TIME_EXACT1_TEST02 tests sets up a finer test case.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    24 September 2015
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  vtn = 41
  vxn = 41
  nu = 0.01 / np.pi

  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT1_TEST02' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  BURGERS_VISCOUS_TIME_EXACT1 computes solution #1' )
  print ( '  to the Burgers equation.' )
  print ( '' )
  print ( '  Viscosity NU = %g' % ( nu ) )
  print ( '  NX = %d' % ( vxn ) )
  print ( '  NT = %d' % ( vtn ) )

  xlo = -1.0
  xhi = +1.0
  vx = np.linspace ( xlo, xhi, vxn )
  r8vec_print ( vxn, vx, '  X grid points:' )

  tlo = 0.0
  thi = 3.0 / np.pi
  vt = np.linspace ( tlo, thi, vtn )
  r8vec_print ( vtn, vt, '  T grid points:' )

  vu = burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt )

  filename = 'burgers_solution_test02.txt'

  r8mat_write ( filename, vxn, vtn, vu )

  print ( '' )
  print ( '  Data written to file "%s"' % ( filename ) )
#
#  Terminate
#
  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT1_TEST02' )
  print ( '  Normal end of execution.' )
  return

def burgers_viscous_time_exact2 ( nu, xn, x, tn, t ):

#*****************************************************************************80
#
## BURGERS_VISCOUS_TIME_EXACT2 evaluates a solution to the Burgers equation.
#
#  Discussion:
#
#    The form of the Burgers equation considered here is
#
#      du       du        d^2 u
#      -- + u * -- = nu * -----
#      dt       dx        dx^2
#
#    for 0.0 < x < 2 Pi, and 0 < t.
#
#    The initial condition is
#
#      u(x,0) = 4 - 2 * nu * dphi(x,0)/dx / phi(x,0)
#
#    where
#
#      phi(x,t) = exp ( - ( x-4*t      ) / ( 4*nu*(t+1) ) )
#               + exp ( - ( x-4*t-2*pi ) / ( 4*nu*(t+1) ) )
#
#    The boundary conditions are periodic:
#
#      u(0,t) = u(2 Pi,t)
#
#    The viscosity parameter nu may be taken to be 0.01, but other values
#    may be chosen.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    27 September 2015
#
#  Author:
#
#    John Burkardt.
#
#  Reference:
#
#    Claude Basdevant, Michel Deville, Pierre Haldenwang, J Lacroix,
#    J Ouazzani, Roger Peyret, Paolo Orlandi, Anthony Patera,
#    Spectral and finite difference solutions of the Burgers equation,
#    Computers and Fluids,
#    Volume 14, Number 1, 1986, pages 23-41.
#
#  Parameters:
#
#    Input, real NU, the viscosity.
#
#    Input, integer XN, the number of spatial grid points.
#
#    Input, real X(XN), the spatial grid points.
#
#    Input, integer TN, the number of time grid points.
#
#    Input, real T(TN), the time grid points.
#
#    Output, real U(XN,TN), the solution of the Burgers
#    equation at each space and time grid point.
#
  import numpy as np

  u = np.zeros ( [ xn, tn ] )

  for j in range ( 0, tn ):

    for i in range ( 0, xn ):

      a = ( x[i] - 4.0 * t[j] )
      b = ( x[i] - 4.0 * t[j] - 2.0 * np.pi )
      c = 4.0 * nu * ( t[j] + 1.0 )
      phi = np.exp ( - a * a / c ) + np.exp ( - b * b / c )
      dphi = - 2.0 * a * np.exp ( - a * a / c ) / c \
             - 2.0 * b * np.exp ( - b * b / c ) / c
      u[i,j] = 4.0 - 2.0 * nu * dphi / phi

  return u

def burgers_viscous_time_exact2_test01 ( ):

#*****************************************************************************80
#
##  BURGERS_VISCOUS_TIME_EXACT2_TEST01 tests sets up a small test case.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    27 September 2015
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  vtn = 11
  vxn = 11
  nu = 0.5

  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT2_TEST01' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  BURGERS_VISCOUS_TIME_EXACT2 evaluates solution #2' )
  print ( '  to the Burgers equation.' )
  print ( '' )
  print ( '  Viscosity NU = %g' % ( nu ) )
  print ( '  NX = %d' % ( vxn ) )
  print ( '  NT = %d' % ( vtn ) )

  xlo = 0.0
  xhi = 2.0 * np.pi
  vx = np.linspace ( xlo, xhi, vxn )
  r8vec_print ( vxn, vx, '  X grid points:' )

  tlo = 0.0
  thi = 1.0
  vt = np.linspace ( tlo, thi, vtn )
  r8vec_print ( vtn, vt, '  T grid points:' )

  vu = burgers_viscous_time_exact2 ( nu, vxn, vx, vtn, vt )

  r8mat_print ( vxn, vtn, vu, '  U(X,T) at grid points:' )

  filename = 'burgers_solution_test03.txt'

  r8mat_write ( filename, vxn, vtn, vu )

  print ( '' )
  print ( '  Data written to file "%s"' % ( filename ) )
#
#  Terminate
#
  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT2_TEST01' )
  print ( '  Normal end of execution.' )
  return

def burgers_viscous_time_exact2_test02 ( ):

#*****************************************************************************80
#
## BURGERS_VISCOUS_TIME_EXACT2_TEST02 tests sets up a finer test case.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    27 September 2015
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  vtn = 41
  vxn = 41
  nu = 0.5

  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT2_TEST02' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  BURGERS_VISCOUS_TIME_EXACT2 computes solution #2' )
  print ( '  to the Burgers equation.' )
  print ( '' )
  print ( '  Viscosity NU = %g' % ( nu ) )
  print ( '  NX = %d' % ( vxn ) )
  print ( '  NT = %d' % ( vtn ) )

  xlo = 0.0
  xhi = 2.0 * np.pi
  vx = np.linspace ( xlo, xhi, vxn )
  r8vec_print ( vxn, vx, '  X grid points:' )

  tlo = 0.0
  thi = 1.0
  vt = np.linspace ( tlo, thi, vtn )
  r8vec_print ( vtn, vt, '  T grid points:' )

  vu = burgers_viscous_time_exact2 ( nu, vxn, vx, vtn, vt )

  filename = 'burgers_solution_test04.txt'

  r8mat_write ( filename, vxn, vtn, vu )

  print ( '' )
  print ( '  Data written to file "%s"' % ( filename ) )
#
#  Terminate
#
  print ( '' )
  print ( 'BURGERS_VISCOUS_TIME_EXACT2_TEST02' )
  print ( '  Normal end of execution.' )
  return

def hermite_ek_compute ( n ):

#*****************************************************************************80
#
## HERMITE_EK_COMPUTE computes a Gauss-Hermite quadrature rule.
#
#  Discussion:
#
#    The code uses an algorithm by Elhay and Kautsky.
#
#    The abscissas are the zeros of the N-th order Hermite polynomial.
#
#    The integral:
#
#      integral ( -oo < x < +oo ) exp ( - x * x ) * f(x) dx
#
#    The quadrature rule:
#
#      sum ( 1 <= i <= n ) w(i) * f ( x(i) )
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    15 June 2015
#
#  Author:
#
#    John Burkardt.
#
#  Reference:
#
#    Sylvan Elhay, Jaroslav Kautsky,
#    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of
#    Interpolatory Quadrature,
#    ACM Transactions on Mathematical Software,
#    Volume 13, Number 4, December 1987, pages 399-415.
#
#  Parameters:
#
#    Input, integer N, the number of abscissas.
#
#    Output, real X(N), the abscissas.
#
#    Output, real W(N), the weights.
#
  import numpy as np
#
#  Define the zero-th moment.
#
  zemu = r8_gamma ( 0.5 )
#
#  Define the Jacobi matrix.
#
  bj = np.zeros ( n )
  for i in range ( 0, n ):
    bj[i] = np.sqrt ( float ( i + 1 ) / 2.0 )

  x = np.zeros ( n )

  w = np.zeros ( n )
  w[0] = np.sqrt ( zemu )
#
#  Diagonalize the Jacobi matrix.
#
  x, w = imtqlx ( n, x, bj, w )
#
#  If N is odd, force the center X to be exactly 0.
#
  if ( ( n % 2 ) == 1 ):
    x[(n-1)//2] = 0.0

  for i in range ( 0, n ):
    w[i] = w[i] ** 2

  return x, w

def hermite_ek_compute_test ( ):

#*****************************************************************************80
#
## HERMITE_EK_COMPUTE_TEST tests HERMITE_EK_COMPUTE.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    15 June 2015
#
#  Author:
#
#    John Burkardt
#
  import platform

  print ( '' )
  print ( 'HERMITE_EK_COMPUTE_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  HERMITE_EK_COMPUTE computes a Hermite quadrature rule' )
  print ( '  using the Elhay-Kautsky algorithm.' )
  print ( '' )
  print ( '  Index       X             W' )

  for n in range ( 1, 11 ):

    x, w = hermite_ek_compute ( n )

    print ( '' )

    for i in range ( 0, n ):
      print ( '  %2d  %24.16g  %24.16g' % ( i, x[i], w[i] ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'HERMITE_EK_COMPUTE_TEST:' )
  print ( '  Normal end of execution.' )

  return

def imtqlx ( n, d, e, z ):

#*****************************************************************************80
#
## IMTQLX diagonalizes a symmetric tridiagonal matrix.
#
#  Discussion:
#
#    This routine is a slightly modified version of the EISPACK routine to
#    perform the implicit QL algorithm on a symmetric tridiagonal matrix.
#
#    The authors thank the authors of EISPACK for permission to use this
#    routine.
#
#    It has been modified to produce the product Q' * Z, where Z is an input
#    vector and Q is the orthogonal matrix diagonalizing the input matrix.
#    The changes consist (essentially) of applying the orthogonal 
#    transformations directly to Z as they are generated.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    15 June 2015
#
#  Author:
#
#    John Burkardt.
#
#  Reference:
#
#    Sylvan Elhay, Jaroslav Kautsky,
#    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of
#    Interpolatory Quadrature,
#    ACM Transactions on Mathematical Software,
#    Volume 13, Number 4, December 1987, pages 399-415.
#
#    Roger Martin, James Wilkinson,
#    The Implicit QL Algorithm,
#    Numerische Mathematik,
#    Volume 12, Number 5, December 1968, pages 377-383.
#
#  Parameters:
#
#    Input, integer N, the order of the matrix.
#
#    Input, real D(N), the diagonal entries of the matrix.
#
#    Input, real E(N), the subdiagonal entries of the
#    matrix, in entries E(1) through E(N-1). 
#
#    Input, real Z(N), a vector to be operated on.
#
#    Output, real LAM(N), the diagonal entries of the diagonalized matrix.
#
#    Output, real QTZ(N), the value of Q' * Z, where Q is the matrix that 
#    diagonalizes the input symmetric tridiagonal matrix.
#
  import numpy as np

  lam = np.zeros ( n )
  for i in range ( 0, n ):
    lam[i] = d[i]

  qtz = np.zeros ( n )
  for i in range ( 0, n ):
    qtz[i] = z[i]

  if ( n == 1 ):
    return lam, qtz

  itn = 30

  prec = r8_epsilon ( )

  e[n-1] = 0.0

  for l in range ( 1, n + 1 ):

    j = 0

    while ( True ):

      for m in range ( l, n + 1 ):

        if ( m == n ):
          break

        if ( abs ( e[m-1] ) <= prec * ( abs ( lam[m-1] ) + abs ( lam[m] ) ) ):
          break

      p = lam[l-1]

      if ( m == l ):
        break

      if ( itn <= j ):
        print ( '' )
        print ( 'IMTQLX - Fatal error!' )
        print ( '  Iteration limit exceeded.' )
        exit ( 'IMTQLX - Fatal error!' )

      j = j + 1
      g = ( lam[l] - p ) / ( 2.0 * e[l-1] )
      r = np.sqrt ( g * g + 1.0 )

      if ( g < 0.0 ):
        t = g - r
      else:
        t = g + r

      g = lam[m-1] - p + e[l-1] / ( g + t )
 
      s = 1.0
      c = 1.0
      p = 0.0
      mml = m - l

      for ii in range ( 1, mml + 1 ):

        i = m - ii
        f = s * e[i-1]
        b = c * e[i-1]

        if ( abs ( g ) <= abs ( f ) ):
          c = g / f
          r = np.sqrt ( c * c + 1.0 )
          e[i] = f * r
          s = 1.0 / r
          c = c * s
        else:
          s = f / g
          r = np.sqrt ( s * s + 1.0 )
          e[i] = g * r
          c = 1.0 / r
          s = s * c

        g = lam[i] - p
        r = ( lam[i-1] - g ) * s + 2.0 * c * b
        p = s * r
        lam[i] = g + p
        g = c * r - b
        f = qtz[i]
        qtz[i]   = s * qtz[i-1] + c * f
        qtz[i-1] = c * qtz[i-1] - s * f

      lam[l-1] = lam[l-1] - p
      e[l-1] = g
      e[m-1] = 0.0

  for ii in range ( 2, n + 1 ):

     i = ii - 1
     k = i
     p = lam[i-1]

     for j in range ( ii, n + 1 ):

       if ( lam[j-1] < p ):
         k = j
         p = lam[j-1]

     if ( k != i ):

       lam[k-1] = lam[i-1]
       lam[i-1] = p

       p        = qtz[i-1]
       qtz[i-1] = qtz[k-1]
       qtz[k-1] = p

  return lam, qtz

def imtqlx_test ( ):

#*****************************************************************************80
#
## IMTQLX_TEST tests IMTQLX.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    15 June 2015
#
#  Author:
#
#    John Burkardt.
#
  import numpy as np
  import platform

  print ( '' )
  print ( 'IMTQLX_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  IMTQLX takes a symmetric tridiagonal matrix A' )
  print ( '  and computes its eigenvalues LAM.' )
  print ( '  It also accepts a vector Z and computes Q\'*Z,' )
  print ( '  where Q is the matrix that diagonalizes A.' )

  n = 5
  d = np.zeros ( n )
  for i in range ( 0, n ):
    d[i] = 2.0;
  e = np.zeros ( n )
  for i in range ( 0, n - 1 ):
    e[i] = -1.0
  e[n-1] = 0.0
  z = np.ones ( n )

  lam, qtz = imtqlx ( n, d, e, z )

  r8vec_print ( n, lam, '  Computed eigenvalues:' )

  lam2 = np.zeros ( n )
  for i in range ( 0, n ):
    angle = float ( i + 1 ) * np.pi / float ( 2 * ( n + 1 ) )
    lam2[i] = 4.0 * ( np.sin ( angle ) ) ** 2

  r8vec_print ( n, lam2, '  Exact eigenvalues:' )

  r8vec_print ( n, z, '  Vector Z:' )
  r8vec_print ( n, qtz, '  Vector Q''*Z:' )
#
#  Terminate.
#
  print ( '' )
  print ( 'IMTQLX_TEST:' )
  print ( '  Normal end of execution.' )
  return

def r8_epsilon ( ):

#*****************************************************************************80
#
## R8_EPSILON returns the R8 roundoff unit.
#
#  Discussion:
#
#    The roundoff unit is a number R which is a power of 2 with the 
#    property that, to the precision of the computer's arithmetic,
#      1 < 1 + R
#    but 
#      1 = ( 1 + R / 2 )
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    03 June 2013
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Output, real VALUE, the roundoff unit.
#
  value = 2.220446049250313E-016

  return value

def r8_epsilon_test ( ):

#*****************************************************************************80
#
## R8_EPSILON_TEST tests R8_EPSILON.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    01 September 2012
#
#  Author:
#
#    John Burkardt
#
  import platform

  print ( '' )
  print ( 'R8_EPSILON_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8_EPSILON produces the R8 roundoff unit.' )
  print ( '' )

  r = r8_epsilon ( )
  print ( '  R = R8_EPSILON()         = %e' % ( r ) )

  s = ( 1.0 + r ) - 1.0
  print ( '  ( 1 + R ) - 1            = %e' % ( s ) )

  s = ( 1.0 + ( r / 2.0 ) ) - 1.0
  print ( '  ( 1 + (R/2) ) - 1        = %e' % ( s ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8_EPSILON_TEST' )
  print ( '  Normal end of execution.' )
  return

def r8_gamma ( x ):

#*****************************************************************************80
#
## R8_GAMMA evaluates Gamma(X) for a real argument.
#
#  Discussion:
#
#    This routine calculates the gamma function for a real argument X.
#
#    Computation is based on an algorithm outlined in reference 1.
#    The program uses rational functions that approximate the gamma
#    function to at least 20 significant decimal digits.  Coefficients
#    for the approximation over the interval (1,2) are unpublished.
#    Those for the approximation for 12 <= X are from reference 2.
#
#    PYTHON provides a GAMMA function, which is likely to be faster, and more
#    accurate.  
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    24 July 2014
#
#  Author:
#
#    Original FORTRAN77 version by William Cody, Laura Stoltz.
#    PYTHON version by John Burkardt.
#
#  Reference:
#
#    William Cody,
#    An Overview of Software Development for Special Functions,
#    in Numerical Analysis Dundee, 1975,
#    edited by GA Watson,
#    Lecture Notes in Mathematics 506,
#    Springer, 1976.
#
#    John Hart, Ward Cheney, Charles Lawson, Hans Maehly,
#    Charles Mesztenyi, John Rice, Henry Thatcher,
#    Christoph Witzgall,
#    Computer Approximations,
#    Wiley, 1968,
#    LC: QA297.C64.
#
#  Parameters:
#
#    Input, real X, the argument of the function.
#
#    Output, real VALUE, the value of the function.
#
  import numpy as np
  from math import exp
  from math import floor
  from math import log
  from math import sin
#
#  Coefficients for minimax approximation over (12, INF).
#
  c = np.array ( [
   -1.910444077728E-03, \
    8.4171387781295E-04, \
   -5.952379913043012E-04, \
    7.93650793500350248E-04, \
   -2.777777777777681622553E-03, \
    8.333333333333333331554247E-02, \
    5.7083835261E-03 ] )
#
#  Mathematical constants
#
  r8_pi = 3.141592653589793
  sqrtpi = 0.9189385332046727417803297
#
#  Machine dependent parameters
#
  xbig = 171.624
  xminin = 2.23E-308
  eps = 2.22E-16
  xinf = 1.79E+308
#
#  Numerator and denominator coefficients for rational minimax
#  approximation over (1,2).
#
  p = np.array ( [ \
   -1.71618513886549492533811E+00, \
    2.47656508055759199108314E+01, \
   -3.79804256470945635097577E+02, \
    6.29331155312818442661052E+02, \
    8.66966202790413211295064E+02, \
   -3.14512729688483675254357E+04, \
   -3.61444134186911729807069E+04, \
    6.64561438202405440627855E+04 ] )

  q = np.array ( [ \
   -3.08402300119738975254353E+01, \
    3.15350626979604161529144E+02, \
   -1.01515636749021914166146E+03, \
   -3.10777167157231109440444E+03, \
    2.25381184209801510330112E+04, \
    4.75584627752788110767815E+03, \
   -1.34659959864969306392456E+05, \
   -1.15132259675553483497211E+05 ] )

  parity = 0
  fact = 1.0
  n = 0
  y = x
#
#  Argument is negative.
#
  if ( y <= 0.0 ):

    y = - x
    y1 = floor ( y )
    res = y - y1

    if ( res != 0.0 ):

      if ( y1 != floor ( y1 * 0.5 ) * 2.0 ):
        parity = 1

      fact = - r8_pi / sin ( r8_pi * res )
      y = y + 1.0

    else:

      res = xinf
      value = res
      return value
#
#  Argument is positive.
#
  if ( y < eps ):
#
#  Argument < EPS.
#
    if ( xminin <= y ):
      res = 1.0 / y
    else:
      res = xinf

    value = res
    return value

  elif ( y < 12.0 ):

    y1 = y
#
#  0.0 < argument < 1.0.
#
    if ( y < 1.0 ):

      z = y
      y = y + 1.0
#
#  1.0 < argument < 12.0.
#  Reduce argument if necessary.
#
    else:

      n = int ( floor ( y ) - 1 )
      y = y - n
      z = y - 1.0
#
#  Evaluate approximation for 1.0 < argument < 2.0.
#
    xnum = 0.0
    xden = 1.0
    for i in range ( 0, 8 ):
      xnum = ( xnum + p[i] ) * z
      xden = xden * z + q[i]

    res = xnum / xden + 1.0
#
#  Adjust result for case  0.0 < argument < 1.0.
#
    if ( y1 < y ):

      res = res / y1
#
#  Adjust result for case 2.0 < argument < 12.0.
#
    elif ( y < y1 ):

      for i in range ( 0, n ):
        res = res * y
        y = y + 1.0

  else:
#
#  Evaluate for 12.0 <= argument.
#
    if ( y <= xbig ):

      ysq = y * y
      sum = c[6]
      for i in range ( 0, 6 ):
        sum = sum / ysq + c[i]
      sum = sum / y - y + sqrtpi
      sum = sum + ( y - 0.5 ) * log ( y )
      res = exp ( sum )

    else:

      res = xinf
      value = res
      return value
#
#  Final adjustments and return.
#
  if ( parity ):
    res = - res

  if ( fact != 1.0 ):
    res = fact / res

  value = res

  return value

def r8_gamma_test ( ):

#*****************************************************************************80
#
## R8_GAMMA_TEST demonstrates the use of R8_GAMMA.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    25 July 2014
#
#  Author:
#
#    John Burkardt
#
  import platform

  print ( '' )
  print ( 'R8_GAMMA_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8_GAMMA evaluates the Gamma function.' )
  print ( '' )
  print ( '      X            GAMMA(X)      R8_GAMMA(X)' )
  print ( '' )

  n_data = 0

  while ( True ):

    n_data, x, fx1 = gamma_values ( n_data )

    if ( n_data == 0 ):
      break

    fx2 = r8_gamma ( x )

    print ( '  %12g  %24.16g  %24.16g' % ( x, fx1, fx2 ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8_GAMMA_TEST' )
  print ( '  Normal end of execution.' )
  return

def r8mat_print ( m, n, a, title ):

#*****************************************************************************80
#
## R8MAT_PRINT prints an R8MAT.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    31 August 2014
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer M, the number of rows in A.
#
#    Input, integer N, the number of columns in A.
#
#    Input, real A(M,N), the matrix.
#
#    Input, string TITLE, a title.
#
  r8mat_print_some ( m, n, a, 0, 0, m - 1, n - 1, title )

  return

def r8mat_print_test ( ):

#*****************************************************************************80
#
## R8MAT_PRINT_TEST tests R8MAT_PRINT.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    10 February 2015
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  print ( '' )
  print ( 'R8MAT_PRINT_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8MAT_PRINT prints an R8MAT.' )

  m = 4
  n = 6
  v = np.array ( [ \
    [ 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 ], 
    [ 21.0, 22.0, 23.0, 24.0, 25.0, 26.0 ], 
    [ 31.0, 32.0, 33.0, 34.0, 35.0, 36.0 ], 
    [ 41.0, 42.0, 43.0, 44.0, 45.0, 46.0 ] ], dtype = np.float64 )
  r8mat_print ( m, n, v, '  Here is an R8MAT:' )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8MAT_PRINT_TEST:' )
  print ( '  Normal end of execution.' )
  return

def r8mat_print_some ( m, n, a, ilo, jlo, ihi, jhi, title ):

#*****************************************************************************80
#
## R8MAT_PRINT_SOME prints out a portion of an R8MAT.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    10 February 2015
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer M, N, the number of rows and columns of the matrix.
#
#    Input, real A(M,N), an M by N matrix to be printed.
#
#    Input, integer ILO, JLO, the first row and column to print.
#
#    Input, integer IHI, JHI, the last row and column to print.
#
#    Input, string TITLE, a title.
#
  incx = 5

  print ( '' )
  print ( title )

  if ( m <= 0 or n <= 0 ):
    print ( '' )
    print ( '  (None)' )
    return

  for j2lo in range ( max ( jlo, 0 ), min ( jhi + 1, n ), incx ):

    j2hi = j2lo + incx - 1
    j2hi = min ( j2hi, n )
    j2hi = min ( j2hi, jhi )
    
    print ( '' )
    print ( '  Col: ', end = '' )

    for j in range ( j2lo, j2hi + 1 ):
      print ( '%7d       ' % ( j ), end = '' )

    print ( '' )
    print ( '  Row' )

    i2lo = max ( ilo, 0 )
    i2hi = min ( ihi, m )

    for i in range ( i2lo, i2hi + 1 ):

      print ( '%7d :' % ( i ), end = '' )
      
      for j in range ( j2lo, j2hi + 1 ):
        print ( '%12g  ' % ( a[i,j] ), end = '' )

      print ( '' )

  return

def r8mat_print_some_test ( ):

#*****************************************************************************80
#
## R8MAT_PRINT_SOME_TEST tests R8MAT_PRINT_SOME.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    31 October 2014
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  print ( '' )
  print ( 'R8MAT_PRINT_SOME_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8MAT_PRINT_SOME prints some of an R8MAT.' )

  m = 4
  n = 6
  v = np.array ( [ \
    [ 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 ], 
    [ 21.0, 22.0, 23.0, 24.0, 25.0, 26.0 ], 
    [ 31.0, 32.0, 33.0, 34.0, 35.0, 36.0 ], 
    [ 41.0, 42.0, 43.0, 44.0, 45.0, 46.0 ] ], dtype = np.float64 )
  r8mat_print_some ( m, n, v, 0, 3, 2, 5, '  Here is an R8MAT:' )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8MAT_PRINT_SOME_TEST:' )
  print ( '  Normal end of execution.' )
  return

def r8mat_write ( filename, m, n, a ):

#*****************************************************************************80
#
## R8MAT_WRITE writes an R8MAT to a file.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    12 October 2014
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, string FILENAME, the name of the output file.
#
#    Input, integer M, the number of rows in A.
#
#    Input, integer N, the number of columns in A.
#
#    Input, real A(M,N), the matrix.
#
  output = open ( filename, 'w' )

  for i in range ( 0, m ):
    for j in range ( 0, n ):
      s = '  %g' % ( a[i,j] )
      output.write ( s )
    output.write ( '\n' )

  output.close ( )

  return

def r8mat_write_test ( ):

#*****************************************************************************80
#
## R8MAT_WRITE_TEST tests R8MAT_WRITE.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    12 October 2014
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  print ( '' )
  print ( 'R8MAT_WRITE_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  Test R8MAT_WRITE, which writes an R8MAT to a file.' )

  filename = 'r8mat_write_test.txt'
  m = 5
  n = 3
  a = np.array ( (  \
    ( 1.1, 1.2, 1.3 ), \
    ( 2.1, 2.2, 2.3 ), \
    ( 3.1, 3.2, 3.3 ), \
    ( 4.1, 4.2, 4.3 ), \
    ( 5.1, 5.2, 5.3 ) ) )
  r8mat_write ( filename, m, n, a )

  print ( '' )
  print ( '  Created file "%s".' % ( filename ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8MAT_WRITE_TEST:' )
  print ( '  Normal end of execution.' )
  return
  
def r8_sign ( x ):

#*****************************************************************************80
#
## R8_SIGN returns the sign of an R8.
#
#  Discussion:
#
#    The value is +1 if the number is positive or zero, and it is -1 otherwise.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    03 June 2013
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, real X, the number whose sign is desired.
#
#    Output, real VALUE, the sign of X.
#
  if ( x < 0.0 ):
    value = -1.0
  else:
    value = +1.0
 
  return value

def r8_sign_test ( ):

#*****************************************************************************80
#
## R8_SIGN_TEST tests R8_SIGN.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    28 September 2014
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  test_num = 5

  r8_test = np.array ( [ -1.25, -0.25, 0.0, +0.5, +9.0 ] )

  print ( '' )
  print ( 'R8_SIGN_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8_SIGN returns the sign of an R8.' )
  print ( '' )
  print ( '     R8     R8_SIGN(R8)' )
  print ( '' )

  for test in range ( 0, test_num ):
    r8 = r8_test[test]
    s = r8_sign ( r8 )
    print ( '  %8.4f  %8.0f' % ( r8, s ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8_SIGN_TEST' )
  print ( '  Normal end of execution.' )
  return

def r8vec_print ( n, a, title ):

#*****************************************************************************80
#
## R8VEC_PRINT prints an R8VEC.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    31 August 2014
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer N, the dimension of the vector.
#
#    Input, real A(N), the vector to be printed.
#
#    Input, string TITLE, a title.
#
  print ( '' )
  print ( title )
  print ( '' )
  for i in range ( 0, n ):
    print ( '%6d:  %12g' % ( i, a[i] ) )

def r8vec_print_test ( ):

#*****************************************************************************80
#
## R8VEC_PRINT_TEST tests R8VEC_PRINT.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    29 October 2014
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform

  print ( '' )
  print ( 'R8VEC_PRINT_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8VEC_PRINT prints an R8VEC.' )

  n = 4
  v = np.array ( [ 123.456, 0.000005, -1.0E+06, 3.14159265 ], dtype = np.float64 )
  r8vec_print ( n, v, '  Here is an R8VEC:' )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8VEC_PRINT_TEST:' )
  print ( '  Normal end of execution.' )
  return

def timestamp ( ):

#*****************************************************************************80
#
## TIMESTAMP prints the date as a timestamp.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 April 2013
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    None
#
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return None

def timestamp_test ( ):

#*****************************************************************************80
#
## TIMESTAMP_TEST tests TIMESTAMP.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    03 December 2014
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    None
#
  import platform

  print ( '' )
  print ( 'TIMESTAMP_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  TIMESTAMP prints a timestamp of the current date and time.' )
  print ( '' )

  timestamp ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'TIMESTAMP_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  timestamp ( )
  burgers_solution_test ( )
  timestamp ( )

