int n_zero_eval(vector eval, int n){
  // count number of zero eigen values (smaller than threshold)
  // Input: eval - a vector of eigenvalues
  //        n ---- a size of a matrix
  // Output: number of zero eigen values
  real d = eval[1];
  int i = 1 ;
  while (d < 1e-9){
      i += 1 ;
      d = eval[i];
  }
  return i - 1 ;
}

vector eval_zero(vector eval, int k, int n){
  // replace eigenvalues with zero
  vector[n] evalz = eval;
  for (i in 1:k)  evalz[i] = 0.0;
  return evalz ;
}

matrix GS_complete(matrix Evec, int k, int n){
  // Gram-Schmidt process
  // Input: Evec - original sets of eigenvaectors
  //        k, n --the number of zero eigen values, a size of a matrix
  // Output: sets of eigenvaectors after Gram-Schumidt process
  matrix[n,k] Q;
  matrix[n,k] V ;
  matrix[n,k] X = Evec[,1:k];
  X[,1] =  rep_vector(1/sqrt(n),n) ;
  V = X ;
  Q[,1] =  V[,1];
  for (i in 2:k){
    for (j in 1:(i-1)){
      V[,i] = V[,i] - ((Q[,j]')*X[,i])*Q[,j];
    }
    Q[,i] = V[,i]/sqrt(sum(V[,i].*V[,i]));
  }
  return append_col(Q,Evec[,(k+1):n]);
}

matrix cen_eigen_decompose(matrix K, int N){
  // Output: Eigenvalues and eigenvector of a centered Gram matrix
  // Input: K - Centred Gram matrix, N - nrow = ncol of K
  matrix[N, N+1] R;
  {
    matrix[N,N] Q ;
    vector[N] l = eigenvalues_sym(K);
    {
      int k = n_zero_eval(l,N);
      if (k > 0){
        Q = eigenvectors_sym(K);
        l = eval_zero(l, k, N);
        if (k>1){
            Q = GS_complete(Q, k, N);
        } else {
          Q[,1] = rep_vector(1/sqrt(N),N);
        }
      } else {
        reject("k must be positive; found k=", k);
      }
    }
    R[1:N,1:N] = Q ;
    R[1:N, N+1] = l;
  }
  return R ;
}

matrix Gram_SE(matrix X, int N, real rho){
  // Output: n by n Gram matrix with squared exponential kernel
  // Input: X - predictor
  //        n - n_rows of X
  //        rho - length scale of s.e. kernel
  matrix[N,N] K = diag_matrix(rep_vector(1,N));
  real sq_rho = square(rho);
  for (i in 1:(N-1)){
    for (j in (i+1):N){
      real r2 = squared_distance(X[i,],X[j,]);
      K[i,j] = exp(-r2/(2*sq_rho));
      K[j,i] = K[i,j];
    }
  }
  return K ;
}

matrix Gram_fBM(matrix X, int N, real Hurst){
  // Output: Gram matrix with fractional brownian motion
  // Input: X - predictor
  //        N - nrow of X
  //        Hurst - Hurst coefficeint of fBM kernel
  matrix[N, N] K;
  matrix[N, N] E ;
  matrix[N, N] B = rep_matrix(0, N, N);;
  vector[N] d;
  matrix[N, N] A = diag_matrix(rep_vector(1,N)) - (1.0/N)*rep_matrix(1, N, N);
  matrix[N, N] Xcp = X * X' ;
  vector[N] dvec = diagonal(Xcp);
  for (i in 1:(N-1)){
    d[i] = pow(fabs(dvec[i]), Hurst);
    for (j in (i+1):N){
      B[i,j] = pow(fabs(dvec[i] + dvec[j] - 2 * Xcp[i,j]), Hurst);
      B[j,i] = B[i,j];
    }
  }
  d[N] = pow(fabs(dvec[N]), Hurst);
  E = rep_matrix(d, N);
  K = 0.5 * (E + E' - B);
  //K = A * K * A;
  return K ;
}

matrix Gram_fBM_sq_cen(matrix X, int N, real Hurst){
  // Output: Gram matrix with square centered fractional brownian motion
  // Input: X - predictor
  //        N - nrow of X
  //        Hurst - Hurst coefficeint of fBM kernel
  matrix[N, N] K = Gram_fBM(X, N, Hurst);
  matrix[N, N] A = diag_matrix(rep_vector(1,N)) - (1.0/N)*rep_matrix(1, N, N);
  K = A * K * A;
  return K * K ;
}

matrix Gram_centring(matrix K, int N){
    // Output: centered Gram matrix
    // Input: K - Uncentered Gram matrix
    //        N - nrow of K
    matrix[N, N] A = diag_matrix(rep_vector(1,N)) - (1.0/N)*rep_matrix(1, N, N);
    return A * K * A ;
}

matrix Gram_square(matrix K){
    // Output: squared Gram matrix
    // Input: K -  Gram matrix
    //        N - nrow of K
        return K * K ;
}

vector kvec_SE(matrix X, vector x_tes, int N, real rho){
  // Output: a vector of k(x*, x_1),...,k(x*, x_N) for a given test point x*
  //          with squared exponential kernel
  // Input: X - (traing) predictor
  //        x_tes - test point
  //        N - n_rows of X
  //        rho - length scale of s.e. kernel
  vector[N] kvec ;
  real sq_rho = square(rho);
  for (i in 1:N){
    real r2 = squared_distance(x_tes, X[i,]) ;
    kvec[i] = exp(-r2/(2*sq_rho));
  }
  return kvec ;
}


real kstar_SE(vector x_tes, real rho){
  real r2 = squared_distance(X[i,],X[j,]);
  return exp(-r2/(2*square(rho)));
}

vector kvec_SE_cen(matrix X, vector x_tes, matrix K, int N, real rho){
  // Output: the same as kvec_SE but centred version
  // Input Additional to kvec_SE, the uncentred Gram matrix is needed
  vector[N] k1 = kvec_SE(X, x_tes, N, rho);
  real      k2 = sum(k1);
  vector[N] k3 = K * rep_vector(1,N) ; //rowsum
  real      k4 = sum(k3);

  return (k1 - rep_vector(k2/N, N) - (1.0/N)*k3 + rep_vector(k4/square(N), N));
}
/*
vector kvec_cen(vector kvec, vector x_tes, matrix K, int N){

  vector[N] k1 = kvec;
  real      k2 = sum(k1);
  vector[N] k3 = K * rep_vector(1,N) ; //rowsum
  real      k4 = sum(k3);

  return (k1 - rep_vector(k2/N, N) - (1.0/N)*k3 + rep_vector(k4/square(N), N));
}
*/
vector kvec_cen(vector kvec, vector Krsum, real Ksum, int N){
  return kvec - rep(sum(k1)/N, N) -(1.0/N)*Krsum + rep_vector(Ksum/square(N), N);
}

real kstar_cen(real kstar,vector kvec, real Ksum,int N){
  return kstar - 2*sum(kvec)/N + Ksum/square(N) ;
}
