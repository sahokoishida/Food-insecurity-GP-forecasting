functions {
  #include GP_helper.stan
}
data {
  int<lower=1> N1 ;
  int<lower=1> N2 ;
  int<lower=1> n1 ;
  int<lower=2> n2 ;
  matrix[N1,2] X1;
  matrix[N2,1] X2;
  vector[N1*N2] y ;
  matrix[n1, 2] x1_new ;
  matrix[n2, 1] x2_new ;

  // parameteers
  real<lower=0> alpha0 ;
  real<lower=0> alpha1 ;
  real<lower=0> alpha2 ;
  real<lower=0> rho1 ;
  real<lower=0> rho2 ;
  real<lower = 0> sigma ;
}
transformed data {
  int N = N1*N2 ;
  int n = n1*n2;
  real mloglik;
  matrix[n, n] L_new;
  vector[n] mu_new;
  {
    vector[N] m ;
    vector[N] eval ;
    vector[N] q ;
    matrix[N1, N1] Q1 ;
    matrix[N2, N2] Q2 ;
    {
      vector[N1] l1;
      vector[N2] l2;
      matrix[N2, N1] Y = to_matrix(y, N2, N1);
      {
        matrix[N1,N1] K1 = rep_matrix(1, N1,N1) + square(alpha1)*Gram_SE(X1,N1, rho1);
        {
          l1 = eigenvalues_sym(K1);
          Q1 = eigenvectors_sym(K1);
        }
        matrix[N2, N2] K2 = rep_matrix(1, N2, N2) + square(alpha2)*Gram_SE(X2,N2, rho2);
        {
          l2 = eigenvalues_sym(K2);
          Q2 = eigenvectors_sym(K2);
        }
      }
      // computing m = (Q1' \otimes Q2')y
      m = to_vector(Q2' * (Y * Q1));
    // computing eigenvalues of the model matrix K + sigma^2*I
      eval = square(alpha0)*to_vector(l2 * l1') + square(sigma)*rep_vector(1,N);
      q = m ./ eval ;
     }
     mloglik = -0.5 * sum(square(m)./eval) - 0.5 * sum(log(eval)) - 0.5 *log(2*pi());

    /// posterior predective mean and variance
    {
      matrix[n, n] K_new;
      matrix[N1, n1] B1;
      matrix[N2, n2] B2;
      matrix[n1, n1] K1_new;
      matrix[n2, n2] K2_new;
      matrix[N1, n1] K1_trnew;
      matrix[N2, n2] K2_trnew;
      matrix[N, n] B ;
      K1_new = rep_matrix(1, n1, n1) + square(alpha1)*Gram_SE(x1_new,n1,rho1);
      K2_new = rep_matrix(1, n2, n2) + square(alpha2)*Gram_SE(x2_new,n2,rho2);
      for (i in 1:n1){
        K1_trnew[,i] = rep_vector(1,N1) + square(alpha1)*kvec_SE(X1, to_vector(x1_new[i,]), N1, rho1);
      }
      for (i in 1:n2){
        K2_trnew[,i] =  rep_vector(1,N2) + square(alpha2)*kvec_SE(X2, to_vector(x2_new[i,]), N2, rho2);
      }
      B1 = Q1'*K1_trnew;
      B2 = Q2'*K2_trnew;
      // mean
      mu_new = square(alpha0)*to_vector(B2' * (to_matrix(q, N2, N1)*B1));
      // variance
      B = square(alpha0)*kronecker_prod(B1, B2);
      K_new = square(alpha0)*kronecker_prod(K1_new,K2_new) - B'*diag_pre_multiply((rep_vector(1,N)./eval),B);
      L_new = cholesky_decompose(K_new + diag_matrix(rep_vector(1e-9,n)));
    }
  }
}
model{
}
generated quantities {
  vector[n] y_new;
  real mllik = mloglik;
  y_new = multi_normal_cholesky_rng(mu_new, L_new);
}
