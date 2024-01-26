functions {
  #include GP_helper.stan
}
data {
  int<lower=1> N1 ; //training
  int<lower=1> N2 ;
  int<lower=1> n1 ;//test
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
  real<lower=0> sigma1 ;
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
    vector[N1] K1rowsum;
    vector[N2] K2rowsum;
    real K1sum;
    real K2sum;
    {
      vector[N1] l1;
      vector[N2] l2;
      matrix[N2, N1] Y = to_matrix(y, N2, N1);
      {
        matrix[N1,N1] K1 = Gram_SE(X1,N1,rho1);
        K1rowsum =  K1 * rep_vector(1,N1);
        K1sum = sum(K1rowsum);
        K1 = Gram_centring(K1, N1);
        {
          matrix[N1, N1+1] R = cen_eigen_decompose(K1, N1);
          Q1 = R[1:N1, 1:N1] ;
          l1 = to_vector(R[1:N1, N1+1]);
        }
        matrix[N2, N2] K2 = Gram_SE(X2,N2,rho2);
        K2rowsum =  K2 * rep_vector(1,N2);
        K2sum = sum(K2rowsum);
        K2 = Gram_centring(K2, N2);
        {
          matrix[N2, N2+1] R = cen_eigen_decompose(K2, N2);
          Q2 = R[1:N2, 1:N2] ;
          l2 = to_vector(R[1:N2, N2+1]);
        }
      }
      // computing m = (Q1' \otimes Q2')y
      m = to_vector(Q2' * (Y * Q1));
    // computing eigenvalues of the model matrix K + sigma^2*I
      {
        vector[N1] e1 = square(alpha1) * l1;
        vector[N2] e2 = square(alpha2) * l2;
        vector[N1] d1 = rep_vector(0,N1);
        vector[N2] d2 = rep_vector(0,N2);
        d1[1] = N1;
        d2[1] = N2;
        {
          vector[N] t0 = to_vector(d2 * d1');
          vector[N] t1 = to_vector(d2 * e1');
          vector[N] t2 = to_vector(e2 * d1');
          vector[N] t12= to_vector(e2 * e1');
          vector[N] tre = to_vector(d2 * rep_vector(1,N1)');
          eval = square(alpha0)*(t0 + t1 + t2 + t12) + square(sigma1)*tre + square(sigma)*rep_vector(1,N);
          q = m ./ eval ;
         }
       }
     }
     mloglik = -0.5 * sum(square(m)./eval) - 0.5 * sum(log(eval)) - 0.5 *log(2*pi());

    /// posterior predective mean and variance
    {
      matrix[N1, n1] B1;
      matrix[N2, n2] B2;
      matrix[N1, n1] C1;
      matrix[N2, n2] C2;
      matrix[n1, n1] K1_new;
      matrix[n2, n2] K2_new;
      matrix[N1, n1] K1_trnew;
      matrix[N2, n2] K2_trnew;
      matrix[N1, n1] RE_trnew;
      matrix[N, n] B ;
      matrix[N, n] C ;

      {
        vector[n1] k1tmp;
        vector[n2] k2tmp;
        for (i in 1:n1){
          //K1_trnew[,i] = rep_vector(1,N1) + square(alpha1) * kvec_cen(kvec_SE(X1, to_vector(x1_new[i,]), N1, rho1), K1rowsum);
          K1_trnew[,i] = kvec_SE(X1, to_vector(x1_new[i,]), N1, rho1);
          k1tmp[i] = sum(K1_trnew[,i]);
          for (j in 1:i){
            K1_new[i,j] =  1 + square(alpha1)*(kernel_SE(to_vector(x1_new[i,]),to_vector(x1_new[j,]),rho1) - k1tmp[i]/n1 - k1tmp[j]/n1 +K1sum/square(n1));
            K1_new[j,i] = K1_new[i,j];
          }
        }
        for (i in 1:n1){
          K1_trnew[,i] = rep_vector(1,N1) + square(alpha1)*kvec_cen(K1_trnew[,i] ,K1rowsum);
          for (j in 1:N1){
            if (distance(x1_new[i,], X1[j,]) < 1e-6) {
              RE_trnew[i,j] = 1 ;
            } else {
            RE_trnew[i,j] = 0 ;
            }
          }
        }
        for (i in 1:n2){
          //K2_trnew[,i] =  rep_vector(1,N2) + square(alpha2) * kvec_cen(kvec_SE(X2, to_vector(x2_new[i,]), N2, rho2), K2rowsum);
          K2_trnew[,i] = kvec_SE(X2, to_vector(x2_new[i,]), N2, rho2);
          k2tmp[i] = sum(K2_trnew[,i]);
          for (j in 1:i){
            K2_new[i,j] = 1 + square(alpha2)*(kernel_SE(to_vector(x2_new[i,]),to_vector(x2_new[j,]),rho2) - k2tmp[i]/n2 - k2tmp[j]/n2 +K2sum/square(n2));
            K2_new[j,i] = K2_new[i,j];
          }
        }
        for (i in 1:n2){
          K2_trnew[,i] = rep_vector(1,N2) + square(alpha2) * kvec_cen(K2_trnew[,i] ,K2rowsum);
        }
        // centre K1_trnew and K2_trnew
      }
      B1 = Q1'*K1_trnew;
      C1 = Q1'*RE_trnew;
      B2 = Q2'*K2_trnew;
      C2 = Q2'*rep_matrix(1, N2, n2);
      // mean
      mu_new = square(alpha0)*to_vector(B2' * (to_matrix(q, N2, N1)*B1)) + square(sigma1)*to_vector(C2' * (to_matrix(q, N2, N1)*C1));
      // variance
      B = square(alpha0)*kronecker_prod(B1, B2);
      C = square(sigma1)*kronecker_prod(C1, C2);
      {
       matrix[n, n] K_new;
       K_new = square(alpha0)*kronecker_prod(K1_new,K2_new) + square(sigma1)*kronecker_prod(diag_matrix(rep_vector(1, n1)),rep_matrix(1,n2,n2))  - B'*diag_pre_multiply((rep_vector(1,N)./eval),B)- C'*diag_pre_multiply((rep_vector(1,N)./eval),C);/// add RE term HERE
       L_new = cholesky_decompose(K_new + diag_matrix(rep_vector(1e-9,n)));
      }
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
