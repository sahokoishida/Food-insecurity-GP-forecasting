functions {
  #include GP_helper.stan
}
data {
  int<lower=1> N1 ;
  int<lower=1> N2 ;
  matrix[N1, 2] = S ;
  matrix[N2, 1] = T ;
  vector[N1*N2] = y ;
}
transformed data {
  int N = N1*N2 ;
  matrix[N2, N1] Y = matrix(y, N2, N1);
}
parameters {
  real<lower=0> alpha0 ;
  real<lower=0> alpha1 ;
  real<lower=0> alpha2 ;
  real<lower=0> rho1 ;
  real<lower=0> rho2 ;
  real<lower = 0> sigma ;
}
model{
  vector[N] m;
  vector[N] eval ;
  // computing m = (Q1' \otimes Q2')* y
  {
    matrix[N1, N1] Q1 ;
    matrix[N2, N2] Q2 ;
    vector[N1] l1;
    vector[N2] l2;
    {
      matrix[N1,N1] K1 = Gram_SE(S, N1, rho1);
      K1 = Gram_centring(K1, N1);
      {
        matrix[N1, N1+1] R = cen_eigen_decompose(K1, N1);
        Q1 = R[1:N1, 1:N1] ;
        l1 = to_vector(R[1:N1, N1+1]);
      }
    }
    {
      matrix[N2, N2] K2 = Gram_SE(T, N2, rho2);
      K1 = Gram_centring(K2, N2);
      {
        matrix[N2, N2+1] R = cen_eigen_decompose(K2, N2);
        Q2 = R[1:N2, 1:N2] ;
        l2 = to_vector(R[1:N2, N2+1]);
      }
    }
    m = to_vector(Q2' * (Y * Q1));
  }
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
      vector[N] t1 = to_vector(e2 * e1');
      eval = square(alpha0)*(t0 + t1 + t2 + t12) + square(sigma)*rep_vector(1,N);
     }
  }
  //prior
  target += std_normal_lpdf(alpha0);
  target += std_normal_lpdf(alpha1);
  target += std_normal_lpdf(alpha2);
  target += std_normal_lpdf(sigma);
  target += inv_gamma_lpdf(rho1|5,5);
  target += inv_gamma_lpdf(rho2|5,5);
  //likelihood
  target += -0.5 * sum(square(m)./eval) - 0.5 * sum(log(eval));
}
