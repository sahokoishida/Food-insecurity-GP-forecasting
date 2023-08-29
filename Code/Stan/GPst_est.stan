functions {
  #include GP_helper.stan
}
data {
  int<lower=1> N1 ;
  int<lower=1> N2 ;
  array[N1] vector[2] X1;
  array[N2] real X2;
  vector[N1*N2] y ;
}
transformed data {
  int N = N1*N2 ;
  matrix[N2, N1] Y = to_matrix(y, N2, N1);
}
parameters {
  real<lower=0> alpha0 ;
  real<lower=0> alpha1 ;
  real<lower=0> alpha2 ;
  real<lower=0.0001> rho1 ;
  real<lower=0.0001> rho2 ;
  real<lower=0> sigma ;
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
      matrix[N1,N1] K1 = rep_matrix(1, N1,N1) + gp_exp_quad_cov(X1, alpha1, rho1);
      //K1 = Gram_centring(K1, N1);
      {
        l1 = eigenvalues_sym(K1);
        Q1 = eigenvectors_sym(K1);
      }
      matrix[N2, N2] K2 = rep_matrix(1, N2, N2) + gp_exp_quad_cov(X2, alpha2, rho2);
      //K2 = Gram_centring(K2, N2);
      {
        l2 = eigenvalues_sym(K2);
        Q2 = eigenvectors_sym(K2);
      }
    }
    m = to_vector(Q2' * (Y * Q1));
  // computing eigenvalues of the model matrix K + sigma^2*I
    eval = square(alpha0)*to_vector(l2 * l1') + square(sigma)*rep_vector(1,N);
   }
  //prior
  //target += std_normal_lpdf(alpha0);
  //target += std_normal_lpdf(alpha1);
  //target += std_normal_lpdf(alpha2);
  //target += std_normal_lpdf(sigma);
  target += lognormal_lpdf(alpha0|0,1);
  target += lognormal_lpdf(alpha1|0,1);
  target += lognormal_lpdf(alpha2|0,1);
  target += lognormal_lpdf(sigma|0,1);
  target += inv_gamma_lpdf(rho1|5,5);
  target += inv_gamma_lpdf(rho2|5,5);
  //likelihood
  target += -0.5 * sum(square(m)./eval) - 0.5 * sum(log(eval));
}
