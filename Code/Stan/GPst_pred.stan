functions {
  #include GP_helper.stan
}
data {
  int<lower=1> N1 ;
  int<lower=1> N2 ;
  int<lower=1> n1 ;
  int<lower=2> n2 ;
  array[N1] vector[2] X1;
  array[N2] real X2;
  vector[N1*N2] = y ;
  matrix[n1, 2] = x1_new ;
  matrix[n2, 1] = x2_new ;

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
  int n = n1 * n2;
  real mloglik;
  vector[n] f_new[2] ;

  {
    vector[N] m ;
    vector[N] eval ;
    vector[N] m_div_eval ;
    matrix[N1, N1] Q1 ;
    matrix[N2, N2] Q2 ;
    vector[N1] l1;
    vector[N2] l2;
    matrix[N2, N1] Y = matrix(y, N2, N1);
    {
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
      // computing m = (Q1' \otimes Q2')y
      m = to_vector(Q2' * (Y * Q1));
    // computing eigenvalues of the model matrix K + sigma^2*I
      eval = square(alpha0)*to_vector(l2 * l1') + square(sigma)*rep_vector(1,N);
      m_div_eval = m ./ eval ;
     }
     mloglik = -0.5 * sum(square(m)./eval) - 0.5 * sum(log(eval)) - 0.5 *log(2*pi());

    /// posterior predective mean and variance
    for (i in 1:n1){
      // z' = (k1 \otimes k2)' * (Q1\ otimes Q2) = ( k1' * Q1 )\otimes (k2' \otimes Q2)
      // z  = (Q1'  * k1 ) \otimes (Q2' * k2 )
      // mu_new = z'
      // var_new = k(x_new,x_new) - z'z./eval
      vector[N1] k1 = rep_vector(1,N1) + square(alpha1)*kvec_SE(X1, x1_new[i,], N1, rho1);
      real k1_star = 1 +  square(alpha1) * kernel_SE(to_vector(x1_new[i,]),to_vector(x1_new[i,]), square(rho1)) ;
      vector[N1] Qk1 = Q1' * k1 ;
      for (j in 1:n2){
        vector[N2] k2 = rep_vector(1,N2) + square(alpha2)*kvec_SE(X2, x2_new[j,], N2, rho2);
        real k2_star = 1 +  square(alpha2) * kernel_SE(to_vector(x2_new[j,]),to_vector(x2_new[j,]), square(rho2)) ;
        vector[N2] Qk2 = Q2' * k2 ;
        vector[N] z = square(alpha0) * to_vector(Qk2 * Qk1');
        real k_star = square(alpha0) * (k1_star * k2_star);
        f_new[1, (i-1)*n2 + j] = z' * m_div_eval ;
        f_new[2, (i-1)*n2 + j] =  k_star - sum(square(z)./eval);
      }
    }
  }
}
model{
}
generated quantities {
  // mean
  vector[n] mu = f_new[1] ;
  // variance
  vector[n] v = f_new[2] ;
  // marginal likelihood
  real mllik = mloglik;
  // if sampling from posterior
  // real f_bar[n] = normal_rng(mu, v);
}
