functions {
  #include GP_helper.stan
}
data {
  int<lower=1> N1 ;
  int<lower=1> N2 ;
  int<lower=1> n1 ;
  int<lower=2> n2 ;
  matrix[N1, 2] = S ;
  matrix[N2, 1] = T ;
  vector[N1*N2] = y ;
  matrix[n1, 2] = s ;
  matrix[n2, 1] = t ;

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
  real f_new0[2];
  vector[n1] f_new[2] ;
  vector[n2] f_new[2] ;
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
      real K1sum ;
      real K2sum ;
      vector[N1] K1rsum ;
      vector[N2] K2rsum ;
      {
        matrix[N1,N1] K = Gram_SE(S, N1, rho1);
        matrix[N1,N1] C = Gram_centring(K1, N1);
        matrix[N1, N1+1] R = cen_eigen_decompose(K1, N1);
        Q1 = R[1:N1, 1:N1] ;
        l1 = to_vector(R[1:N1, N1+1]);
      }
      {
        matrix[N2,N2] K = Gram_SE(S, N2, rho2);
        matrix[N2,N2] C = Gram_centring(K2, N2);
        matrix[N2, N2+1] R = cen_eigen_decompose(K2, N2);
        Q2 = R[1:N2, 1:N2] ;
        l2 = to_vector(R[1:N2, N2+1]);
      }
      m = to_vector(Q2' * (Y * Q1));
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
          m_div_eval = m ./ eval ;
         }
      }
      mloglik = -0.5 * sum(square(m)./eval) - 0.5 * sum(log(eval)) - 0.5 *log(2*pi());
    }
    /// posterior predective mean and variance
    vector[N1] Qv1 = Q1' * rep_vector(1,N1) ;
    vector[N2] Qv2 = Q2 * rep_vector(1, N2) ;
    {
      vector[N] z0 = square(alpha0) * to_vector(Qv2 * Qv1');
      f_new0[1] = z0' * m_div_eval ;
      f_new0[2] = square(alpha0) -sum(square(z0)./eval) ;
    }
    for (i in 1:n1){
      real v1;
      vector[N] Qk1 ;
      {
        vector[N] z1 ;
        vector[N] kvec = kvec_SE(S, s[i,], K1, N1, rho1);
        vector[N] k1 = square(alpha1)*kvec_cen(kvec, K1sum,N1);
        Qk1 = Q1' * k1 ;
        z1 = square(alpha0) * to_vector(Qv2 * Qk1');
        v1 = square(alpha1) * kstar_cen(kstar_SE(s[i,],rho1), kvec, K1sum, N1);
        f_new1[1,i] = z1' * m_div_eval ;
        f_new1[2,i] =  square(alpha0) * v1 - sum(square(z1)./eval);
      }
      for (j in 1:n2){
        real v2;
        real v12;
        vector[N] Qk2 ;
        {
          vector[N] z2 ;
          vector[N] z12 ;
          vector[N] kvec = kvec_SE(T, t[j,], K2, N2, rho2);
          vector[N] k2 = square(alpha2)*kvec_cen(kvec, K2sum,N2);
          Qk2 = Q2' * k2 ;
          z2  = square(alpha0) * to_vector(Qk2 * Qv1');
          z12 = square(alpha0) * to_vector(Qk2 * Qk1');
          v1  = square(alpha2) * kstar_cen(kstar_SE(t[i,],rho2), kvec, K2sum, N2);
          v12 = v1 * v2 ;
          f_new1[1,j] = z2' * m_div_eval ;
          f_new1[2,j] =  square(alpha0) * v2 - sum(square(z2)./eval);
          f_new12[1, (i-1)*n2 + j] = z12' * m_div_eval ;
          f_new12[2, (i-1)*n2 + j] =  square(alpha0) * v12 - sum(square(z12)./eval);
        }
      }
    }
  }
}
model{
}
generated quantities {
  // mean
  real mu0 = f_new0[1];
  vector[N1_new] mu1 = f_new1[1];
  vector[N2_new] mu2 = f_new2[1];
  vector[N1_new*N2_new] mu12 = f_new12[1] ;
  // variance
  real v0 = f_new0[2];
  vector[N1_new] v1 = f_new1[2];
  vector[N2_new] v2 = f_new2[2];
  vector[N1_new*N2_new] v12 = f_new12[2] ;
  // marginal likelihood
  real mllik = mloglik;
}
