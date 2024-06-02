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
  real<lower=0> rho1 ;
  real<lower=0,upper=1> Hurst2;
}
transformed data {
  int N = N1*N2 ;
  int n = n1*n2;
  matrix[N2, N1] Y = to_matrix(y, N2, N1);
  vector[N1] l1;
  matrix[N1, N1] Q1 ;
  matrix[N2, N2] K22;
  vector[N1] K1rowsum;
  vector[N2] K22rowsum;
  {
    matrix[N1, N1] K1 = Gram_exponential(X1,rho1);
    K1rowsum = K1 * rep_vector(1,N1);
    K1 = Gram_centring(K1);
    {
      matrix[N1, N1+1] R = cen_eigen_decompose(K1);
      Q1 = R[1:N1, 1:N1] ;
      l1 = to_vector(R[1:N1, N1+1]);
    }
    K22 = Gram_fBM(X2, Hurst2);
    K22rowsum = K22 * rep_vector(1,N2);
    K22 =  Gram_centring(K22);
  }
}
parameters {
  real<lower=0> alpha0 ;
  real<lower=0> alpha1 ;//scale parameter for Matern kernel
  real<lower=0> alpha21; //scale parameter for fBM kernel (time)
  real<lower=0> alpha22; //scale parameter for fBM kernel (time)
  real<lower=0, upper=N2> rho2;
  real<lower=0> sigma1 ; // random effect for space
  vector[N] f ;
  real<lower=0> sigma ;
}
model{
  vector[N] eval ;
  vector[N] m ;
  vector[N] p ;
  {
    //vector[N1] l1;
    vector[N2] l2;
    //matrix[N1, N1] Q1 ;
    matrix[N2, N2] Q2 ;
    {
      matrix[N2, N2] K2 ;
      {
        matrix[N2,N2] K21 = Gram_exponential(X2, rho2);
        K21 = Gram_centring(K21);
        K2 = square(alpha21)*K21 + square(alpha22)*K22;
      }

      {
        matrix[N2, N2+1] R = cen_eigen_decompose(K2);
        Q2 = R[1:N2, 1:N2] ;
        l2 = to_vector(R[1:N2, N2+1]);
      }
    }
    m = to_vector(Q2' * (to_matrix(f,N2,N1) * Q1));
    p = 1 /(1+exp(-f));
    {
      vector[N1] e1 = square(alpha1) * l1;
      //vector[N2] e2 = square(alpha2) * l2;
      vector[N2] e2 = l2;
      vector[N1] d1 = rep_vector(0,N1);
      vector[N2] d2 = rep_vector(0,N2);
      d1[1] = N1;
      d2[1] = N2;
      {
        vector[N] t0 = to_vector(d2 * d1');
        vector[N] t1 = to_vector(d2 * e1');
        vector[N] t2 =  to_vector(e2 * d1');
        vector[N] t12 = to_vector(e2 * e1');
        vector[N] tre1 = to_vector(d2 * rep_vector(1,N1)');
        eval = square(alpha0)*(t0 + t1 + t2 + t12) + square(sigma1)*tre1 +1e-9; //+ square(sigma)*rep_vector(1,N);
       }
     }
    }
  //prior
  target += lognormal_lpdf(alpha0|0,1);
  target += lognormal_lpdf(alpha1|0,1);
  target += lognormal_lpdf(alpha21|0,1);
  target += lognormal_lpdf(alpha22|0,1);
  target += inv_gamma_lpdf(rho2|2,5);
  target += lognormal_lpdf(sigma1|0,1);
  target += lognormal_lpdf(sigma|0,1);

  // likelihood
  //likelihood
  target += -0.5 * sum(square(m)./eval) - 0.5 * sum(log(eval));
  target += normal_lpdf(y|p,sigma);
}
generated quantities{
  vector[n] y_new;
   {
    vector[n] f_new;
    {
        vector[n] mu_new;
        matrix[n,n] L_new;
        {
        vector[N] q ;
        vector[N] m ;
        vector[N] eval;
        {
            vector[N2] l2;
            matrix[N2, N2] Q2 ;
            vector[N2] K21rowsum;
            {
                matrix[N2, N2] K2 ;
                {
                    matrix[N2,N2] K21 = Gram_exponential(X2, rho2);
                    K21rowsum = K22 * rep_vector(1,N2);
                    K21 = Gram_centring(K21);
                    K2 = square(alpha21)*K21 + square(alpha22)*K22;
                }
                {
                    matrix[N2, N2+1] R = cen_eigen_decompose(K2);
                    Q2 = R[1:N2, 1:N2] ;
                    l2 = to_vector(R[1:N2, N2+1]);
                }
            }
            m = to_vector(Q2' * (to_matrix(f,N2,N1) * Q1));
            {
                vector[N1] e1 = square(alpha1) * l1;
                //vector[N2] e2 = square(alpha2) * l2;
                vector[N2] e2 = l2;
                vector[N1] d1 = rep_vector(0,N1);
                vector[N2] d2 = rep_vector(0,N2);
                d1[1] = N1;
                d2[1] = N2;
                {
                    vector[N] t0 = to_vector(d2 * d1');
                    vector[N] t1 = to_vector(d2 * e1');
                    vector[N] t2 =  to_vector(e2 * d1');
                    vector[N] t12 = to_vector(e2 * e1');
                    vector[N] tre1 = to_vector(d2 * rep_vector(1,N1)');
                    eval = square(alpha0)*(t0 + t1 + t2 + t12) + square(sigma1)*tre1 +1e-9; //+ square(sigma)*rep_vector(1,N);
                }
                q = m./eval;
            }
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
                    vector[n2] k21tmp;
                    vector[n2] k22tmp;
                    matrix[N2, n2] K21_trnew;
                    matrix[N2, n2] K22_trnew;
                    matrix[n2, n2] K21_new;
                    matrix[n2, n2] K22_new;
                    for (i in 1:n1){
                    K1_trnew[,i] = kvec_exponential(X1, to_vector(x1_new[i,]), rho1);
                    k1tmp[i] = sum(K1_trnew[,i]);
                    for (j in 1:i){
                        K1_new[i,j] =  1 + square(alpha1)*(kernel_exponential(to_vector(x1_new[i,]),to_vector(x1_new[j,]),rho1) - k1tmp[i]/n1 - k1tmp[j]/n1 +sum(K1rowsum)/square(n1));
                        K1_new[j,i] = K1_new[i,j];
                        }
                    }
                    for (i in 1:n1){
                    K1_trnew[,i] = rep_vector(1,N1) + square(alpha1)*kvec_cen(K1_trnew[,i] ,K1rowsum);
                    for (j in 1:N1){
                        if (distance(x1_new[i,], X1[j,]) < 1e-6) {
                        RE_trnew[j,i] = 1 ;
                        } else {
                        RE_trnew[j,i] = 0 ;
                        }
                    }
                    }
                    for (i in 1:n2){
                    K21_trnew[,i] = kvec_exponential(X2, to_vector(x2_new[i,]), rho2);
                    K22_trnew[,i] = kvec_fBM(X2, to_vector(x2_new[i,]), Hurst2);
                    k21tmp[i] = sum(K21_trnew[,i]);
                    k22tmp[i] = sum(K22_trnew[,i]);
                    //print("k21tmp:", k21tmp[i],"k22tmp:", k22tmp[i]);
                    for (j in 1:i){
                        real k21star = kernel_exponential(to_vector(x2_new[i,]),to_vector(x2_new[j,]),rho2) - k21tmp[i]/n2 - k21tmp[j]/n2 +sum(K21rowsum)/square(n2);
                        real k22star = kernel_fBM(to_vector(x2_new[i,]),to_vector(x2_new[j,]),Hurst2) - k22tmp[i]/n2 - k22tmp[j]/n2 +sum(K22rowsum)/square(n2);
                        K2_new[i,j] =  1 + square(alpha21)* k21star + square(alpha22)* k22star;
                        K2_new[j,i] = K2_new[i,j];
                    }
                    }
                    for (i in 1:n2){
                        K2_trnew[,i] = rep_vector(1,N2) + square(alpha21)*kvec_cen(K21_trnew[,i] ,K21rowsum) + square(alpha22)*kvec_cen(K22_trnew[,i] ,K22rowsum);
                    }
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
     f_new = multi_normal_cholesky_rng(mu_new, L_new);
    }
    for (i in 1:n){
        y_new[i] = normal_rng(inv_logit(f_new[i]),sigma);
    }    
  }
}