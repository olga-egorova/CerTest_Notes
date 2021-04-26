//
//   Bayesian calibration using GPR modelling
//
//   https://github.com/adChong/bc-stan
//    
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> n;                 // number of field data
  int<lower=1> m;                 // number of computer simulation
  int<lower=1> n_pred;            // number of predictions
  int<lower=1> p;                 // number of observable inputs x
  int<lower=1> q;                 // number of calibration parameters t
  vector[n] y;                    // field observations
  vector[m] eta;                  // output of computer simulations
  matrix[n, p] xf;                // observable inputs corresponding to y
  matrix[m, p] xc;                // (xc, tc): design points corresponding to eta
  matrix[m, q] tc; 
}

transformed data {
  int<lower=1> N;
  vector[n+m] z;                       // z = [y, eta]
  //real mu;
  
  //mu = 0;
  N = n + m;                           // number of all "data": field and simulation
  z = append_row(y, eta); 
}

parameters {
  real mu;                                         // mu: constant mean of the eta GP
  row_vector[q] tf;               // tf: calibration parameters
  row_vector<lower=0,upper=1>[2] rho_eta;        // rho_eta: reparameterisation of cl_eta  (simulator)
  real rho_delta;                               // rho_delta: reparameterisation of cl_delta (discrepancy term)
  real<lower=0> lambda_eta;                        // precision parameter for eta (=1/variance)
  real<lower=0> lambda_delta;                      // precision parameter for delta
  real<lower=0> tau2_delta;                         // scale parameter for delta kernel
  real<lower=0> tau2_eta;                           // scale parameter for eta kernel
  real<lower=0> lambda_e;                          // observation error precision
}

transformed parameters {
  // cl2_delta_inv: inverse squared correlation parameter for bias term
  // cl2_eta_inv: inverse squared correlation parameter for eta
  row_vector[2] cl2_eta_inv;
  real cl2_delta_inv;
  cl2_eta_inv = -4.0 * log(rho_eta);
  cl2_delta_inv = -4.0 * log(rho_delta);
}

model {
  // declare variables
  matrix[N, (p+q)] xt;
  matrix[N, N] sigma_eta;   // simulator covariance 
  matrix[n, n] sigma_delta; // bias term covariance 
  matrix[N, N] sigma_z;     // covariance matrix
  matrix[N, N] L;           // cholesky decomposition of covariance matrix 
  row_vector[p] temp_delta;
  row_vector[p+q] temp_eta;

  // xt = [[xf,tf],[xc,tc]]  
  xt[1:n, 1:p] = xf;         // (1,1)-block: field observed inputs n*p
  xt[(n+1):N, 1:p] = xc;     // (2,1)-block: observed simulation inputs m*p
  xt[1:n, (p+1):(p+q)] = rep_matrix(tf, n);   //(1,2)-block: field 'suggested'/'current' theta-s n*q
  xt[(n+1):N, (p+1):(p+q)] = tc;              //(2,2)-block: simulation theta-s m*q

  // diagonal elements of sigma_eta
  sigma_eta = diag_matrix(rep_vector((1 / lambda_eta + 0.1^10), N) + 
                          tau2_eta*(xt[,1] .* xt[,1])/lambda_eta);  // adding a nugget tau2
 
  // off-diagonal elements of sigma_eta
  for (i in 1:(N-1)) {
    for (j in (i+1):N) {
      temp_eta = xt[i] - xt[j];
      sigma_eta[i, j] = cl2_eta_inv[1] * (temp_eta[1:p] * temp_eta'[1:p]) +
                        cl2_eta_inv[2] * (temp_eta[(p+1):(p+q)] * temp_eta'[(p+1):(p+q)]); 
      sigma_eta[i, j] = exp(-0.5 * sigma_eta[i, j]) / lambda_eta;
      sigma_eta[j, i] = sigma_eta[i, j];
    }
  }

  // diagonal elements of sigma_delta
  sigma_delta = diag_matrix(rep_vector((1 / lambda_delta), n) + 
                            tau2_delta*(xf[,1] .* xf[,1])/lambda_delta);

  // off-diagonal elements of sigma_delta
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      temp_delta = xf[i] - xf[j];
      sigma_delta[i, j] = cl2_delta_inv * (temp_delta * temp_delta');
      sigma_delta[i, j] = exp(-0.5 * sigma_delta[i, j]) / lambda_delta;
      sigma_delta[j, i] = sigma_delta[i, j];
    }   
  }

  // computation of covariance matrix sigma_z 
  sigma_z = sigma_eta;
  sigma_z[1:n, 1:n] = sigma_eta[1:n, 1:n] + sigma_delta;

  // add observation errors
  for (i in 1:n) {
    sigma_z[i, i] = sigma_z[i, i] + (1.0 / lambda_e);
  }  

  // Specify priors here
  rho_eta[1:2] ~ beta(1.0, 0.3);
  rho_delta ~ beta(1.0, 0.3);
  tau2_eta ~ beta(2, 10);
  tau2_delta ~ beta(2, 10);
  lambda_eta ~ gamma(10, 10); // gamma (shape, rate)
  lambda_delta ~ gamma(10, 0.3); 
  lambda_e ~ gamma(10, 0.03); 
  mu ~ normal(0,1);
  tf[1] ~ normal(0.2, 0.15);
  tf[2] ~ normal(0.47, 0.15);
  tf[3] ~ normal(0.73, 0.15);
  L = cholesky_decompose(sigma_z); // cholesky decomposition
  z ~ multi_normal_cholesky(rep_vector(mu, N), L);
}
