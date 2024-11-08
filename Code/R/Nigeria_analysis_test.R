library(cmdstanr)
library(ggplot2)
library(dplyr)
library(here)

base_path = '/Users/ishidasahoko/GitHub/Food-insecurity-GP-forecasting/'
setwd(base_path)
stan_path = paste0(base_path, 'Code/Stan')

prep_stan_data = function(df_long, df_loc, regions, y_transform = TRUE, main_var= 'fcs', features = NULL,center = FALSE){
  df = df_long[df_long$adm1_code%in%regions,]
  n1 = length(regions)
  y = df[,main_var]
  if (y_transform){
    y = log(y/(1-y))
  }
  n2 = length(y)/n1
  loc = df_loc[df_loc$adm1_code%in%regions,c('Easting','Northing')]/1000
  res = list(N1=n1, N2=n2, X1=loc, X2=matrix(c(1:n2),n2,1), y = y)
  
  if (!is.null(features)){
    covmat = as.matrix(df[,features])
    if (center){
      N = n1* n2
      C = diag(N) - matrix(1/N, N, N)
      covmat = C%*%covmat
    }
    res = list(N1=n1, N2=n2, P=length(features),
               X1=loc, X2=matrix(c(1:n2),n2,1), D = covmat,
               y = y)
  }
  return(res)
}

prep_stan_data_pred = function(df_long, df_loc, regions, main_var= 'fcs',
                               features = NULL, center = FALSE, tr_region=NULL){
  df = df_long[df_long$adm1_code%in%regions,]
  y = df[,main_var]
  n1 = length(regions)
  n2 = length(y)/n1
  loc = df_loc[df_loc$adm1_code%in%regions,c('Easting','Northing')]/1000
  res = list(list(n1=n1, n2=n2, x1_new=loc, x2_new=matrix(c(1:n2),n2,1)), y)
  
  if (!is.null(features)){
    covmat = as.matrix(df[,features])
    if (center){
      n = n1 * n2
      P=length(features)
      trcov_mean = as.matrix(df_long[df_long$adm1_code%in%tr_region,features])|>colMeans()
      covmat = as.matrix(df[,features])
      covmat = covmat - matrix(rep(trcov_mean, times = n), n, P, byrow=TRUE)
    }
    res = list(list(n1=n1, n2=n2, x1_new=loc, x2_new=matrix(c(1:n2),n2,1),
                    D_new = covmat), y)
  }
  return(res)
}
here()
country = "Nigeria"
iso3 = 'NGA'
file = here("Data","New","output_data",country,paste0(country,'-weekly-with-features.csv'))
df = read.csv(file, header = TRUE)
df = df[order(df$adm1_name,df$Datetime),]
file = here("Data","New","output_data",country,paste0(country,'-static-data.csv'))
df_loc = read.csv(file, header = TRUE)
df = df[order(df$adm1_name,df$Datetime),]
df$Muslim = df$Muslim/100

df$Ramadan_past90days_Muslimpop = df$Ramadan_past90days * df$Muslim
df$food_inflation_MPI = df$food_inflation_cubic_intp * df$MPI
df$log_food_inflation = log(df$food_inflation_cubic_intp)
df$log_food_inflation_MPI = df$log_food_inflation * df$MPI
df$log_fatalities_battles = log(df$n_fatalities_Battles_rolsum_90days + 1)
df$log_fatalities_violenceagainstcivilians = log(df$n_fatalities_Violence_against_civilians_rolsum_90days + 1)
df$log_fatalities_explosions = log(df$n_fatalities_Explosions.Remote_violence_rolsum_90days + 1)

region_90 = df$adm1_code[df$data_type=='SURVEY 90 days'] |> unique()
region_90_name = df$adm1_name[df$data_type=='SURVEY 90 days'] |> unique()
df_90 = df[df$data_type=='SURVEY 90 days',]

region_30 = df$adm1_code[df$data_type=='SURVEY 30 days'] |> unique()
region_30_name = df$adm1_name[df$data_type=='SURVEY 30 days'] |> unique()
df_30 = df[df$data_type=='SURVEY 30 days',]

region_p = df$adm1_code[df$data_type=='PREDICTION'] |> unique()
region_p_name = df$adm1_name[df$data_type=='PREDICTION'] |> unique()
df_pred = df[df$data_type=='PREDICTION',]


model_choice = "exp_expBM_cent_re1"
stan_file = here(stan_path,"kernel_exploration",model_choice,"GPst_est_rho1fixed.stan")
mod = cmdstan_model(stan_file, include_paths=stan_path)
params = c('alpha0','alpha1','alpha21','alpha22','rho2','sigma1','sigma')

data_list_tr = prep_stan_data(df, df_loc,region_90)
data_list_est = append(data_list_tr,
                       list(rho1=120,
                            Hurst2 =0.5
                       ))

fit = mod$sample(data = data_list_est, seed=123,
                 iter_warmup = 300,
                 iter_sampling = 400,
                 save_warmup = TRUE,
                 chains = 2,
                 parallel_chains = 2,
                 refresh = 50)

post_mean = fit$summary(variables = params)$mean
names(post_mean) = params

post_samples = fit$draws(format = "df",  inc_warmup = F)
post_samples$chain = as.character(post_samples$.chain)
for (param in params){
  gg = ggplot(data=post_samples, aes_string(x=".iteration", y = param, color="chain"))+
    geom_line() + theme_minimal() 
  print(gg)
}
for (param in params){
  gg = ggplot(data=post_samples, aes_string(x=param,color="chain", fill="chain"))+
    geom_histogram(alpha=0.5, position="identity") + theme_minimal() 
  print(gg)
}
