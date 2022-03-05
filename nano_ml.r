# Akmal Aulia (in collaboration with UPM)
# Created: 29/12/2020


rm(list = ls()) # remove all data




# ----------- set constant --------------------
t_fix <- c(15,30,45,60) # chosen measured time.  
dist_min <- 0
dist_max <- 100
dist_len <- 100
dist_fix <- as.matrix(seq(dist_min, dist_max, length.out=dist_len))
colnames(dist_fix) <- c("Distance")
frac_tr <- 0.66 # fraction for training set
k <- 5 # k-fold cross validation
hnode_1 <- 6 # neuralnet number of hidden nodes in the 1st hidden layer
hnode_2 <- 4 # neuralnet number of hidden nodes in the 1st hidden layer



# ------------ call library --------------------------

# library(randomForest) # for machine learning modeling
# library(mice) # for imputation
# library(Metrics) # use rmse() to compute RMSE
# library(neuralnet) # for neural network model
# library(plotly)
# library(processx)

if (!require("randomForest")) install.packages("randomForest")
if (!require("mice")) install.packages("mice")
if (!require("Metrics")) install.packages("Metrics")
if (!require("neuralnet")) install.packages("neuralnet")
if (!require("plotly")) install.packages("plotly")
# if (!require("processx")) install.packages("processx")




# ----------- read csv files ------------------

# read input data
dat <- read.csv(file="dat.csv", header=TRUE)
ncol(dat) # check number of columns




# ------------ imputation ---------------------

# impute missing values
# perform imputation
tempData <- mice(dat,
                 m=5,
                 maxit=50,
                 meth="pmm",
                 seed=500,
                 print=FALSE)

d <- mice::complete(tempData,1) # cleaned data
md.pattern(d, plot=FALSE) # check for imputation status




# -------------------------------------------
#
# Neural Network implementation
#
# -------------------------------------------

# data normalization (training set)
maxs <- apply(d, 2, max)
mins <- apply(d, 2, min)
scaled <- as.data.frame(scale(d, center=mins, scale=maxs-mins))




# -------------- Training Neural Network Model -----------------

# set training
train_scaled <- scaled

# train neural net
nn <- neuralnet(Response ~., data=train_scaled, hidden=c(hnode_1,hnode_2), linear.output=T)

# do prediction on training set
pr_scaled_tr.nn <- compute(nn, train_scaled[,1:2])

# denormalized training-based prediction
pr_tr.nn <- as.matrix(pr_scaled_tr.nn$net.result)*as.numeric(maxs[3]-mins[3]) + as.numeric(mins[3])
pr_tr.nn[pr_tr.nn<0] <- 0 # truncate negative values to 0
pr_tr.nn[pr_tr.nn>100] <- 100 # truncate values above 100 to be 100 (value is in percentage)

# store predicted vs actual (this refers to training set only, no test set)
x <- as.data.frame(cbind(pr_tr.nn, d$Response))
colnames(x) <- c("Predicted", "Actual")

# assess training accuracy
# fit <- lm(formula=Predicted ~ 0+Actual, x)
r_sq <- (cor(x$Predicted,x$Actual))^2
#r_sq <- summary(fit)$r.squared
inc_recov.mse <- rmse(x$Predicted, x$Actual)

# create title string for plotting later
main_str <- paste("Train: (RMSE,R_sq) = ", "(", round(inc_recov.mse,2), ",", round(r_sq,3), ")" )
# main_str <- gsub(" ", "", main_str) # trim main_str




# ------------------ Testing Neural Network Model (for plot_both.pdf) ---------------------

# prepare pdf file
pdf(file="plot_extrap.pdf")


# set number of plots
nplots <- length(t_fix)
par(mfrow=c(nplots/2,nplots/2))
#par(mar=c(4,4,4,4))



for (i in 1:length(t_fix))
{
  
  # ---------------------- testing ----------------------
  
  # create test set
  col_time <- matrix(t_fix[i], nrow=dist_len, ncol=1)
  colnames(col_time) <- c("Time")
  test_ <- cbind(col_time, dist_fix)
  
  # data normalization (test set)
  test_scaled <- as.data.frame(scale(test_, center=mins[1:2], scale=maxs[1:2]-mins[1:2]))
  
  # do prediction
  pr_scaled_ts.nn <- compute(nn,test_scaled)
  
  # denormalized predicted values
  pr_ts.nn <- as.matrix(pr_scaled_ts.nn$net.result)*as.numeric(maxs[3]-mins[3]) + as.numeric(mins[3])
  pr_ts.nn[pr_ts.nn < 0] <- 0 # truncate values below 0 to 0
  pr_ts.nn[pr_ts.nn > 100] <- 100 # truncate values above 100 to 100
  
  # create res_tab.nn
  res_tab.nn <- as.data.frame(cbind(test_, pr_ts.nn))
  colnames(res_tab.nn)[3] <- "Response" # rename the response column
  
  # write file
  str_filename <- paste("Results_fixed_time_at_", t_fix[i], "_minutes.csv")
  str_filename <- gsub(" ", "", str_filename)
  write.csv(file=str_filename, res_tab.nn, row.names=FALSE) # write results to a file in active directory
  
  
  
  
  # -------------------- plotting -------------------------
  
  # plot # 1
  main_str2 <- paste("Pred: Fixed Time = ", t_fix[i], "minutes." )
  #main_str2 <- gsub(" ", "", main_str2) # trim main_str
  plot(res_tab.nn$Distance, res_tab.nn$Response, type="l", col="blue", main=main_str2, xlab="Distance", ylab="Knockdown (%)", ylim=c(0,110))
  lines(d$Distance[d$Time==t_fix[i]], d$Response[d$Time==t_fix[i]], col="red", type="p") # plot actual data as red dots
  
}

# turn off pdf generator
dev.off()

# prepare pdf file
pdf(file="plot_evaluate_training.pdf")

# plot # 2
plot(x$Predicted, x$Actual, main=main_str,xlab="Predicted", ylab="Actual")
lines(seq(0,100), seq(0,100), type="l", col="red")

# turn off pdf generator
dev.off()





# --------------------- Plot 3D surface (neuralnet-generated) ---------------------

# convention
#   first axis:  time
#   second axis: distance

# range of axis for neuralnet-based 3D plot
nn_time_max = maxs[1]
nn_time_len = 100
nn_dist_max = dist_max # default: maxs[2]
nn_dist_len = 100


# create matrix
nn_time <- seq(mins[1], nn_time_max, length.out=nn_time_len)
nn_dist <- seq(mins[2], nn_dist_max, length.out=nn_dist_len)
test_3d_ <- cbind(nn_time,nn_dist)
colnames(test_3d_) <- c("Time", "Distance")

# set mesh_mat
time_mat <-  matrix( 0, nrow = nn_time_len, ncol=nn_dist_len) # intiate matrix
dist_mat <-  matrix( 0, nrow = nn_time_len, ncol=nn_dist_len) # intiate matrix
mesh_mat <-  matrix( 0, nrow = nn_time_len, ncol=nn_dist_len) # intiate matrix
for (i in 1:nn_time_len)
{
  for (j in 1:nn_dist_len)
  {
    # set input for neural net
    ts <- matrix(0, nrow=1, ncol=2)
    ts[1,1] <- nn_time[i]
    ts[1,2] <- nn_dist[j]
    colnames(ts) <- c("Time", "Distance")
    
    # normalize
    ts_scaled <- as.data.frame(scale(ts, center=mins[1:2], scale=maxs[1:2]-mins[1:2]))
    
    # predict response of test_3d_scaled using neuralnet
    pr_ts_scaled.nn <- compute(nn,ts_scaled)
    
    # denormalize predicted responses
    pr_ts.nn <- as.matrix(pr_ts_scaled.nn$net.result)*as.numeric(maxs[3]-mins[3]) + as.numeric(mins[3])
    
    # truncation
    pr_ts.nn[pr_ts.nn < 0] <- 0 # truncate values below 0 to 0
    pr_ts.nn[pr_ts.nn > 100] <- 100 # truncate values above 100 to 100
    
    # fill time_mat and dist_mat (for debugging)
    time_mat[i,j] <- nn_time[i]
    dist_mat[i,j] <- nn_dist[j]
    
    # fill mesh_mat 
    mesh_mat[i,j] <- pr_ts.nn
    
    
  }
}


# plot 3d
fig <- plot_ly(x=time_mat,y=dist_mat,z=mesh_mat,type="surface", scene="scene1")
# fig <- fig %>% add_surface(showscale=TRUE)
fig <- fig %>% layout(
  title = "Neuralnet-based Knockdown (%)",
  scene1 = list(
    xaxis = list(title = "Time (min)"),
    yaxis = list(title = "Distance (m)"),
    zaxis = list(title = "Knockdown (%)"),
    aspectmode='manual',
    aspectratio = list(x=1, y=1, z=0.7)
  ))



fig


htmlwidgets::saveWidget(fig, "index.html")
# -----------------------------------------------------------------------------------------





# ----------------- 2d sections of the 3d plot -------------------

# prepare pdf file
pdf(file="plot_2d_of_3d.pdf")

col_index <- 1
for (i in round(seq(1,100,by=10),0))
{
  if (i==1)
  {
    plot(dist_mat[i,],mesh_mat[i,], type="l", col=col_index, xlab=c("Distance"), ylab=c("Knockdown (%)"), main="Time is fixed for each line.")
  } else {
    lines(dist_mat[i,],mesh_mat[i,], type="l", col=col_index)
  }
  
  col_index <- col_index + 1
}

# turn off pdf generator
dev.off()
# -----------------------------------------------------------------------------------------


# plot neural net
plot (nn)





# --------------------------- k-fold cross validation ----------------------
# split training and test set

split_indices <- sample(c(rep(0, frac_tr * nrow(scaled)),
                        rep(1, (1-frac_tr) * nrow(scaled))))
# note that index 0 for training, and 1 for testing.
# also, the 0 index rows will be split into training and validation sets
scaled_tr <- scaled[split_indices == 0,]
scaled_ts <- scaled[split_indices == 1,]
d_tr <- d[split_indices == 0,]
d_ts <- d[split_indices == 1,]

# check dimension
dim(scaled_tr)
dim(scaled_ts)
dim(d_tr)
dim(d_ts)

# performing k-fold cross validation

indices <- sample(1:nrow(scaled_tr))
folds <- cut(indices, breaks=k, labels=FALSE)

scaled_data <- scaled_tr[1:2]
scaled_targets <- scaled_tr[3]

all_mse <- c()
all_r_sq <- c()
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  
  # prepare validation data
  val_indices <- which(folds == i, arr.ind=TRUE)
  val_scaled_data    <- scaled_tr[val_indices,] # generate input data for validation 
  
  # prepare unscaled Response associated with val_scaled_data
  val_unscaled_data    <- d_tr[val_indices,]
  
  # prepare training data
  partial_train_scaled_data   <- scaled_tr[-val_indices,]
  
  # train neural net
  nn <- neuralnet(Response ~., data=partial_train_scaled_data, hidden=c(hnode_1,hnode_2), linear.output=T)
  
  # do prediction on validation set
  pr_scaled_tr.nn <- compute(nn, val_scaled_data[,1:2])
  
  # denormalized training-based prediction
  pr_tr.nn <- as.matrix(pr_scaled_tr.nn$net.result)*as.numeric(maxs[3]-mins[3]) + as.numeric(mins[3])
  pr_tr.nn[pr_tr.nn<0] <- 0 # truncate negative values to 0
  pr_tr.nn[pr_tr.nn>100] <- 100 # truncate values above 100 to be 100 (value is in percentage)
  
  # store predicted vs actual (this refers to training set only, no test set)
  x <- as.data.frame(cbind(pr_tr.nn, val_unscaled_data[,3]))
  colnames(x) <- c("Predicted", "Actual")
  
  # assess validation accuracy
  r_sq <- (cor(x$Predicted,x$Actual))^2
  inc_recov.mse <- rmse(x$Predicted, x$Actual)
  
  # record values
  all_mse <- rbind(all_mse, inc_recov.mse)
  all_r_sq <- rbind(all_r_sq, r_sq)

}
# ----------------------------------------------------------------------------------------------





# --------------------- report k-fold cross validation results------------------------------
average_mse <- mean(all_mse)
average_r_sq <- mean(all_r_sq)

average_mse
average_r_sq

mainstr1 <- paste("Average MSE =", round(average_mse,2))
mainstr2 <- paste("Average R_sq =", round(average_r_sq,2))

pdf(file="k_fold validation.pdf")
par(mfrow=c(2,1))
plot(all_mse,type="p",cex=3,main=mainstr1, xlab="Fold", ylab="MSE",col="red", pch=20, ylim=c(0,25))
plot(all_r_sq,type="p",cex=3,main=mainstr2, xlab="Fold", ylab="R-squared", col="orange", pch=20, ylim=c(0,1.1))

# turn off pdf generator
dev.off()
# -----------------------------------------------------------------------------------





# ----------------------- model performance using test set --------------------------
# train neural net using both training and validation sets
nn <- neuralnet(Response ~., data=scaled_tr, hidden=c(hnode_1,hnode_2), linear.output=T)

# do prediction on test set
pr_scaled_tr.nn <- compute(nn, scaled_ts[,1:2])

# denormalized training-based prediction
pr_tr.nn <- as.matrix(pr_scaled_tr.nn$net.result)*as.numeric(maxs[3]-mins[3]) + as.numeric(mins[3])
pr_tr.nn[pr_tr.nn<0] <- 0 # truncate negative values to 0
pr_tr.nn[pr_tr.nn>100] <- 100 # truncate values above 100 to be 100 (value is in percentage)

# store predicted vs actual (this refers to training set only, no test set)
x <- as.data.frame(cbind(pr_tr.nn, d_ts[,3]))
colnames(x) <- c("Predicted", "Actual")

# assess testing accuracy
r_sq <- (cor(x$Predicted,x$Actual))^2
inc_recov.mse <- rmse(x$Predicted, x$Actual)

mainstr1 <- paste("< R_sq, MSE > = <", round(r_sq,2), ",", round(inc_recov.mse,2), ">")

pdf(file="test_set_evaluation.pdf")
par(mfrow=c(1,1))
plot(x$Actual, x$Predicted,type="p",cex=2,main=mainstr1, xlab="Actual", ylab="Predicted",col="blue", pch=20, xlim=c(0,100), ylim=c(0,100))
lines(seq(0,100), seq(0,100), type="l", col="red")

# turn off pdf generator
dev.off()

# print model performance on test set
r_sq
inc_recov.mse

# -----------------------------------------------------------------------------------
