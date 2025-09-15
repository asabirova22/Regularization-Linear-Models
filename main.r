## Libraries and function ridge required
library(cvTools); library(lars); library(glmnet)

## Loading the data
data(diabetes)
X<-as.matrix(diabetes$x); Y<-as.matrix(diabetes$y)
X<-scale(x=X,center=TRUE,scale = TRUE); Y<-scale(x=Y,center=TRUE,scale=TRUE)
DAT<-data.frame(cbind(X,Y)); colnames(DAT)<-c(colnames(X),"Y")

set.seed(0);
n<-nrow(X); K<-4;
cvFolds(n=n,K=K)->CV
CV$subsets[CV$which!=K]->Train; CV$subsets[CV$which==K]->Test

M2<-lars(x=X[Train,],y = Y[Train],type = "lasso",normalize = FALSE,intercept = FALSE)

P2<-predict.lars(object = M2,newx = X[Test,],type = "fit")
Yobs<-matrix(nrow=nrow(P2$fit),ncol=ncol(P2$fit),byrow=FALSE,Y[Test])
MSE2<-apply((Yobs-P2$fit)^2,2,mean)
par(mar=c(4,4,1,1),mfrow=c(1,2))
plot(MSE2) ## the MSE
abline(v=which.min(MSE2),col="red",lty=2)
plot(M2) ## The lasso path

## Results to be shown later

betalasso<-M2$beta[which.min(MSE2),]
shrinklasso<-sum(abs(betalasso))/sum(abs(M2$beta[nrow(M2$beta),]))
abline(v=shrinklasso,col="red",lty=2)

rangelambda<-10^seq(from=-6,to=6,length.out=200)

MSElasso<-min(MSE2)

M3<-glmnet(x=X[Train,],y=Y[Train],family="gaussian",alpha = 0.5,
           lambda = rangelambda)

P3<-predict.glmnet(object = M3,newx = X[Test,],type="response")
Yobs<-matrix(nrow=nrow(P3),ncol=ncol(P3),
             byrow=FALSE,Y[Test])

MSE3<-apply((Yobs-P3)^2,2,mean)
lambdanet<-rangelambda[which.min(MSE3)]
par(mar=c(4,4,1,1),mfrow=c(1,2)) ## The MSE
plot(rangelambda,MSE3,type="l",log="x")
abline(v=lambdanet,col="red",lty=2)
plot(M3) ## The elastic net path

## Results to be shown later
betanet<-M3$beta[,which.min(MSE3)]
MSEnet<-min(MSE3)

M4<-lm(Y~.-1,data=DAT[Train,])
P4<-predict.lm(object = M4,newdata = DAT[Test,],type="response")
betaols<-M4$coefficients
MSEols<-mean( (Y[Test]-P4)^2)

MSEall<-cbind(MSElasso,MSEnet,MSEols) ## the MSE
betaall<-cbind(betalasso,betanet,betaols) ## Coefficients
round(MSEall,4);

round(betaall,4)

L1s<-apply(X = abs(betaall),MARGIN = 2,FUN = sum)
L2s<-apply(X = betaall^2,MARGIN = 2,FUN = sum)

shrink<-rbind(L1s[1:2]/L1s[3],L2s[1:2]/L2s[3]);
rownames(shrink)<-c("L1","L2"); round(100*shrink,2)
