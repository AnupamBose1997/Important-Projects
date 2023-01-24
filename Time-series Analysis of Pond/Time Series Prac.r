setwd("E:/Leeds University/Time Series & Spectral Analysis/Practical")
load("pond.RData")
X
start(X)
end(X)
mean(X)
var(X)
sd(X)
range(X)
plot(X, xlab="years",ylab="Water levels(in inches)",xlim=c(1966,2015))
title(main = "Monthly water levels in the pond from 1965 to 2015")
abline(h=2.51,col="yellow")
#fit a linear trend

tt=1:length(X)
fit1 = lm(X~tt)

trend1 = fitted(fit1) # trend as a vector; convert it to a "ts" object
trend1 = ts(trend1, start=start(X), end=end(X), frequency=frequency(X))

lines(trend1, col="red")

resid1 = ts(residuals(fit1), start=start(X), end=end(X), 
            frequency=frequency(X))

plot(resid1, ylab="Residuals",main="Water levels after removing trend")

n = length(X)
jan = as.numeric((1:n %% 12) == 1)
feb = as.numeric((1:n %% 12) == 2)
mar = as.numeric((1:n %% 12) == 3)
apr = as.numeric((1:n %% 12) == 4)
may = as.numeric((1:n %% 12) == 5)
jun = as.numeric((1:n %% 12) == 6)
jul = as.numeric((1:n %% 12) == 7)
aug = as.numeric((1:n %% 12) == 8)
sep = as.numeric((1:n %% 12) == 9)
oct = as.numeric((1:n %% 12) == 10)
nov = as.numeric((1:n %% 12) == 11)
dec = as.numeric((1:n %% 12) == 0)

fit2 = lm(resid1 ~ 0 + jan + feb + mar + apr + may + jun + jul + aug +
            sep + aug + oct + nov + dec)
fit2

# Now see what variation is left after removal of trend and seasonal effects.
seasonal = ts(fitted(fit2), start=start(X), end=end(X),frequency=frequency(X))
fv = trend1 + seasonal
Y = X - fv
par(mfrow=c(1,1))
plot(X, xlab="",ylab="Water levels per month",xlim=c(1966,2015))
lines(fv, col="green")
plot(Y,ylab="Residuals",main="Water level after removing Seasonality & Trend")


#part2

acf(Y,xlim=c(0.0,2.5),xlab="Lag in time",ylab="acf of the residual data",main="Correlogram after removal of Seasonality")

#if it was white noise we would have applied MA process but as the correlation is very high it is not white noise so we will now apply AR(Auto Regressive)


#part3

ar(Y)$order
ar(Y)
(Y.ar1=ar(Y, order=1,aic = FALSE))
(Y.ar2=ar(Y, order=2,aic = FALSE))
(Y.ar3=ar(Y, order=3,aic = FALSE))

par(mfrow=c(1,1))

plot(Y.ar1$resid, ylab="AR1 residuals")
plot(Y.ar2$resid, xlab="Time (Years)",ylab="AR(2) residuals")
plot(Y.ar3$resid, ylab="AR3 residuals")

par(mfrow=c(3,1))

acf(Y.ar1$resid, na.action=na.omit, main="Correlogram of residual of AR1")
acf(Y.ar2$resid, na.action=na.omit, main="Correlogram of residual of AR2")
acf(Y.ar3$resid, na.action=na.omit, main="Correlogram of residual of AR3")


#Now we're selecting model ar2 for the reasons of coefficient and giving it Z as said in question
Z=Y.ar2$resid
Z

