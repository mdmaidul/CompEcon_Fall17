library(AER)
library(plm)
library(haven)
library(readxl)
library("texreg")

## From 2001 to 2016

gfdi1 <- read_excel("C:/Users/mmic/Documents/Python Tutorial/Python/GDP_FDI.xlsx", 
                    range = "Sheet1!A2:G98")
attach(gfdi1)

## Log conversion

gfdi1$lGDP <- log(gfdi1$GDP)
gfdi1$lEGS <- log(gfdi1$EGS)
gfdi1$lFDI <- log(gfdi1$FDI)

## Squared term of lFDI

gfdi1$lFDI2 <- (gfdi1$lFDI*gfdi1$lFDI)

## Attaching the file
attach(gfdi1)
Y<-cbind(lGDP)
X<-cbind(GFCF, EPR, lEGS, lFDI)

# Estimation of pooled OLS

pooling <- plm(Y~X, data=gfdi1,index=c("Country", "Year"), model="pooling")
summary(pooling)

# Fixed effect for within estimator

fixed <- plm(Y~X, data=gfdi1,index=c("Country", "Year"), model="within")
summary(fixed)

# Displaying the fixed effects (Constant for each country)

fixef(fixed)

# Random Effect Model

random <- plm(Y~X, data=gfdi1, model="random")
summary(random)

# Hausman test
phtest(fixed,random)

# Printing the table

print(texreg(list(pooling, fixed, random), 
             caption = "Exploring the relationship between GDP and FDI for six South Asian Countries.", 
             dcolumn = TRUE, booktabs = TRUE,use.packages = FALSE, 
             custom.model.names=c('Pooled','Fixed Effect','Random Effect'), 
             custom.coef.names = c('Intercept','GFCF','EPR','log(EGS)','log(FDI)'), 
             label = "tab:3", float.pos = "hb!"))

# Fixed effect model estimation incorporating time dummies

fixedt <- plm(Y~X+factor(Year), data=gfdi1,index=c("Country", "Year"), model="within")
summary(fixedt)

## Comparing the fixed effect model incorporating time dummies
print(texreg(list(fixedt, fixed), 
             caption = "Fixed Effect(FE) Model: with and without time dummies.", 
             dcolumn = TRUE, booktabs = TRUE,use.packages = FALSE, 
             custom.model.names=c('FE with time dummies','FE without time dummies'), 
             custom.coef.names = c('GFCF','EPR','log(EGS)','log(FDI)',
                                   'D02','D03','D04','D05','D06','D07',
                                   'D08','D09','D10','D11','D12','D13',
                                   'D14','D15','D16'), 
             label = "tab:2", float.pos = "hb!"))

# Fixed effect model estimation incorporating non-linear FDI term

fixed2 <- plm(Y~X+lFDI2,data=gfdi1,index=c("Country", "Year"), model="within")
summary(fixed2)

## Comparing the fixed effect model incorporating non-linear FDI term
print(texreg(list(fixed, fixed2), 
             caption = "Fixed Effect(FE) Model: with and without non-linear FDI term.", 
             dcolumn = TRUE, booktabs = TRUE,use.packages = FALSE, 
             custom.model.names=c('FE without squared log(FDI)','FE with squared log(FDI)'), 
             custom.coef.names = c('GFCF','EPR','log(EGS)','log(FDI)','log(FDI)*log(FDI)'), 
             label = "tab:3", float.pos = "hb!"))
