##Limpiar su area de trabajó 
rm(list=ls())

##Debe instalar la biblioteca Bessel, se puede hacer con el comando:
install.packages("Bessel")

##Cargar la libreria
library(Bessel)

##Seleccione el archivo "cetes364.csv"
cetes<-read.csv(file.choose())
##

##fecha de inicio de análisis, justificación en el trabajo escríto 
 
fia<-"20/04/2007"
s<-as.numeric(row.names(cetes[cetes$Fecha == fia,]))
cetes <- cetes[s:nrow(cetes),]
y<-as.Date(as.character(cetes$Fecha),"%d/%m/%Y")
cetes["fecha"]<-y
cetes["diasem"]<-weekdays(y)
diaS<-"miércoles"
h<-cetes$diasem == diaS
r<-(cetes[cetes$diasem == diaS,]$tasa)/100
y<-cetes[cetes["diasem"]==diaS,]$fecha
r[r == 0]<-cetes["2926","tasa"]/100
par(mfrow=c(1,1))

##Un vistazo a nuestros datos
plot(y,r,xlab = "Periodo de análisis", ylab = "Nivel de la tasa", 
     main = "Tasas cetes 360 (Tenor Anual Efectivo)", type = "l",col = "#CD5C5C")
box(lwd = 2)

##Función que cálcula los estimadores por mínimos cuadrados ordinarios
CIR.OLS <- function(r,dt){
                n=length(r)
                Y = t(t((r[2:n]-r[1:(n-1)])/sqrt(r[1:(n-1)])))
                Z = cbind(dt/sqrt(abs(r[1:(n-1)])),dt*sqrt(abs(r[1:(n-1)])))
                beta = (solve(t(Z)%*%Z)%*%t(Z))%*%Y 
                sigma = sqrt(sum(as.numeric( Y -  Z%*%beta)^2))*sqrt(1/(n*dt))  
                alpha = -beta[2]
                mu = beta[1]/alpha  
                return(c(alpha,mu,sigma))
}

##Función que cálcula la verosimilitud de la muestra
CIR.ll<-function(a,r,dt){
                n<-length(r)
                c<-(2*a[1])/(a[3]^(2)*(1 - exp(-a[1]*dt)))
                q<-(2*a[1]*a[2])/(a[3]^2) - 1
                u<-c*r[1:(n-1)]*exp(-a[1]*dt)
                v<-c*r[2:n]
                f<-(n-1)*log(c) + sum(-u-v + (q/2)*(v/u) + log(BesselI(2*sqrt(u*v),q,expon.scaled = TRUE)) + 2*sqrt(u*v))
                return(-f) 
        }
##
CIR<-function(a,r_0,n,dt){
          r<-numeric(n)             
          r[1]<-r_0
          z<-rnorm(n-1) 
          sapply(1:(n-1),function(i)r[i+1]<<-r[i] + a[1]*(a[2] - r[i])*dt + a[3]*sqrt(dt)*sqrt(max(r[i],0))*z[i])
          return(r)
          }

##Primeras simulaciones
CIR.SIM<-function(r,date,tenor = True,deltat = 1/50,N = 50,seed = 46,nsims = 50000,conf = c(.005,.995),n.tr=20){ 
 ifelse(tenor,r<-r,r<-log(1+r))
 v<-tail(date,n = 1) 
 Y<-as.Date(sapply(seq(0,(N-1)*7,7),function(x)v + x),origin = "1970-01-01")
 a<-CIR.OLS(r,dt=deltat) 
 a<-nlm(CIR.ll,a,stepmax = 0.001,iterlim = 100,r=r,dt = deltat)$estimate 
 r_0<-r[length(r)] 
  
 set.seed(seed)
 d<-sapply(rep(N,nsims),function(x)CIR(a=a,r_0,n=x,dt = deltat)) 
 s<-apply(d,1,quantile,probs = .5)
 s.1<-apply(d,1,quantile,probs = conf[2])
 s.3<-apply(d,1,quantile,probs = conf[1])      

 blues<-c("#D0EBF6","#B9D2DC","#AAC2CB","#99AEB6","#8DA1A9","#7B919A","#6A828C","#527280","#456E80","#325D70")
 blues<-rep(blues,30)

 par(mfrow = c(1,3))
 samp<-sample(1:nsims, size = 1) 
 Samp<-sample(1:nsims, size = n.tr)                      
 plot(date,r,col = "#99AEB6",main="Tasas CETES 360",
      xlab = "Fechas",ylab = "Nivel de la tasa", type = "l",lwd = 2) 
      box(lwd=2) 
 plot(Y,d[,samp],col = "red",main="Simulación tasas CETES 360 con modelo CIR \n(Una Trayectoria)",
      xlab = "Fechas",ylab = "", type = "l",lwd=2)
      box(lwd=2) 
 plot(Y,s.3,col = "red",main="Media e intervalo de confianza \n+ trayectorias",
      xlab = "Fechas", type = "l",ylab ="", 
      ylim=c(min(s.3),max(s.1)),lwd = 2, lty = "dashed")
      sapply(1:n.tr,function(i)lines(Y,d[,Samp[i]],col = blues[i],lwd =2))
      lines(Y,s.1,col = "red",lwd = 2,lty ="dashed")
      lines(Y,s,col = "black",lwd = 3,lty = "dotdash")
      box(lwd=2) 
 }

##Tarda aproximadamente 6.17 segundos en correr (Tasa con cap continua)
system.time(
CIR.SIM(r,y,tenor = 0)
)
##Tarda aproximadamente 6.17 segundos en correr (Tasa con cap simple)
system.time(
CIR.SIM(r,y,tenor = 1)
)

##Comparativa vs tasa real
CIR.SIM.M<-function(r,date,tenor = True,deltat = 1/50,N = 50,nsims = 50000,conf = .995,n.tr = 8,color = "g"){ 
 ifelse(tenor,r<-r,r<-log(1+r))
 v<-tail(date,n = 1) 
 Y<-as.Date(sapply(seq(0,(N-1)*7,7),function(x)v + x),origin = "1970-01-01")
 a<-CIR.OLS(r,dt=deltat) 
 a<-nlm(CIR.ll,a,stepmax = 0.001,iterlim = 100,r=r,dt = deltat)$estimate 
 m<-length(r)
 r_0<-r[m] 
 
 d<-sapply(rep(N,nsims),function(x)CIR(a=a,r_0,n=x,dt = deltat)) 
 s<-apply(d,1,quantile,probs = .5)
 s.1<-apply(d,1,quantile,probs =  (1 + conf)/2)
 s.3<-apply(d,1,quantile,probs = (1 - conf)/2 )      
 R<-r[seq(m-25,m-1)] 
 
 par(mfrow = c(1,1))
 samp <- sample(1:nsims,size = n.tr)                        
 M<-c(R,d[,samp[1]]) 
 H<-length(M)
 g<-length(R)
 y<-date[seq(m-25,m-1)]
 n.y<-length(y)
 y<-c(y,Y)

 co<-numeric(1)
 ifelse(color =="g",co<-"#9FFF0A",co<-"blue") 
 blues<-c("#D4FDFF","#28F7FF","#42DEFF","#00CBFF","#00B7E6","#00A8D3","#008BD3","#007BBA","#0038AB","#6600D8")
 blues<-rep(blues,30)
 greens<-c("#84FCC9","#32FFA7","#18FF87","#18FF56","#08FF1D","#20E100","#36CF00","#2FB300","#289A00","#207B00")
 greens<-rep(greens,30)
 ifelse(color == "g",color<-greens,color<-blues) 
 plot(y,M,col = "blue",main=paste("Tasas CETES 360 y posibles escenarios (CIR) \n+ intervalos de confianza al ",as.character(conf*100),"%"),
      xlab = "Fechas",ylab = "Nivel de la tasa", type = "n",pch=20,ylim=c(min(min(s.3),min(R)),max(max(s.1),max(R)))) 
      lines(y[1:(n.y+1)],c(R,r[m]),col = co,lwd = 2)
      abline(h = g) 
      sapply(1:n.tr,function(i)lines(Y,d[,samp[i]],col = color[i],lwd =2))
      lines(Y,s,col = "black",lwd = 3,lty ="dashed")
      lines(Y,s.1,col = "#FA8F00",lwd = 2,lty="dotdash")
      lines(Y,s.3,col = "#FA8F00",lwd = 2,lty="dotdash")
 legend("bottomleft",c("Tasa observada",paste("Intervalo confianza ",as.character(conf*100),"%"),
        "Trayectorias","Media"),lty = c("solid","dotdash","solid","dashed"),
         col = c(co,"#FA8F00",color[as.integer(n.tr/2)],1),cex=0.75,
         box.lwd = 2)
         box(lwd = 2) 
 }

##Tarda aproximadamente 7 s en correr, para 50000 simulaciones
##comparativa Vasicek vs mercado
##conf 99,5% (Tenor continuo) 
system.time(
CIR.SIM.M(r,y,tenor = 0,nsims =10000,n.tr = 9,color = "b")
)
##confianza 95%
system.time(
CIR.SIM.M(r,y,tenor = 0,nsims =10000,n.tr = 9,conf = .95)
)


##Tarda aproximadamente 7 s en correr, para 50000 simulaciones
##comparativa Vasicek vs mercado
##conf 99,5% (Tenor anual simple) 
system.time(
CIR.SIM.M(r,y,tenor = 1,nsims =10000,n.tr = 9,col = "g")
)
##confianza 95%
system.time(
CIR.SIM.M(r,y,tenor = 1,nsims =10000,n.tr = 9,col = "b",conf = .95)
)

