#!/usr/bin/env python
# coding: utf-8

# In[15]:


from enum import Enum
import statistics
import pandas as pd
from datetime import timedelta
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.signal import argrelextrema


# In[3]:


def Triang(price="Close"):
    root = tk.Tk()
    root.withdraw()
    file_path=filedialog.askopenfilename()
    A = pd.read_csv(file_path)
    m=file_path.split("/")
    nombre = m[len(m)-1].split(".")[0]
    
    p = A[price].to_numpy()
    date = pd.to_datetime(A["Date"])

    SIGN = []
    #date.iloc[n:]
    for i in range(0,p.shape[0] - 1):
        SIGN.append(p[i] - p[i+1])
    
    S = np.sign(np.array(SIGN))
    
    #A  lo mejor lo borro 
    I = []
    s = S[0].copy()
    for i in range(1,len(S)-1): 
        if S[i] == s:
            s = s
        else: 
            I.append(i)
            s = S[i].copy()
    p1 = p[I]
    d1 =date.iloc[I]
    g=[]
    for i in range(0,p1.shape[0] - 1 - 4):
        if (abs(p1[i:i+4][2] - 0.5)<=np.mean([p1[i:i+4][2],p1[i:i+4][0]]))and(abs(p1[i:i+4][0] - 0.5)<=np.mean([p1[i:i+4][2],p1[i:i+4][0]])):
            if (all(p1[i:i+4][1] < p1[i:i+4][[0,2,3]]))and(all((p1[i:i+4][3] < p1[i:i+4][[0,2]]))):
                if (abs(p1[i:i+4][3] - p1[i:i+4][1]) > 0.003*p1[i:i+4][1])and(p1[i:i+4][0]<=p1[i:i+4][2]):
                    g.append(np.arange(i,i+4).tolist())
    g = np.unique(np.array(g).reshape(1,np.array(g).shape[0]*np.array(g).shape[1])).tolist()
    ##Triángulos descendentes
    r=[]
    for i in range(0,p1.shape[0] - 1 - 4):
        if (abs(p1[i:i+4][2] - 0.5)<=np.mean([p1[i:i+4][2],p1[i:i+4][0]]))and(abs(p1[i:i+4][0] - 0.5)<=np.mean([p1[i:i+4][2],p1[i:i+4][0]])):
            if (all(p1[i:i+4][1] > p1[i:i+4][[0,2,3]]))and(all((p1[i:i+4][3] > p1[i:i+4][[0,2]]))):
                if (abs(p1[i:i+4][3] - p1[i:i+4][1]) > 0.003*p1[i:i+4][1])and(p1[i:i+4][0]>=p1[i:i+4][2]):
                    r.append(np.arange(i,i+4).tolist())
    r = np.unique(np.array(r).reshape(1,np.array(r).shape[0]*np.array(r).shape[1])).tolist()
    if (len(r)>0)and(len(g)>0):  
        p1inv=p1[g]
        d1inv=d1.iloc[g]
        p1hs=p1[r]
        d1hs=d1.iloc[r]
        get_ipython().run_line_magic('matplotlib', 'qt')
        plt.style.use("seaborn")
        
        plt.plot_date(d1,p1, marker = "",linestyle ="-",color = "black",linewidth = 1)
        plt.plot_date(d1inv,p1inv, marker = "8",color = "red",label = "Triángulos ascendentes",
                 alpha = 0.25)
        plt.plot_date(d1hs,p1hs, marker = "8",color = "#0F0FF7",label = "Triangulos descendentes",
                 alpha=0.25)
        
        plt.title("Posibles formaciones triangulares en la acción: " + nombre)
        plt.xlabel('Fechas')
        plt.ylabel('Precio de la acción ' + nombre)
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif(len(r)==0)&(len(g)>0):
        print("No existen triángulos descendentes en su gráfica")
        p1inv=p1[g]
        d1inv=d1.iloc[g]
        
        get_ipython().run_line_magic('matplotlib', 'qt')
        plt.style.use("seaborn")
        
        plt.plot_date(d1,p1, marker = "",linestyle ="-",color = "black",linewidth = 1)
        plt.plot_date(d1inv,p1inv, marker = "h",color = "red",label = "Triángulos ascendentes",
                 alpha =0.25)
        
        plt.title("Posibles formaciones triangulares en la acción: " + nombre)
        plt.xlabel('Fechas')
        plt.ylabel('Precio de la acción ' + nombre)
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif(len(r) < 0)&(len(g)==0):
        print("No existen triángulos ascendentes en su gráfica")
        p1hs=p1[r]
        d1hs=d1.iloc[r]
        
        get_ipython().run_line_magic('matplotlib', 'qt')
        plt.style.use("seaborn")
        
        plt.plot_date(d1,p1, marker = "",linestyle ="-",color = "black",linewidth = 1)
        plt.plot_date(d1hs,p1hs, marker = "h",color = "red",label = "Triangulos descendentes"
                 ,alpha = 0.25)
        
        plt.title("Posibles formaciones triangulares en la acción: " + nombre)
        plt.xlabel('Fechas')
        plt.ylabel('Precio de la acción ' + nombre)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No existen dichos patrones en su gráfica")


# In[3]:


def bottoms(cota = 0.25, delta = 10, precio = "Close"):
    root = tk.Tk()
    root.withdraw()
    file_path=filedialog.askopenfilename()
    ticker_df = pd.read_csv(file_path)
    m=file_path.split("/")
    nombre = m[len(m)-1].split(".")[0]
    
    x_data = ticker_df.index.tolist()     
    y_data = ticker_df[precio]
    
    
    x = np.linspace(0, max(ticker_df.index.tolist()), max(ticker_df.index.tolist()) + 1)
    
    pol = np.polyfit(x_data, y_data, 17)
    y_pol = np.polyval(pol, x)
    
    data = y_pol
    
    min_max = np.diff(np.sign(np.diff(data))).nonzero()[0] + 1          
    l_min = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1      
    l_max = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1      
    
    print('l_min: ', l_min)
    
    dict_i = dict()
    dict_x = dict()
    
    df_len = len(ticker_df.index)                   
    
    for element in l_min:                           
        l_bound = element - delta                    
        u_bound = element + delta                    
        x_range = range(l_bound, u_bound + 1)        
        dict_x[element] = x_range                    
        y_loc_list = list()
        for x_element in x_range:
            if x_element > 0 and x_element < df_len:                
                y_loc_list.append(ticker_df.Low.iloc[x_element])
        dict_i[element] = y_loc_list                 
    y_delta = 0.12                              
    threshold = np.quantile(ticker_df[precio],cota) 
    y_dict = dict()
    mini = list()
    suspected_bottoms = list()
    for key in dict_i.keys():                      
        mn = sum(dict_i[key])/len(dict_i[key])   
        price_min = min(dict_i[key])    
        mini.append(price_min)                   
        l_y = mn * (1.0 - y_delta)                
        u_y = mn * (1.0 + y_delta)
        y_dict[key] = [l_y, u_y, mn, price_min]
    for key_i in y_dict.keys():    
        for key_j in y_dict.keys():    
            if (key_i != key_j) and (y_dict[key_i][3] < threshold):
                suspected_bottoms.append(key_i) 
    get_ipython().run_line_magic('matplotlib', 'qt')
    plt.style.use("seaborn")            
    plt.figure(figsize=(20, 10), dpi= 120, facecolor='w', edgecolor='k')
    
    plt.plot(x_data, y_data, 'o', markersize=1.5, color='magenta', alpha=0.7)
    
    plt.plot(x_data, ticker_df[precio],linestyle="-", color='green')
    for position in suspected_bottoms:
        plt.axvline(x=position, linestyle='-.', color='r')
    plt.axhline(threshold, linestyle='--', color='b',label= "Quantil " + str(cota*100) + "% de sus datos")    
    for key in dict_x.keys():
        for value in dict_x[key]:
            plt.axvline(x=value, linestyle='-', color = 'lightblue', alpha=0.2)
    
    plt.title("Posibles fondos múltiples en la acción " + nombre)
    plt.ylabel(nombre + "vs Ubicación posibles fondos absolutos (rojos), relativos (turquesa)")
    plt.xlabel("fecha donde 0=" + ticker_df["Date"].tolist()[0] + " y " + 
               str(max(x_data)) + "= " + ticker_df["Date"].tolist()[ticker_df["Date"].shape[0] - 1])
    plt.legend()        
    plt.show()


# In[4]:


def HandS(smoothing = 3,window_range = 10,price ="close"):
    root = tk.Tk()
    root.withdraw()
    file_path=filedialog.askopenfilename()
    prices = pd.read_csv(file_path)
    prices.columns = ['Date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
    n=file_path.split("/")
    nombre = n[len(n)-1].split(".")[0]
    smooth_prices = prices[price].rolling(window=smoothing).mean().dropna()
    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    local_min = argrelextrema(smooth_prices.values, np.less)[0]
    price_local_max_dt = []
    for i in local_max:
        if (i>window_range) and (i<len(prices)-window_range):
            price_local_max_dt.append(prices.iloc[i-window_range:i+window_range]['close'].idxmax())
    price_local_min_dt = []
    for i in local_min:
        if (i>window_range) and (i<len(prices)-window_range):
            price_local_min_dt.append(prices.iloc[i-window_range:i+window_range]['close'].idxmin())  
    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
    max_min = pd.concat([maxima, minima]).sort_index()
    max_min.index.name = 'date'
    max_min = max_min.reset_index()
    max_min = max_min[~max_min.date.duplicated()]
    max_min = max_min.drop("date",axis = 1)
    max_min["Date"] = pd.to_datetime(max_min["Date"])
    prices["Date"] = pd.to_datetime(prices["Date"])
    d1=max_min["Date"] 
    d2=prices["Date"] 
    p1=max_min[price].to_numpy() 
    p2=prices[price].to_numpy()
    ##HSI
    g=[]
    for i in range(0,p1.shape[0] - 1 - 5):
        if all(p1[i:i+5][2] < p1[i:i+5][[0,1,3,4]]):
            if (all(p1[i:i+5][0] < p1[i:i+5][[1,3]]))and(all((p1[i:i+5][4] < p1[i:i+5][[1,3]]))):
                if (abs(p1[i:i+5][1] - p1[i:i+5][3])<=np.mean([p1[i:i+5][1],p1[i:i+5][3]])*0.2):
                    g.append(np.arange(i,i+5).tolist())
    g = np.unique(np.array(g).reshape(1,np.array(g).shape[0]*np.array(g).shape[1])).tolist()
    g
    ##HS 
    r=[]
    for i in range(0,p1.shape[0] - 1 - 5):
        if all(p1[i:i+5][2] > p1[i:i+5][[0,1,3,4]]):
            if (all(p1[i:i+5][0] > p1[i:i+5][[1,3]]))and(all((p1[i:i+5][4] > p1[i:i+5][[1,3]]))):
                if (abs(p1[i:i+5][1] - p1[i:i+5][3])<=np.mean([p1[i:i+5][1],p1[i:i+5][3]])*0.2):
                    r.append(np.arange(i,i+5).tolist())
    r = np.unique(np.array(r).reshape(1,np.array(r).shape[0]*np.array(r).shape[1])).tolist()            
    if (len(r)>0)and(len(g)>0):  
        p1inv=p1[g]
        d1=pd.Series(d1)
        d1inv=d1.iloc[g]
        p1hs=p1[r]
        d1hs=d1.iloc[r]
        get_ipython().run_line_magic('matplotlib', 'qt')
        plt.style.use("seaborn")
        
        plt.plot_date(d2,p2, marker = "",linestyle ="-",color = "black",linewidth = 0.8)
        plt.plot_date(d1inv,p1inv, marker = "8",color = "red",label = "H&S Inverso",
                     alpha = 0.25)
        plt.plot_date(d1hs,p1hs, marker = "8",color = "#0F0FF7",label = "H&S",
                     alpha = 0.25)
        
        plt.title("Posibles formaciones head and Shoulders en la acción: " + nombre)
        plt.xlabel('Fechas')
        plt.ylabel('Precio de la acción ' + nombre)
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif(len(r)==0)&(len(g)>0):
        print("No existen head and shoulders en su gráfica")
        p1inv=p1[g]
        d1=pd.Series(d1)
        d1inv=d1.iloc[g]
        
        get_ipython().run_line_magic('matplotlib', 'qt')
        plt.style.use("seaborn")
        
        plt.plot_date(d2,p2, marker = "",linestyle ="-",color = "black",linewidth = 0.8)
        plt.plot_date(d1inv,p1inv, marker = "8",color = "red",label = "H&S Inverso",alpha = 0.25)
        
        plt.title("Posibles formaciones head and Shoulders en la acción: " + nombre)
        plt.xlabel('Fechas')
        plt.ylabel('Precio de la acción ' + nombre)
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif(len(r) < 0)&(len(g)==0):
        print("No existen head and shoulders inversos en su gráfica")
        d1=pd.Series(d1)
        p1hs=p1[r]
        d1hs=d1.iloc[r]
        
        get_ipython().run_line_magic('matplotlib', 'qt')
        plt.style.use("seaborn")
        
        plt.plot_date(d2,p2, marker = "",linestyle ="-",color = "black",linewidth = 0.8)
        plt.plot_date(d1hs,p1hs, marker = "h",color = "red",label = "H&S Inverso",alpha = 0.25)
        
        plt.title("Posibles formaciones head and Shoulders en la acción: " + nombre)
        plt.xlabel('Fechas')
        plt.ylabel('Precio de la acción ' + nombre)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    else:
        print("No existen dichos patrones en su gráfica") 


# In[32]:


def PPSR(): 
    root = tk.Tk()
    root.withdraw()
    file_path=filedialog.askopenfilename()
    data = pd.read_csv(file_path)
    m=file_path.split("/")
    nombre = m[len(m)-1].split(".")[0]
    PP = pd.Series((data['High'] + data['Low'] + data['Close']) / 3)  #punto pivote
    R1 = pd.Series(2 * PP - data['Low'])  #primer nivel de resistencia
    S1 = pd.Series(2 * PP - data['High'])   #primer nivel de soporte
    #R2 = pd.Series(PP + data['High'] - data['Low'])  
    #S2 = pd.Series(PP - data['High'] + data['Low'])  
    #R3 = pd.Series(data['High'] + 2 * (PP - data['Low']))  
    #S3 = pd.Series(data['Low'] - 2 * (data['High'] - PP))  
    psr = {'PP':PP, 'R1':R1,'S1':S1}# 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
    PSR = pd.DataFrame(psr)  
    data= data.join(PSR) 
    PSR1 = pd.concat([data["Date"],data["Close"],PSR],axis = 1)
    get_ipython().run_line_magic('matplotlib', 'qt')
    
    stockprices = data.set_index('Date')
    plt.style.use("seaborn")
    
    stockprices[['Close','R1','S1']].plot(figsize=(13,7), linewidth=.5)
                 #'R2','S2','R3','S3'
    plt.grid(True)
    plt.title("Precios de soporte y resistencia para " + nombre)
    plt.axis('tight')
    plt.ylabel('Price')
    
    plt.show()
    pd.set_option('display.max_rows', None)
    return PSR1


# In[16]:


def ROC(n = 12, precio = "Close"):
    root = tk.Tk()
    root.withdraw()
    file_path=filedialog.askopenfilename()
    A = pd.read_csv(file_path)
    m=file_path.split("/")
    nombre = m[len(m)-1].split(".")[0]
    
    p = A[precio].to_numpy()
    date = pd.to_datetime(A["Date"])
    
    ROC =[]
    for j in range(0,p.shape[0] - n + 1):
        ROC.append((p[j:j+n][p[j:j+n].shape[0] - 1]-p[j:j+n][0])*100/p[j:j+n][0])
    SIGN = []
    #date.iloc[n:]
    for i in range(0,len(ROC) - 1):
        SIGN.append(ROC[i] - ROC[i+1])
    
    S = np.sign(np.array(SIGN))
    
    #A  lo mejor lo borro 
    I = []
    s = S[0].copy()
    for i in range(1,len(S)-1): 
        if S[i] == s:
            s = s
        else: 
            I.append(i)
            s = S[i].copy()
    Imas=[]
    Imenos=[]
    
    for i in range(0,len(I)-1):
        if np.sign(ROC[I[i]]) != np.sign(ROC[I[i + 1]]):
            if np.sign(ROC[I[i]]) == 1: 
                Imas.append(I[i])
            else:
                Imenos.append(I[i])
    
    dt1 = pd.to_datetime(pd.Series(date[(n-1):].tolist()).loc[Imenos].tolist())
    dt2 = pd.to_datetime(pd.Series(date[(n-1):].tolist()).loc[Imas].tolist())
    
    get_ipython().run_line_magic('matplotlib', 'qt')
    plt.style.use("seaborn")
    
    fig1, (ax1,ax2) = plt.subplots(nrows = 2,ncols = 1,sharex = True)

    ax1.plot_date(date, p, color='#444444',linestyle='-',marker = "", linewidth = 0.5)
    ax1.set_title('Precio de la acción ' + nombre + " VS ROC")
    ax1.set_ylabel('Precio')    
    
    ax2.plot_date(date.iloc[(n-1):], ROC,marker = "",linestyle = "-",linewidth = 0.5)
    ax2.plot_date(dt2,np.array(ROC)[Imas], marker = "8",color = "red",label = "Inicia cruce (+) a (-)",
                 alpha = 0.5)
    ax2.plot_date(dt1,np.array(ROC)[Imenos], marker = "8",color = "green",label = "Inicia cruce de (-) a (+)" ,
                 alpha =0.5)
    ax2.fill_between(date.iloc[(n-1):],y1 = ROC,y2 = 0,
                     where = (np.array(ROC) >= 0), interpolate = True,color = "green",alpha = 0.25)

    ax2.fill_between(date.iloc[(n-1):],y1 = ROC,y2 = 0,
                     where = (np.array(ROC) < 0),interpolate = True,color ="red",alpha = 0.25)
   
    ax2.set_xlabel('Fechas')
    ax2.set_ylabel('Valor del índice')
    ax2.legend()
    
    plt.tight_layout() 
    plt.show()  
    
    Pr = pd.concat([pd.Series(date.tolist()),pd.Series(p)],axis = 1)
    Pr.columns =["fecha","Precio"]
    Pos = pd.concat([pd.Series(dt2), pd.Series(np.array(ROC)[Imas])],axis = 1)
    Pos.columns =["fecha","ROC"]
    Neg = pd.concat([pd.Series(dt1), pd.Series(np.array(ROC)[Imenos])],axis = 1)
    Neg.columns =["fecha","ROC"]
    Neg["Estrategia"] = ["Inicia momentum al alza"]*Neg.shape[0]
    Pos["Estrategia"] = ["Inicia momentum a la baja"]*Pos.shape[0]
    Neg["BS_flag"] = ["Compra"]*Neg.shape[0]
    Pos["BS_flag"] = ["Vende"]*Pos.shape[0]
    Neg["Activo"] = [nombre]*Neg.shape[0]
    Pos["Activo"] = [nombre]*Pos.shape[0]
    Neg["Dec_algo"] = ["Se recomienda comprar la acción " + nombre + "."]*Neg.shape[0]
    Pos["Dec_algo"] = ["Se recomienda vender la acción " + nombre + "."]*Pos.shape[0]
    Neg["Precio"] = Pr.loc[Pr["fecha"].isin(Neg["fecha"].tolist()),"Precio"].tolist()
    Pos["Precio"] = Pr.loc[Pr["fecha"].isin(Pos["fecha"].tolist()),"Precio"].tolist()
    P = pd.concat([Neg,Pos],axis = 0)
    pd.set_option('display.max_rows', None)
    return P.sort_values("fecha",ascending = False).set_index("fecha")


# In[8]:


def RSI(n = 14,s = 30,precio = "Close"):
    root = tk.Tk()
    root.withdraw()
    file_path=filedialog.askopenfilename()
    A = pd.read_csv(file_path)
    m=file_path.split("/")
    nombre = m[len(m)-1].split(".")[0]
    
    p = A[precio].to_numpy()
    date = pd.to_datetime(A["Date"])
    z = p.shape[0]
    a = p[1:]
    b = p[0:(z-1)]
    mov = a - b
    RS = []
    RSI = []
    
    for i in range(0,mov.shape[0] - n + 1):
        m = mov[i:i+n]
        RS.append(np.sum(m[m>0])/abs(np.sum(m[m<0])))
        RSI.append(100 - 100/(1 + RS[i]))
    get_ipython().run_line_magic('matplotlib', 'qt')
    
    plt.style.use("seaborn")
    
    fig, (ax1,ax2) = plt.subplots(nrows = 2,ncols = 1,sharex = True)

    ax1.plot_date(date, p, color='#444444',linestyle='-',marker = "", linewidth = 0.5)
    ax1.set_title('Precio de la acción ' + nombre + " VS RSI")
    ax1.set_ylabel('Precio')
    
    ax2.plot_date(date.iloc[n:], RSI,linestyle = "-",marker = "",linewidth = 0.5)
    ax2.fill_between(date.iloc[n:],y1 = RSI,y2 = 100-s,
                     where = (np.array(RSI) >= 100-s), interpolate = True,color = "green",alpha = 0.25,
                     label = "Overbought")

    ax2.fill_between(date.iloc[n:],y1 = RSI,y2 = 30,
                     where = (np.array(RSI) <= s),interpolate = True,color ="red",alpha = 0.25,
                     label = "Oversold")
    ax2.axhline(y=100 - s,linestyle = "--",color = "green",linewidth = 0.4)
    ax2.axhline(y=s, linestyle = "--",color = "red",linewidth = 0.4)
    
    ax2.set_xlabel('Fechas')
    ax2.set_ylabel('Valor del índice')
    
    ax2.legend()

    plt.tight_layout()
    
    plt.show()


    A = pd.concat([pd.Series(date[n:].tolist()),pd.Series(RSI)],axis = 1) 
    E = pd.concat([pd.Series(date.tolist()),pd.Series(p)],axis = 1)
    E.columns = ["fecha","Precio"]
    A.columns = ["fecha","RSI"]
    B = A.loc[A["RSI"] >= 70].copy()
    C = A.loc[A["RSI"] <= 30].copy()
    B["Estrategia"] = ["Overbought"]*B.shape[0]
    C["Estrategia"] = ["Oversold"]*C.shape[0]
    B["BS_flag"] = ["Vende"]*B.shape[0]
    C["BS_flag"] = ["Compra"]*C.shape[0]
    B["Activo"] = [nombre]*B.shape[0]
    C["Activo"] = [nombre]*C.shape[0]
    B["Dec_algo"] = ["Se recomienda vender la acción " + nombre + "."]*B.shape[0]
    C["Dec_algo"] = ["Se recomienda comprar la acción " + nombre + "."]*C.shape[0]
    B["Precio"] = E.loc[E["fecha"].isin(B["fecha"].tolist()),"Precio"].tolist()
    C["Precio"] = E.loc[E["fecha"].isin(C["fecha"].tolist()),"Precio"].tolist()
    D = pd.concat([B,C],axis = 0)
    pd.set_option('display.max_rows', None)
    return D.sort_values("fecha",ascending = False).set_index("fecha")


# In[6]:


def SO(n = 14, d = 3, fill = True,precio = "Close"):
    root = tk.Tk()
    root.withdraw()
    file_path=filedialog.askopenfilename()
    A = pd.read_csv(file_path)
    m=file_path.split("/")
    nombre = m[len(m)-1].split(".")[0]
    
    p = A[precio].to_numpy()
    P = A["Close"].to_numpy()
    H = A['High'].to_numpy()
    l = A['Low'].to_numpy()
    date = pd.to_datetime(A["Date"])
    K =[]
    D =[]
    for i in range(0,p.shape[0] - n + 1):
        K.append(100*(P[i:i+n][P[i:i+n].shape[0]-1]-np.min(l[i:i+n]))/(np.max(H[i:i+n]) - np.min(l[i:i+n])))
    for i in range(0,len(K) - d + 1):
        D.append(np.asarray(K[i:i+d]).mean())
    #Len(K) = len(p) - n + 1    date.iloc[(n-1): ]
    #len(D) = len(K) - 2 = len(p)-n-1 date.iloc[(n+1): ]
    
    get_ipython().run_line_magic('matplotlib', 'qt')
    plt.style.use("seaborn")
    
    fig, (ax1,ax2) = plt.subplots(nrows = 2,ncols = 1,sharex = True)

    ax1.plot_date(date, p, color='#444444',linestyle='-',marker = "", 
                  label = nombre,linewidth = 0.5)
    
    ax1.legend()
    ax1.set_title('Precio de la acción ' + nombre + " VS Oscilador Estocástico")
    ax1.set_ylabel('Precio')
    
    ax2.plot_date(date.iloc[(n-1):], K,color ="#6d904f",linestyle = "-",
                  label = "%K",marker = "",linewidth = 0.8)
    ax2.plot_date(date.iloc[(n+1):], D,color ="#fc4f30",linestyle = "-",
                  label = "%D",marker = "",linewidth = 0.8)
    
    if fill:
        ax2.fill_between(date.iloc[(n+1):],y1 = np.array(K)[(d-1): ],y2 = D,
                 where = ( np.array(K)[(d-1): ] > D),
                 interpolate = True,color = "green",alpha = 0.15,
                 label = "Buy signal")
        ax2.fill_between(date.iloc[(n+1):],y1 = np.array(K)[(d-1): ],y2 = D,
                 where = (np.array(K)[(d-1): ] <= D),
                 interpolate = True,color ="red",alpha = 0.15,
                 label = "Sell signal")
        ax2.set_ylim(-30,150)
        ax2.set_xlabel('Fechas')
        ax2.set_ylabel('Valor del índice')
    
        ax2.legend()

        plt.tight_layout()
    
        plt.show()
        
        print("Se recomienda ampliar la ventana de graficos y, "+ 
             "hacer zoom en las partes en las que se desee tener una mejor apreciación")    
    else:
        ax2.set_ylim(-30,130)
        ax2.set_xlabel('Fechas')
        ax2.set_ylabel('Valor del índice')
    
        ax2.legend()

        plt.tight_layout()
    
        plt.show()   
        print("Se recomienda ampliar la ventana de graficos y, "+ 
              "hacer zoom en las partes en las que se desee tener una mejor apreciación")
    #Calculo del data frame a ocupar 
    E = np.array(K[(d-1):]) - np.array(D)
    S = np.sign(E)
    
    #A  lo mejor lo borro 
    I = []
    s = S[0].copy()
    for i in range(1,len(D)-1): 
        if S[i] == s:
            s = s
        else: 
            I.append(i-1)
            s = S[i].copy()
            
    R = pd.Series(pd.Series(date[(n+1):].tolist()).loc[I].tolist()).copy() 
    Pr = pd.concat([pd.Series(date.tolist()),pd.Series(p)],axis = 1)
    EE = pd.concat([R,pd.Series(E[I]),pd.Series(S[I])],axis = 1)
    Pr.columns =["fecha","Precio"]
    EE.columns = ["fecha","Resta %K - %D","Signo"]
    Pos = EE.loc[EE["Signo"] == 1, "fecha":"Resta %K - %D"].copy()
    Neg = EE.loc[EE["Signo"] == -1, "fecha":"Resta %K - %D"].copy()
    Neg["Estrategia"] = ["Indicador Bullish (Golden Cross)"]*Neg.shape[0]
    Pos["Estrategia"] = ["Indicador Bearish (Dead Cross)"]*Pos.shape[0]
    Neg["BS_flag"] = ["Compra"]*Neg.shape[0]
    Pos["BS_flag"] = ["Vende"]*Pos.shape[0]
    Neg["Activo"] = [nombre]*Neg.shape[0]
    Pos["Activo"] = [nombre]*Pos.shape[0]
    Neg["Dec_algo"] = ["Se recomienda comprar la acción " + nombre + "."]*Neg.shape[0]
    Pos["Dec_algo"] = ["Se recomienda vender la acción " + nombre + "."]*Pos.shape[0]
    Neg["Precio"] = Pr.loc[Pr["fecha"].isin(Neg["fecha"].tolist()),"Precio"].tolist()
    Pos["Precio"] = Pr.loc[Pr["fecha"].isin(Pos["fecha"].tolist()),"Precio"].tolist()
    P = pd.concat([Neg,Pos],axis = 0)
    pd.set_option('display.max_rows', None)
    return P.sort_values("fecha",ascending = False).set_index("fecha")


# In[10]:


def Mmov(precio = "Close"):
    root = tk.Tk()
    root.withdraw()
    file_path=filedialog.askopenfilename()
    A = pd.read_csv(file_path)
    m=file_path.split("/")
    nombre = m[len(m)-1].split(".")[0]
    
    rm1= A[precio].rolling(window=30).mean()  #calcular  media movil 30
    rm2= A[precio].rolling(window=60).mean()  #calcular  media movil 60
    A['MA30']=rm1
    A['MA60']=rm2
    f=[]
    D=[]
    p = A[precio]
    P=[]
    date = pd.to_datetime(A["Date"])
    
    
    get_ipython().run_line_magic('matplotlib', 'qt')
    A = A.set_index("Date") 
    plt.style.use("seaborn")
    
    A[[precio,'MA30','MA60']].plot(figsize=(15,5))
    plt.grid(True)
    plt.title('Medias Moviles de ' + nombre)
    plt.axis('tight')
    plt.ylabel('Price')
    
    plt.show()
    
    for i in range(0,len(A)-1):
    
        if (A['MA30'][i] > A['MA60'][i] and A['MA30'][i+1] <= A['MA60'][i+1]):      
             f.append("Death cross") 
    
        elif (A['MA30'][i] < A['MA60'][i] and A['MA30'][i+1] >= A['MA60'][i+1]):
                f.append("Golden cross")
    for i in range(0,len(A)-1):
    
        if (A['MA30'][i] > A['MA60'][i] and A['MA30'][i+1] <= A['MA60'][i+1]):      
             D.append(date[i+1]) 
    
        elif (A['MA30'][i] < A['MA60'][i] and A['MA30'][i+1] >= A['MA60'][i+1]):
                D.append(date[i+1])
                
    for i in range(0,len(A)-1):
    
        if (A['MA30'][i] > A['MA60'][i] and A['MA30'][i+1] <= A['MA60'][i+1]):      
             P.append(p[i+1]) 
    
        elif (A['MA30'][i] < A['MA60'][i] and A['MA30'][i+1] >= A['MA60'][i+1]):
                P.append(p[i+1])
                
        
  #  continue              
    df=pd.DataFrame(f)
    df1= pd.DataFrame(D)

    df2= pd.DataFrame(P)
    MM = pd.concat([df1,df],axis = 1)
    MM.columns =["Fecha","Estrategia"]
    Est = MM.loc[MM["Estrategia"] == "Golden cross", "BS_Flag"]= "Buy"
    Est = MM.loc[MM["Estrategia"] == "Death cross","BS_Flag" ]= "Sell"
    Activo = MM.loc[MM["BS_Flag"] != "", "Activo"]= nombre
    Dec_algo = MM.loc[MM["BS_Flag"] == "Sell", "Dec_algo"]="Se recomienda vender la acción "+nombre
    Dec_algo = MM.loc[MM["BS_Flag"] == "Buy", "Dec_algo"]="Se recomienda comprar la acción "+ nombre
    MM1 = pd.concat([MM,df2],axis = 1)
    MM1.columns =["Fecha","Estrategia","BS_Flag","Activo","Dec_algo","Close"]
    MM1.set_index("Fecha",inplace= True)
    return MM1


# In[1]:


def tops(cota = 0.75, delta = 10,precio = "Close"):
    root = tk.Tk()
    root.withdraw()
    file_path=filedialog.askopenfilename()
    ticker_df = pd.read_csv(file_path)
    m=file_path.split("/")
    nombre = m[len(m)-1].split(".")[0]
    
    x_data = ticker_df.index.tolist()     
    y_data = ticker_df[precio]
    
    
    x = np.linspace(0, max(ticker_df.index.tolist()), max(ticker_df.index.tolist()) + 1)
    
    pol = np.polyfit(x_data, y_data, 17)
    y_pol = np.polyval(pol, x)
    
    data = y_pol
    
    min_max = np.diff(np.sign(np.diff(data))).nonzero()[0] + 1          
    l_min = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1      
    l_max = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1      
    
    print('l_max: ', l_max)
    
    dict_i = dict()
    dict_x = dict()
    
    df_len = len(ticker_df.index)                   
    
    for element in l_max:                           
        l_bound = element - delta                    
        u_bound = element + delta                    
        x_range = range(l_bound, u_bound + 1)        
        dict_x[element] = x_range                    
        y_loc_list = list()
        for x_element in x_range:
            if x_element > 0 and x_element < df_len:                
                y_loc_list.append(ticker_df.Low.iloc[x_element])
        dict_i[element] = y_loc_list                 
    y_delta = 0.12                              
    threshold = np.quantile(ticker_df[precio],cota) 
    y_dict = dict()
    mini = list()
    suspected_bottoms = list()
    for key in dict_i.keys():                      
        mn = sum(dict_i[key])/len(dict_i[key])   
        price_min = min(dict_i[key])    
        mini.append(price_min)                   
        l_y = mn * (1.0 - y_delta)                
        u_y = mn * (1.0 + y_delta)
        y_dict[key] = [l_y, u_y, mn, price_min]
    for key_i in y_dict.keys():    
        for key_j in y_dict.keys():    
            if (key_i != key_j) and (y_dict[key_i][3] > threshold):
                suspected_bottoms.append(key_i) 
    get_ipython().run_line_magic('matplotlib', 'qt')
    plt.style.use("seaborn")            
    plt.figure(figsize=(20, 10), dpi= 120, facecolor='w', edgecolor='k')
    
    plt.plot(x_data, y_data, 'o', markersize=1.5, color='magenta', alpha=0.7)
    
    plt.plot(x_data, ticker_df[precio],linestyle="-", color='green')
    for position in suspected_bottoms:
        plt.axvline(x=position, linestyle='-.', color='r')
    plt.axhline(threshold, linestyle='--', color='b',label= "Quantil " + str(cota*100) + "% de sus datos")    
    for key in dict_x.keys():
        for value in dict_x[key]:
            plt.axvline(x=value, linestyle='-', color = 'lightblue', alpha=0.2)
    
    plt.title("Posibles techos múltiples en la acción " + nombre)
    plt.ylabel("Activo "+ nombre + " vs Ubicación posibles techos absolutos (rojos), relativos (turquesa)")
    plt.xlabel("fecha donde 0=" + ticker_df["Date"].tolist()[0] + " y " + 
               str(max(x_data)) + "= " + ticker_df["Date"].tolist()[ticker_df["Date"].shape[0] - 1])
    plt.legend()        
    plt.show()


# In[28]:


def BB(desv=2,s = 30,precio = "Close",fill = True):
    root = tk.Tk()
    root.withdraw()
    file_path=filedialog.askopenfilename()
    datos1 = pd.read_csv(file_path)
    datos2 = pd.read_csv(file_path)
    m=file_path.split("/")
    Accion = m[len(m)-1].split(".")[0]

    for item in (datos1,datos2):
        item['Media Movil ' +  str(s) +' días'] = item[precio].rolling(window=s).mean()

        # set .std(ddof=0) for population std instead of sample
        item['Desv St últimos ' +  str(s) +' días'] = item[precio].rolling(window=s).std(ddof=0) 

        item['Banda Superior'] = item['Media Movil ' +  str(s) +' días'] + (item['Desv St últimos ' + str(s) +' días'] * desv)
        item['Banda Inferior'] = item['Media Movil ' +  str(s) +' días'] - (item['Desv St últimos ' + str(s) +' días'] * desv) 
    
    a=datos1.loc[~(pd.isnull(datos1['Banda Inferior'])),['Date',precio,'Media Movil ' +  str(s) +' días','Banda Inferior','Banda Superior']].copy()
    S= a['Banda Superior'].to_numpy()
    i= a['Banda Inferior'].to_numpy()
    mm= a['Media Movil ' +  str(s) +' días'].to_numpy()
    p= a[precio].to_numpy()
    date= pd.to_datetime(a['Date'].tolist())

    get_ipython().run_line_magic('matplotlib', 'qt')
    plt.style.use("seaborn")

    plt.plot_date(date,S,linestyle = "--",marker = "", color =  "#fc4f30",linewidth = 1,label="Banda Superior") 
    plt.plot_date(date,i,linestyle = "--",marker = "", color =  "#fc4f30",linewidth = 1,label="Banda Inferior")
    plt.plot_date(date,mm,linestyle = "--",marker = "", color = "green",linewidth = 0.5,label= 'Media Movil a ' +  str(s) +' días')
    plt.plot_date(date,p,linestyle = "-",marker = "",linewidth = 1,label= precio)

    if fill: 
        plt.fill_between(date,y1 = p,y2 = S,
                         where = (p >= S), interpolate = True,color = "green",alpha = 0.25,
                         label = "Vende")

        plt.fill_between(date,y1 = p,y2 = i,
                         where = (p <= i),interpolate = True,color ="red",alpha = 0.25,
                         label = "Compra")

        plt.title("Bandas de Bollinger para " + Accion)
        plt.xlabel('Fechas')
        plt.ylabel('Valor del índice')  
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        plt.title("Bandas de Bollinger para " + Accion)
        plt.xlabel('Fechas')
        plt.ylabel('Valor del índice')  
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    B=a.loc[a[precio].to_numpy()<=a['Banda Inferior'].to_numpy(),['Date',precio,'Media Movil ' +  str(s) +' días','Banda Inferior','Banda Superior']].copy()
    b=a.loc[a[precio].to_numpy()>=a['Banda Superior'].to_numpy(),['Date',precio,'Media Movil ' +  str(s) +' días','Banda Inferior','Banda Superior']].copy()

    B['Estrategia']=['La acción de ' + Accion + ' por debajo de la banda inferior de Bollinger']*B.shape[0]
    b['Estrategia']=['La acción de ' + Accion + ' rebasó la banda superior de Bollinger']*b.shape[0]

    B['BS_Flag']=['Compra']*B.shape[0]
    b['BS_Flag']=['Venta']*b.shape[0]

    B['Activo']=[Accion]*B.shape[0]
    b['Activo']=[Accion]*b.shape[0]

    B['Dec_Algo']=['Se recomienda compra la acción ' + Accion]*B.shape[0]
    b['Dec_Algo']=['Se recomienda vender la acción ' + Accion]*b.shape[0]

    B['Close']=B[precio]
    b['Close']=b[precio]

    del B[precio]
    del b[precio]

    pd.set_option('display.max_rows', None)
    return pd.concat([B,b],axis = 0).sort_values('Date',ascending=False).set_index("Date")  


# In[ ]:




