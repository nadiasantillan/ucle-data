import numpy as np

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import pandas as pd
from pmdarima.arima import CHTest


class Cubo:
    
    def __init__(self, archivo_cubo, archivo_tiempo):
        with open(archivo_cubo, 'r') as f:
            a = np.loadtxt(archivo_cubo, delimiter=",")
            self.datos = a.reshape(-1, 3, 3).transpose(1, 2, 0)
        with open(archivo_tiempo, 'r') as f:
            self.tiempo = np.loadtxt(archivo_tiempo, delimiter=",")

class Tendencia:
    
    def __init__(self, tiempo, serie, nombre):
        self.tiempo = tiempo
        self.serie = serie
        self.nombre = nombre
        
    def tendencia(self):
        return seasonal_decompose(self.serie, model="additive", period=23)
    
    def grafico(self):
        descomposicion = self.tendencia()
        
        plt.figure(figsize=(15,6))
        plt.plot(self.tiempo, self.serie, label="Datos originales")
        plt.plot(self.tiempo, descomposicion.trend, label="Tendencia")
        plt.title(f"Tendencia {self.nombre}")
        plt.legend()
        plt.show()
    
class Autocorrelacion:
    def __init__(self, tiempo, serie, nombre):
        self.tiempo = tiempo
        self.serie = serie
        self.nombre = nombre    
        
    def acf(self, lags=100):
        acf_v, _ = acf(self.serie, nlags=lags, alpha=0.05)
        return acf_v
        
    def acf_significativos(self, lags=100):
        acf_v, acf_ci = acf(self.serie, nlags=lags, alpha=0.05)
        return [(i, round(acf_v[i], 3)) for i in range(1, lags+1)
            if acf_ci[i, 0] > 0 or acf_ci[i, 1] < 0]

    def pacf_significativos(self, lags=100):        
        pacf_v, pacf_ci = pacf(self.serie, nlags=lags, alpha=0.05, method="ywm")

        return [(i, round(pacf_v[i], 3)) for i in range(1, lags+1)
            if pacf_ci[i, 0] > 0 or pacf_ci[i, 1] < 0]
        
        
    def grafico(self, lags=100):
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        plot_acf(self.serie, lags=lags, ax=ax[0], title=f"ACF {self.nombre}")
        plot_pacf(self.serie, lags=lags, ax=ax[1], method="ywm", title=f"PACF {self.nombre}")
        plt.show()

class Serie:
    def __init__(self, tiempo, serie1, serie2, nombre, nombre1, nombre2):
        self.tiempo = tiempo
        self.serie1 = serie1
        self.serie2 = serie2
        self.nombre = nombre       
        self.nombre1 = nombre1
        self.nombre2 = nombre2
        
    def grafico(self):
        fig, ax1 = plt.subplots(figsize=(15,6))

        color_1 = "tab:green"
        ax1.set_xlabel("Tiempo")
        ax1.set_ylabel(self.nombre1, color=color_1)
        ax1.plot(self.tiempo, self.serie1, color=color_1)
        ax1.tick_params(axis="y", labelcolor=color_1)


        ax2 = ax1.twinx()

        color_2 = "tab:orange"
    
        ax2.set_ylabel(self.nombre2, color=color_2)
        ax2.plot(self.tiempo, self.serie2, color=color_2)
        ax2.tick_params(axis="y", labelcolor=color_2)

        fig.tight_layout()
        plt.title(f"Serie {self.nombre}")
        plt.legend()        
        plt.show()
        
class Estacionaridad:
    def __init__(self, serie, nombre):
        self.serie = serie
        self.nombre = nombre    
        
    def adf_corto_plazo(self):
        resultado_constante = adfuller(self.serie, regression="c")
        resultado_tendencia = adfuller(self.serie, regression="ct")
        resultado_cuadratica = adfuller(self.serie, regression="ctt")
        tabla = pd.DataFrame({
            "Regresión": ["Constante", "Tendencia", "Cuadrática"],
            "Estadístico ADF": [resultado_constante[0], resultado_tendencia[0], resultado_cuadratica[0]],
            "Valor p": [resultado_constante[1], resultado_tendencia[1], resultado_cuadratica[1]],
            "Número de lags": [resultado_constante[2], resultado_tendencia[2], resultado_cuadratica[2]],
        })
        return tabla    
    
    def adf_largo_plazo(self, s=23):
        serie_dif_s = self.serie[s:] - self.serie[:-s]

        resultado_constante = adfuller(serie_dif_s, regression="c")
        resultado_tendencia = adfuller(serie_dif_s, regression="ct")
        resultado_cuadratica = adfuller(serie_dif_s, regression="ctt")
        tabla = pd.DataFrame({
            "Regresión": ["Constante", "Tendencia", "Cuadrática"],
            "Estadístico ADF": [resultado_constante[0], resultado_tendencia[0], resultado_cuadratica[0]],
            "Valor p": [resultado_constante[1], resultado_tendencia[1], resultado_cuadratica[1]],
            "Número de lags": [resultado_constante[2], resultado_tendencia[2], resultado_cuadratica[2]],
        })
        return tabla      
    
    def confirmacion_raiz_estacional(self, s=23) :
        return CHTest(m=s).estimate_seasonal_differencing_term(self.serie)
    
class AjusteSARIMA:
    def __init__(self, serie, ajuste, nombre):
        self.serie = serie
        self.ajuste = ajuste
        self.nombre = nombre
    
    def graficos_residuos(self, lags= 100):
        
        pred = self.ajuste.get_prediction()

        yfit = np.asarray(pred.predicted_mean)
        R = self.serie - yfit

        _, ax = plt.subplots(1, 2, figsize=(16, 3))
        plot_acf(R, lags=lags, ax=ax[0], title=f"ACF {self.nombre}")
        plot_pacf(R, lags=lags, ax=ax[1], method="ywm", title=f"PACF {self.nombre}")
        plt.show()

        acf_v, acf_ci = acf(R, nlags=lags, alpha=0.025)
        pacf_v, pacf_ci = pacf(R, nlags=lags, alpha=0.025, method="ywm")

        return {"acf significativos": [(i, round(acf_v[i], 3)) for i in range(1, lags+1)
            if acf_ci[i, 0] > 0 or acf_ci[i, 1] < 0],
            "pacf_significativos":[(i, round(pacf_v[i], 3)) for i in range(1, lags+1)
            if pacf_ci[i, 0] > 0 or pacf_ci[i, 1] < 0]}        
        
class AjusteSARIMAX:    
    def __init__(self, serie, ajuste, nombre):
        self.serie = serie
        self.ajuste = ajuste
        self.nombre = nombre
    
    def graficos_residuos(self, lags= 100):
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        plot_acf(self.ajuste.resid[5:], lags=lags, ax=axes[0])
        axes[0].set_title("ACF de los Residuos")
        plot_pacf(self.ajuste.resid[5:], lags=lags, ax=axes[1])
        axes[1].set_title("PACF de los Residuos")
        plt.tight_layout()
        plt.show()       