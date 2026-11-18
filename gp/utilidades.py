import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, ConstantKernel,
                                              WhiteKernel, ExpSineSquared, RationalQuadratic)
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

_AMPLITUD = lambda: ConstantKernel(1., (1e-5, 1))
_RUIDO = lambda: WhiteKernel(.1, (1e-5, 0.25))
_KERNELS = {
        'RBF': _AMPLITUD()*RBF(.3, (.02, 20.)) + _RUIDO(),
        'Matérn 3/2': _AMPLITUD()*Matern(.3, (.02, 20.), nu=1.5) + _RUIDO(),
        'RationalQuadratic': _AMPLITUD()*RationalQuadratic(
           .3, 1.,
           length_scale_bounds=(.02, 20.),
           alpha_bounds=(.01, 100.)
        ) + _RUIDO(),
        'Periódico Puro': (
            _AMPLITUD()*ExpSineSquared(
                length_scale=1., periodicity=1.,
                length_scale_bounds=(.05, 20.),
                periodicity_bounds=(.5, 2.0))
            )
            + _RUIDO(),

        'Periódico + Matérn': (
            _AMPLITUD()*ExpSineSquared(
                length_scale=1., periodicity=1.,
                length_scale_bounds=(.05, 20.),
                periodicity_bounds=(.5, 2.0)
            )
            + _AMPLITUD()*Matern(.3, (.02, 10.), nu=1.5)
            + _RUIDO()
        )
    }

class Mascara:
    def __init__(self, data, valor_valido = False, valor_invalido = True):
        self.data = data
        self.valor_valido = valor_valido
        self.valor_invalido = valor_invalido

    def dibujo(self, ancho=10):
        values = [self.valor_valido, self.valor_invalido]

        plt.figure(figsize=(ancho, ancho*self.data.shape[0]/self.data.shape[1]))
        im = plt.imshow(self.data, cmap='gray_r', vmin=0, vmax=1, origin='lower', alpha=0.5)
        colors = [ im.cmap(im.norm(value), alpha=0.5) for value in values ]
        patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
        plt.title('Máscara de píxeles')
        plt.legend(labels=["Píxel válido", "Píxel inválido"], handles=patches, bbox_to_anchor=(.80, .96), loc='upper left', borderaxespad=0.)
        plt.show()
    
    def enmascara(self, data):
        return np.array(
            [data[i, j] if self.data[i, j] == self.valor_valido else np.repeat(np.nan, data.shape[2]) 
             for i in range(self.data.shape[0]) 
             for j in range(self.data.shape[1])]).reshape(data.shape)

class VegetacionMedia:
    def __init__(self, evi):
        n = np.sum(np.isfinite(evi), axis=2)
        self.mapa = np.divide(
            np.nansum(evi, axis=2),
            n,
            out=np.full(n.shape, np.nan),
            where=n > 0
        )
        
    def dibujo(self, ancho=10, i = None, j = None):
        plt.figure(figsize=(ancho, ancho*self.mapa.shape[0]/self.mapa.shape[1]))
        im = plt.imshow(np.ma.masked_invalid(self.mapa), 
                        cmap='gist_earth_r',     
                        interpolation="nearest",
                        aspect="equal",
                        origin='lower')
        if i is not None and j is not None:
            plt.scatter(j, i, s=100, facecolors="none", edgecolors='red', linewidth=2, label=f'Píxel ({i}, {j})')
        plt.legend(loc='upper right')
        plt.xlabel('Columna (j)')
        plt.ylabel('Fila (i)')
        plt.title('Vegetación media')
        plt.colorbar(im)
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
    def __init__(self, valores, tiempo, nombre=None):
        self.valores = valores
        self.tiempo = tiempo
        self.nombre = nombre
        
    def dibujo(self, ancho=10):
        plt.figure(figsize=(ancho, ancho/2))
        plt.plot(self.tiempo, self.valores, color='tab:green', label=self.nombre)
        plt.legend(loc='upper right')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor')
        plt.title('Serie temporal')
        plt.grid()
        plt.show()
        
class Ajuste:
    def __init__(self, serie, tiempo, n_prueba):
        self.b_entrenamiento = serie[:-n_prueba]
        self.b_prueba = serie[-n_prueba:]
        self.t_entrenamiento = tiempo[:-n_prueba]
        self.t_prueba = tiempo[-n_prueba:]

    def ajuste(self, kernel, t_entrenamiento, y_entrenamiento, restarts=0):
        modelo = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8, normalize_y=False, n_restarts_optimizer=restarts, random_state=42)
        print(f'Ajustando modelo con kernel: {kernel}')
        modelo.fit(t_entrenamiento.reshape(-1, 1)-t_entrenamiento[0], 
                   (y_entrenamiento-y_entrenamiento.mean())/y_entrenamiento.std())
        return modelo
       
    def estimacion(self, gp, t_entrenamiento, y_entrenamiento, t_prueba):        
        t_entrenamiento_2_d = t_entrenamiento.reshape(-1, 1)-t_entrenamiento[0]
        t_prueba_2_d = t_prueba.reshape(-1, 1)-t_entrenamiento[0]
        
        mu_train_norm, sd_train_norm = gp.predict(t_entrenamiento_2_d, return_std=True)
        mu_test_norm, sd_test_norm = gp.predict(t_prueba_2_d, return_std=True)

        media_train, escala_train = y_entrenamiento.mean(), y_entrenamiento.std()

        mu_train = media_train + escala_train * mu_train_norm
        sd_train = escala_train * sd_train_norm
        mu_test = media_train + escala_train * mu_test_norm
        sd_test = escala_train * sd_test_norm

        return mu_train, sd_train, mu_test, sd_test

    def dibujo(self, gp, ax = plt, title=None):
        mu_train, sd_train, mu_test, sd_test = self.estimacion(gp, self.t_entrenamiento, self.b_entrenamiento, self.t_prueba)
        ax.plot(self.t_entrenamiento, self.b_entrenamiento, '.', ms=3, color='tab:blue', label='Entrenamiento')
        ax.plot(self.t_prueba, self.b_prueba, 'o-', ms=3, color='tab:orange', label='Observaciones de prueba')
        ax.axvline(
            self.t_entrenamiento[-1],
            color='gray',
            ls='--',
            label='Fin del entrenamiento'
        )
        ax.plot(self.t_entrenamiento, mu_train, '-', lw=1.8, color='tab:green', label='GP ajustado')
        ax.plot(self.t_prueba, mu_test, '-', lw=1.8, color='tab:red', label='Pronóstico GP')
        ax.fill_between(
            self.t_entrenamiento,
            mu_train - 1.96 * sd_train,
            mu_train + 1.96 * sd_train,
            color='tab:green',
            alpha=0.2,
            label='IC 95 % del ajuste'
        )
        ax.fill_between(
            self.t_prueba,
            mu_test - 1.96 * sd_test,
            mu_test + 1.96 * sd_test,
            color='tab:red',
            alpha=0.2,
            label='Intervalo predictivo 95 %'
        )

        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2)
        if title:
            ax.set_title(title)
            
    def evaluar_modelos(self):
        resultados = {}
        for nombre, kernel in _KERNELS.items():
            resultados[nombre] = self._modelo(nombre, kernel)
        return resultados
        
    def validacion_kernels(self, ventana=160, horizonte=10):
        b_serie = np.concatenate([self.b_entrenamiento, self.b_prueba])
        t_serie = np.concatenate([self.t_entrenamiento, self.t_prueba])
        origenes = np.linspace(ventana, len(b_serie)-horizonte, 4, dtype=int)[1:]
        registros = []
        for kernel_nombre, kernel in _KERNELS.items():
            for origen in origenes:
                tt = t_serie[origen-ventana:origen]
                yy = b_serie[origen-ventana:origen]
                tf = t_serie[origen:origen+horizonte]
                yf = b_serie[origen:origen+horizonte]
                
                modelo = self.ajuste(kernel, tt, yy, restarts=0)
                _, _, pred, sd =self.estimacion(modelo, tt, yy, tf)

                for h, obs, mu_h, sd_h in zip(range(1, horizonte+1), yf, pred, sd):
                    registros.append((kernel_nombre, h, (obs-mu_h)**2, obs-mu_h, abs(obs-mu_h) <= 1.96*sd_h))

        return pd.DataFrame(registros, columns=['kernel', 'horizonte', 'error^2', 'error', 'cobertura'])
        
        
    def _modelo(self, nombre, kernel):
        modelo = self.ajuste(kernel, self.t_entrenamiento, self.b_entrenamiento, restarts=1)
        
        _, _, mu_prueba, sd_prueba = self.estimacion(modelo, 
                                                                                   self.t_entrenamiento, 
                                                                                   self.b_entrenamiento, 
                                                                                   self.t_prueba)
        rmse = np.sqrt(np.mean((self.b_prueba - mu_prueba)**2))
        cobertura = np.mean(np.abs(self.b_prueba - mu_prueba) <= 1.96 * sd_prueba)

        # Log-verosimilitud marginal maximizada
        logML = modelo.log_marginal_likelihood_value_

        # Número de hiperparámetros libres
        k = modelo.kernel_.n_dims

        # Criterio de información de Akaike
        AIC = 2*k - 2*logML

        print(
            f'{nombre:21s} '
            f'RMSE={rmse:.5f}  '
            f'cobertura={cobertura:.1%}  '
            f'logML={logML:.2f}  '
            f'k={k}  '
            f'AIC={AIC:.2f}'
        )

        print('  ', modelo.kernel_)

        return (modelo, mu_prueba, sd_prueba, rmse, cobertura, AIC)

    
    def _x_entrenamiento_columna(self):
        return self.t_entrenamiento.reshape(-1, 1) - self.t_entrenamiento[0]
    
    def _x_prueba_columna(self):
        return self.t_prueba.reshape(-1, 1) - self.t_entrenamiento[0]
    
    def _y_media_entrenamiento(self):
        return self.b_entrenamiento.mean()
    
    def _y_escala_entrenamiento(self):
        return self.b_entrenamiento.std()
    
    def _y_entrenamiento_normalizado(self):
        return (self.b_entrenamiento - self._y_media_entrenamiento()) / self._y_escala_entrenamiento()
    
    def _y_prueba_normalizado(self):
        return (self.b_prueba - self._y_media_entrenamiento()) / self._y_escala_entrenamiento()
    

