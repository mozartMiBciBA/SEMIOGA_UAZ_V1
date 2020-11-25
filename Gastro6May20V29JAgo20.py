#!/usr/bin/env python3 # Esta linea no se debe de borrar
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import scipy
from scipy import stats
from scipy.stats import kurtosis
from PyQt5 import QtWidgets

from principal import Ui_MainWindow
from segunda import Ui_segunda
from tres import Ui_tres
from cuatro import Ui_cuatro
from cinco import Ui_cinco
from ayuda import Ui_ayuda


#class VentanaPrincipal(QMainWindow):
class VentanaPrincipal(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(VentanaPrincipal, self).__init__()
        QtWidgets.QMainWindow.__init__(self)
        #loadUi('principal.ui', self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.boton6_4.clicked.connect(self.abrirAyuda)
        self.boton1_0.clicked.connect(self.abrirVen2)
        self.boton3_0.clicked.connect(self.abrirVen3)
        self.boton4_0.clicked.connect(self.abrirVen4)
        self.boton5_0.clicked.connect(self.abrirVen5)


    def abrirAyuda(self):
        self.hide()
        otraventana= ventanaAyu(self)
        otraventana.show()

    def abrirVen2(self):
        self.hide()
        otraventana= ventanaSeg(self)
        otraventana.show()

    def abrirVen3(self):
        #self.hide()
        otraventana= ventanaTres(self)
        otraventana.show()

    def abrirVen4(self):
        #self.hide()
        otraventana= ventanaCua(self)
        otraventana.show()

    def abrirVen5(self):
        #self.hide()
        otraventana= ventanaCin(self)
        otraventana.show()


#class ventanaSeg(QMainWindow):
class ventanaSeg(QtWidgets.QMainWindow, Ui_segunda):
    def __init__(self, parent=None):
        super(ventanaSeg, self).__init__(parent)
        #QtWidgets.QMainWindow.__init__(self)
        #loadUi('segunda.ui', self)
        Ui_segunda.__init__(self)
        self.setupUi(self)
        self.boton6_4.clicked.connect(self.abrirVentanaPrincipal)

    def conversion(self):
        a=0
       
    def abrirVentanaPrincipal(self):
        self.parent().show()
        self.close()

class ventanaTres(QtWidgets.QMainWindow, Ui_tres):
#class ventanaTres(QMainWindow):
    def __init__(self, parent=None):
        super(ventanaTres, self).__init__(parent)
        #loadUi('tres.ui', self)
        Ui_tres.__init__(self)
        self.setupUi(self)
        #Botón regresar a ventana principal
        self.boton6_4.clicked.connect(self.abrirVentanaPrincipal)
        #
        # Botones
        # Aquí va el botón ctrl
        self.boton1.clicked.connect(self.getCSV1)
        self.boton1_1.clicked.connect(self.plotCSV1time)
        self.boton1_2.clicked.connect(self.plotmeanPsdC1)

        # Aquí va el botón Palm
        self.boton3.clicked.connect(self.getCSV2)
        self.boton3_1.clicked.connect(self.plotCSV2time)
        self.boton3_2.clicked.connect(self.plotmeanPsdC2)

        # Aquí va el botón Pesc
        self.boton4.clicked.connect(self.getCSV3)
        self.boton4_1.clicked.connect(self.plotCSV3time)
        self.boton4_2.clicked.connect(self.plotmeanPSDC3)

        # Botón comparación 3 señales ch1
        self.boton2.clicked.connect(self.plotPSDch1)

        # Botón comparación 3 señales ch2
        self.boton2_1.clicked.connect(self.plotPSDch2)

        # Boton reiniciar captura archivos ctr
        self.boton1_3.clicked.connect(self.resC1)
        # Boton reiniciar captura archivos Palma
        self.boton3_3.clicked.connect(self.resC2)
        # Boton reiniciar captura archivos Pescado
        self.boton4_3.clicked.connect(self.resC3)

        # Matrices archivo general de caracteristicas
        self.caracteristicasSenales = np.empty((0, 127))

        # Matrices archivo CSV Grupo1 Ctr
        self.C1Ch1 = np.empty((0, 129))
        self.C1Ch2 = np.empty((0, 129))

        # Matrices archivo CSV Grupo2 Palm
        self.C2Ch1 = np.empty((0, 129))
        self.C2Ch2 = np.empty((0, 129))

        # Matrices archivo CSV Grupo3
        self.C3Ch1 = np.empty((0, 129))
        self.C3Ch2 = np.empty((0, 129))
        #

    #### Aquí van las funciones

    def getCSV1(self):
        resultado_st = "getCSV1" + "\n"
        self.resultado.setText(resultado_st)

        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #
            def entropia(X):
                """Devuelve el valor de entropia de una muestra de datos"""
                probs = [np.mean(X == valor) for valor in set(X)]
                return round(np.sum(-p * np.log2(p) for p in probs), 3)
            #
            def curtoSis(y):
                curtosisY = kurtosis(y)
                return curtosisY
            #
            def estaBasica(y):
                estDesy = np.std(y, dtype=np.float64)
                meany = np.mean(y, dtype=np.float64)
                vary = np.var(y, dtype=np.float)
                medianY = np.median(y)
                Q1 = np.quantile(np.sort(y), 0.25)
                Q3 = np.quantile(np.sort(y), 0.75)
                return vary, meany, estDesy, medianY, Q1, Q3
            #
            def maximo(x):
                max = np.amax(x)
                inmax = np.argmax(x)
                return max, inmax
            #
            def pearsonCorr(x, y):
                corrpear = scipy.stats.pearsonr(x, y)
                """
                        The p-value roughly indicates the probability of an uncorrelated system
                        producing datasets that have a Pearson correlation at least as extreme
                        as the one computed from these datasets. The p-values are not entirely
                        reliable but are probably reasonable for datasets larger than 500 or so.
                        Parameters
                """

                return corrpear
            #
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values
            # def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
            #
            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label='PaBaja')
                # axs[0].set_ylim(yinf, ysup)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)

                axs[1].plot(x_valueDowSamy1, y1filPaBaDoSam1, linestyle='-', color='g', label='PaBaja')
                # axs[1].set_ylim(yinf, ysup)
                axs[1].set_xlim(x1, x2)
                axs[1].set_ylabel(y1nom)
                axs[1].grid(True)

                axs[2].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label='PaBaja')
                # axs[2].set_ylim(yinf, ysup)
                axs[2].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[2].set_ylabel(y2nom)
                axs[2].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):

                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                """
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")
                """
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 27-04-20
                """
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Est. y0 Hz", y1nom="Est. y0 Hz", y2nom="Ciego y1 Hz s/Ruido")
                """
                ######################################################
                # y0Guardar = y0filPaBaDoSam[0:longMenorHz]
                # y1Guardar = y1filPaBaDoSam[0:longMenorHz]
                # y0y1HzStack = np.stack((y0Guardar, y1Guardar), axis=-1)
                # guardarArchivo(y0y1HzStack, nombreExCa)



                # longMenor = np.min(np.array([longy0, longy1, longy2]))
                longMenor = np.min(np.array([longy0, longy1, longy2, longy0Hzsr, longy1Hzsr]))
                #print("longMenor: ", longMenor)
                # y0Guardar = y0filPaBaDoSam[0:longMenor]
                # y1Guardar = y1filPaBaDoSam[0:longMenor]
                # y2Guardar = y2filPaBaDoSam[0:longMenor]
                # y0y1y2Stack = np.stack((y0Guardar, y1Guardar, y2Guardar), axis=-1)
                # guardarArchivo(y0y1y2Stack, nombreExCa)

                ventanas0 = longMenor // 130
                #print("ventanas")
                #print(ventanas0)

                inicioDs = 0
                finDs = longMenor
                #print("len(y0filPaBaDoSam): ", len(y0filPaBaDoSam))
                #print("finDs", finDs)
                # Gráfica señales
                # plotDownSamp(y0filPaBaDoSam, y1filPaBaDoSam, y2filPaBaDoSam, yinf, ysup, inicioDs, finDs, nombreExCa,
                #            y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")


                #print("########### ExtractFeature ###############")
                # caracteristicasSenal = np.empty((0, 83))
                caracteristicasSenal = np.empty((0, 127))
                carFreSenalCh1 = np.empty((0, 129))
                carFreSenalCh2 = np.empty((0, 129))

                # for i in range(8):
                for i in range(ventanas0 - 1):
                    #print("i: ", i)
                    dt = 1
                    x_valueDowSam = np.arange(0, longMenor, dt)
                    inicioDs = 130 * i
                    finDs = inicioDs + 260

                    # Ventana de 2 minuto 10 segundos señal down sampled
                    # fs=2 hz         # 2 muestras por segundo
                    # para 60 segundos tengo 120 muestras

                    y0filPaBaDoSamTF = y0filPaBaDoSam[inicioDs:finDs]
                    y1filPaBaDoSamTF = y1filPaBaDoSam[inicioDs:finDs]
                    y2filPaBaDoSamTF = y2filPaBaDoSam[inicioDs:finDs]
                    y0HzfilPaBaDoSamTF = y0HzfilPaBaDoSam[inicioDs:finDs]
                    y1HzfilPaBaDoSamTF = y1HzfilPaBaDoSam[inicioDs:finDs]

                    # Se llama función plotDownSamp()
                    # plotDownSamp(y0filPaBaDoSam, y0filPaBaDoSam, y0filPaBaDoSamTF, yinf, ysup, inicioDs, finDs,
                    #            nombreExCa, y0nom="Estomago", y1nom="Estomago", y2nom="Estomago")

                    # plotDownSamp(y1filPaBaDoSam, y1filPaBaDoSam, y1filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #            y0nom="Ciego", y1nom="Ciego", y2nom="Ciego")

                    # plotDownSamp(y2filPaBaDoSam, y2filPaBaDoSam, y2filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #             y0nom="Ileum", y1nom="Ileum", y2nom="Ileum")

                    kurTiemy0 = curtoSis(y0filPaBaDoSamTF)
                    kurTiemy1 = curtoSis(y1filPaBaDoSamTF)
                    kurTiemy2 = curtoSis(y2filPaBaDoSamTF)

                    entTemy0 = entropia(y0filPaBaDoSamTF)
                    entTemy1 = entropia(y1filPaBaDoSamTF)
                    entTemy2 = entropia(y2filPaBaDoSamTF)

                    varY0Tf, meanY0Tf, stdY0Tf, medianY0Tf, Q1Y0Tf, Q3Y0Tf = estaBasica(y0filPaBaDoSamTF)
                    varY1Tf, meanY1Tf, stdY1Tf, medianY1Tf, Q1Y1Tf, Q3Y1Tf = estaBasica(y1filPaBaDoSamTF)
                    varY2Tf, meanY2Tf, stdY2Tf, medianY2Tf, Q1Y2Tf, Q3Y2Tf = estaBasica(y2filPaBaDoSamTF)

                    f_valuesch1, ch1Psd_values = get_psd_values(y0filPaBaDoSamTF, T, N, f_s)
                    f_valuesch2, ch2Psd_values = get_psd_values(y1filPaBaDoSamTF, T, N, f_s)
                    f_valuesch3, ch3Psd_values = get_psd_values(y2filPaBaDoSamTF, T, N, f_s)
                    # y1HzfilPaBaDoSamTF
                    f_valuesch1Hz, ch1HzPsd_values = get_psd_values(y0HzfilPaBaDoSamTF, T, N, f_s)
                    f_valuesch2Hz, ch2HzPsd_values = get_psd_values(y1HzfilPaBaDoSamTF, T, N, f_s)

                    #print("len(ch2HzPsd_values)")
                    #print(len(ch2HzPsd_values))

                    # plt.plot(f_valuesch1, ch1Psd_values, linestyle='-', color='black', label='Estomago')
                    # plt.plot(f_valuesch2, ch2Psd_values, linestyle='-', color='g', label='Ciego')
                    # plt.plot(f_valuesch3, ch3Psd_values, linestyle='-', color='r', label='Ileon')
                    # plt.plot(f_valuesch1Hz, ch1HzPsd_values, linestyle='-', color='blue', label='Ch1')
                    # plt.plot(f_valuesch2Hz, ch2HzPsd_values, linestyle='-', color='orange', label='Ch2')
                    # plt.xlabel('Frequencia [Hz]', fontsize=16)
                    # plt.ylabel('PSD [uV**2 / Hz]', fontsize=16)
                    # plt.title("Espectro Ciego, Estomago e Ileum", fontsize=16)
                    # plt.show()

                    r = pearsonCorr(ch1HzPsd_values, ch2HzPsd_values)

                    # Estomago
                    ch1PsdValVentana = ch1Psd_values[7:26]  # 12 datos
                    varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Ch1Tf, Q3Ch1Tf = estaBasica(ch1PsdValVentana)

                    # Ciego
                    # ch2PsdValVentana = ch2Psd_values[42:68] 26 datos
                    ch2PsdValVentana = ch2Psd_values[0:7]  # 8 datos
                    varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Ch2Tf, Q3Ch2Tf = estaBasica(ch2PsdValVentana)

                    # Ileum
                    ch3PsdValVentana = ch3Psd_values[40:84]
                    varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Ch3Tf, Q3Ch3Tf = estaBasica(ch3PsdValVentana)

                    kurFreCh1 = curtoSis(ch1PsdValVentana)
                    kurFreCh2 = curtoSis(ch2PsdValVentana)
                    kurFreCh3 = curtoSis(ch3PsdValVentana)
                    entFreCh1 = entropia(ch1PsdValVentana)
                    entFreCh2 = entropia(ch2PsdValVentana)
                    entFreCh3 = entropia(ch3PsdValVentana)

                    maxch2, inmaxch2 = maximo(ch2PsdValVentana)
                    #print("Ciego: máximo valor e indice", maxch2, inmaxch2)

                    maxch1, inmaxch1 = maximo(ch1PsdValVentana)
                    #print("Estomago: máximo valor e indice", maxch1, inmaxch1)

                    maxch3, inmaxch3 = maximo(ch3PsdValVentana)
                    #print("Ileum: máximo valor e indice", maxch3, inmaxch3)

                    caracteristicaVentana = np.hstack(
                        [ch1PsdValVentana, kurTiemy0, kurFreCh1, entTemy0, entFreCh1, maxch1, inmaxch1, varY0Tf,
                         meanY0Tf, stdY0Tf, medianY0Tf, varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Y0Tf, Q3Y0Tf,
                         Q1Ch1Tf, Q3Ch1Tf,
                         ch2PsdValVentana, kurTiemy1, kurFreCh2, entTemy1, entFreCh2, maxch2, inmaxch2, varY1Tf,
                         meanY1Tf, stdY1Tf, medianY1Tf, varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Y1Tf, Q3Y1Tf,
                         Q1Ch2Tf, Q3Ch2Tf,
                         ch3PsdValVentana, kurTiemy2, kurFreCh3, entTemy2, entFreCh3, maxch3, inmaxch3, varY2Tf,
                         meanY2Tf, stdY2Tf, medianY2Tf, varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Y2Tf, Q3Y2Tf,
                         Q1Ch3Tf, Q3Ch3Tf,
                         r, etiquetaSenal])
                    #print("caracteristicaVentana.shape")
                    #print(caracteristicaVentana.shape)
                    caracteristicasSenal = np.vstack((caracteristicasSenal, caracteristicaVentana))
                    carFreSenalCh1 = np.vstack((carFreSenalCh1, ch1HzPsd_values))
                    carFreSenalCh2 = np.vstack((carFreSenalCh2, ch2HzPsd_values))

                return caracteristicasSenal, carFreSenalCh1, carFreSenalCh2

            #

            df1 = self.df

            # Filtrado
            nomb = "C1"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            f, carFreSenalCh1, carFreSenalCh2 = extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
            # Acumula verticalmente las caracteristicas de todas las ratitas
            self.caracteristicasSenales = np.vstack([self.caracteristicasSenales, f])
            self.C1Ch1 = np.vstack([self.C1Ch1, carFreSenalCh1])
            self.C1Ch2 = np.vstack([self.C1Ch2, carFreSenalCh2])
            print("caracteristicasSenales")
            print(self.caracteristicasSenales.shape)
            print("C1Ch1")
            print(self.C1Ch1.shape)
            print("C1Ch2")
            print(self.C1Ch2.shape)

    #29 Ago 20
    def plotCSV1time(self):
        resultado_st = "plotCSV1time" + "\n"
        self.resultado.setText(resultado_st)
        #C:\Users\Mozart\Mozart\QtDesigner
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/Users/Mozart/Mozart/QtDesigner')
        # Cambiar la ruta
        # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #27-jul-20
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values

            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom, xnom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label=y0nom)
                # axs[0].set_ylim(yinf, ysup)
                axs[0].legend(frameon=False, fontsize=10)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)


                axs[1].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label=y2nom)
                # axs[2].set_ylim(yinf, ysup)
                axs[1].legend(frameon=False, fontsize=10)
                #axs[1].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[1].set_xlabel(xnom)
                axs[1].set_ylabel(y2nom)
                axs[1].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):
                # etapaFiltrado(df1, nomb)
                #print("etapaFiltrado")
                #print("Info df1")
                #print(df1.info())
                #print('\n' * 2)
                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()
                #print("Info df6")
                #print(df6.info())
                #print("\n")

                #print("Info df1=df3")
                #print(df3.info())
                #print('\n' * 2)

                #print("****Estadisticas  total****")
                # horizontal_stack = pd.concat([df3, df4], axis=1)
                #print(df3.describe(include=[np.number]))
                #print('\n' * 2)

                #print('Correlación Ch1, Ch2 y Ch3')
                #print(df3.corr())
                #print('\n' * 2)

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)
                #"""
                #27-jul-20 Calcula y gráfica la FFT empleando Welch

                # https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html
                f_valuesch1, ch1Psd_values = get_psd_values(y0, T, N, f_s)
                x = np.arange(len(ch1Psd_values))
                figura = plt.figure()
                plt.title(nombre)
                #plt.ylim(0, 20000)
                plt.xlabel("Frecuencia (Hz).")
                plt.ylabel("PSD [uV**2 / Hz]")
                #plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
                plt.plot(f_valuesch1, ch1Psd_values, "blue", label='y0:Estómago')
                #plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
                plt.legend(frameon=False, fontsize=10)
                plt.grid(True)
                # plt.xticks(x * 0.007813)
                plt.xscale('linear')
                plt.show()
                #29 Ago 20 Graficar PA antes de ser procesada por semioga

                #"""
                x = np.arange(len(ch1Psd_values))  # the label locations
                width = 0.35  # the width of the bars
                fig, ax = plt.subplots()
                # 21-07-20
                #rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width, color="blue", label='y0:Estómago',
                #                yerr=stErrCh1, align='center', alpha=0.8, ecolor='black', capsize=2)
                rects1 = ax.bar(f_valuesch1, ch1Psd_values, width, color="blue", label='y0:Estómago', align='center')
                ax.set_ylabel('PA [uV**2] y stdErr')
                #ax.set_title(nomEP)
                ax.set_xlabel('Frecuencia 0-1 Hz')
                # ax.set_xticks(x)
                ax.set_xscale('linear')

                ax.legend()
                plt.show()
                ####


                #############################################################################
                ###Se quita el 27-04-20
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estómago", y1nom="Ciego", y2nom="Ciego",xnom="30720 muestras = 1 Min")
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)
                # Señales Estomago, Ciego, Ileum
                # plotDownSamp(y0filPaBaDoSam0, y1CifilPaBaDoSam1, y1filPaBaDoSam1, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="DS Estomago", y1nom="Ds Ciego", y2nom="Ds Ileum Filtro2")
                # Señales sensor Estomago y Ciego con frecuencia corte a 0.75 Hz
                # plotDownSamp(y0HzDoSamPaBaj, y0HzDoSamPaBaj, y1HzDoSamPaBaj, yinf, ysup, inicioDs, finDs, nombre,
                #            y0nom="DS Estomago 0Hz", y1nom="Ds Estomago 0Hz", y2nom="Ds Ileum 1Hz")


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 20-04-20
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Estómago: y0 ", y1nom="Est. y0 Hz", y2nom="Ciego: y1",xnom="120 muestras = 1 Min")

                return 1


            df1 = self.df

            # Filtrado
            nomb = "C1"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            nomb = "C1 Filtrada"
            a0=extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)

    #
    def plotmeanPsdC1(self):
        #x=self.df['col1']
        resultado_st = "plotmeanPsdC1" + "\n"
        self.resultado.setText(resultado_st)
        nomb2="C1"
        #
        def estadisticaPotencia(potenciaEnFreqCh1, potenciaEnFreqCh2, nomEP):
            # estadisticaPotencia(caracteristicasFreqBasalCh1, caracteristicasFreqBasalCh2)
            # mediaPotenciaCh1 = np.median(potenciaEnFreqCh1, axis=0)
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            #print(" SSSSSSSS len(potenciaEnFreqCh1) SSSSSSSSSSSSSSSS")
            #print(len(potenciaEnFreqCh1))
            # stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            # stErrCh1=stdPotenciaCh1/(np.sqrt(len(stdPotenciaCh1)))
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))


            #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            # plt.plot(range(1, mediaPotenciaCh1.shape[0] + 1), mediaPotenciaCh1, "b")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots()
            #21-07-20
            #rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width,color="blue", label='y0:Estómago', yerr=stErrCh1)
            rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width, color="blue", label='y0:Estómago', yerr=stErrCh1,align='center', alpha=0.8, ecolor='black', capsize=2 )
            #rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width,color="r", label='y1:Ciego', yerr=stErrCh2)
            rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width, color="r", label='y1:Ciego', yerr=stErrCh2, align='center', alpha=0.8, ecolor='black', capsize=2 )
            ax.set_ylabel('PA [uV**2] y stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz')
            #ax.set_xticks(x)
            ax.set_xscale('linear')

            ax.legend()
            plt.show()

            return 0
        #
        estadisticaPotencia(self.C1Ch1, self.C1Ch2, nomb2)

    ###############################CSV2
    def getCSV2(self):
        resultado_st = "getCSV2" + "\n"
        self.resultado.setText(resultado_st)

        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #
            def entropia(X):
                """Devuelve el valor de entropia de una muestra de datos"""
                probs = [np.mean(X == valor) for valor in set(X)]
                return round(np.sum(-p * np.log2(p) for p in probs), 3)
            #
            def curtoSis(y):
                curtosisY = kurtosis(y)
                return curtosisY
            #
            def estaBasica(y):
                estDesy = np.std(y, dtype=np.float64)
                meany = np.mean(y, dtype=np.float64)
                vary = np.var(y, dtype=np.float)
                medianY = np.median(y)
                Q1 = np.quantile(np.sort(y), 0.25)
                Q3 = np.quantile(np.sort(y), 0.75)
                return vary, meany, estDesy, medianY, Q1, Q3
            #
            def maximo(x):
                max = np.amax(x)
                inmax = np.argmax(x)
                return max, inmax
            #
            def pearsonCorr(x, y):
                corrpear = scipy.stats.pearsonr(x, y)
                """
                        The p-value roughly indicates the probability of an uncorrelated system
                        producing datasets that have a Pearson correlation at least as extreme
                        as the one computed from these datasets. The p-values are not entirely
                        reliable but are probably reasonable for datasets larger than 500 or so.
                        Parameters
                """

                return corrpear
            #
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values
            # def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
            #
            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label='PaBaja')
                # axs[0].set_ylim(yinf, ysup)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)

                axs[1].plot(x_valueDowSamy1, y1filPaBaDoSam1, linestyle='-', color='g', label='PaBaja')
                # axs[1].set_ylim(yinf, ysup)
                axs[1].set_xlim(x1, x2)
                axs[1].set_ylabel(y1nom)
                axs[1].grid(True)

                axs[2].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label='PaBaja')
                # axs[2].set_ylim(yinf, ysup)
                axs[2].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[2].set_ylabel(y2nom)
                axs[2].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):

                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                """
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")
                """
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 27-04-20
                """
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Est. y0 Hz", y1nom="Est. y0 Hz", y2nom="Ciego y1 Hz s/Ruido")
                """
                ######################################################
                # y0Guardar = y0filPaBaDoSam[0:longMenorHz]
                # y1Guardar = y1filPaBaDoSam[0:longMenorHz]
                # y0y1HzStack = np.stack((y0Guardar, y1Guardar), axis=-1)
                # guardarArchivo(y0y1HzStack, nombreExCa)



                # longMenor = np.min(np.array([longy0, longy1, longy2]))
                longMenor = np.min(np.array([longy0, longy1, longy2, longy0Hzsr, longy1Hzsr]))
                #print("longMenor: ", longMenor)
                # y0Guardar = y0filPaBaDoSam[0:longMenor]
                # y1Guardar = y1filPaBaDoSam[0:longMenor]
                # y2Guardar = y2filPaBaDoSam[0:longMenor]
                # y0y1y2Stack = np.stack((y0Guardar, y1Guardar, y2Guardar), axis=-1)
                # guardarArchivo(y0y1y2Stack, nombreExCa)

                ventanas0 = longMenor // 130
                #print("ventanas")
                #print(ventanas0)

                inicioDs = 0
                finDs = longMenor
                #print("len(y0filPaBaDoSam): ", len(y0filPaBaDoSam))
                #print("finDs", finDs)
                # Gráfica señales
                # plotDownSamp(y0filPaBaDoSam, y1filPaBaDoSam, y2filPaBaDoSam, yinf, ysup, inicioDs, finDs, nombreExCa,
                #            y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")


                #print("########### ExtractFeature ###############")
                # caracteristicasSenal = np.empty((0, 83))
                caracteristicasSenal = np.empty((0, 127))
                carFreSenalCh1 = np.empty((0, 129))
                carFreSenalCh2 = np.empty((0, 129))

                # for i in range(8):
                for i in range(ventanas0 - 1):
                    #print("i: ", i)
                    dt = 1
                    x_valueDowSam = np.arange(0, longMenor, dt)
                    inicioDs = 130 * i
                    finDs = inicioDs + 260

                    # Ventana de 2 minuto 10 segundos señal down sampled
                    # fs=2 hz         # 2 muestras por segundo
                    # para 60 segundos tengo 120 muestras

                    y0filPaBaDoSamTF = y0filPaBaDoSam[inicioDs:finDs]
                    y1filPaBaDoSamTF = y1filPaBaDoSam[inicioDs:finDs]
                    y2filPaBaDoSamTF = y2filPaBaDoSam[inicioDs:finDs]
                    y0HzfilPaBaDoSamTF = y0HzfilPaBaDoSam[inicioDs:finDs]
                    y1HzfilPaBaDoSamTF = y1HzfilPaBaDoSam[inicioDs:finDs]

                    # Se llama función plotDownSamp()
                    # plotDownSamp(y0filPaBaDoSam, y0filPaBaDoSam, y0filPaBaDoSamTF, yinf, ysup, inicioDs, finDs,
                    #            nombreExCa, y0nom="Estomago", y1nom="Estomago", y2nom="Estomago")

                    # plotDownSamp(y1filPaBaDoSam, y1filPaBaDoSam, y1filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #            y0nom="Ciego", y1nom="Ciego", y2nom="Ciego")

                    # plotDownSamp(y2filPaBaDoSam, y2filPaBaDoSam, y2filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #             y0nom="Ileum", y1nom="Ileum", y2nom="Ileum")

                    kurTiemy0 = curtoSis(y0filPaBaDoSamTF)
                    kurTiemy1 = curtoSis(y1filPaBaDoSamTF)
                    kurTiemy2 = curtoSis(y2filPaBaDoSamTF)

                    entTemy0 = entropia(y0filPaBaDoSamTF)
                    entTemy1 = entropia(y1filPaBaDoSamTF)
                    entTemy2 = entropia(y2filPaBaDoSamTF)

                    varY0Tf, meanY0Tf, stdY0Tf, medianY0Tf, Q1Y0Tf, Q3Y0Tf = estaBasica(y0filPaBaDoSamTF)
                    varY1Tf, meanY1Tf, stdY1Tf, medianY1Tf, Q1Y1Tf, Q3Y1Tf = estaBasica(y1filPaBaDoSamTF)
                    varY2Tf, meanY2Tf, stdY2Tf, medianY2Tf, Q1Y2Tf, Q3Y2Tf = estaBasica(y2filPaBaDoSamTF)

                    f_valuesch1, ch1Psd_values = get_psd_values(y0filPaBaDoSamTF, T, N, f_s)
                    f_valuesch2, ch2Psd_values = get_psd_values(y1filPaBaDoSamTF, T, N, f_s)
                    f_valuesch3, ch3Psd_values = get_psd_values(y2filPaBaDoSamTF, T, N, f_s)
                    # y1HzfilPaBaDoSamTF
                    f_valuesch1Hz, ch1HzPsd_values = get_psd_values(y0HzfilPaBaDoSamTF, T, N, f_s)
                    f_valuesch2Hz, ch2HzPsd_values = get_psd_values(y1HzfilPaBaDoSamTF, T, N, f_s)

                    #print("len(ch2HzPsd_values)")
                    #print(len(ch2HzPsd_values))

                    # plt.plot(f_valuesch1, ch1Psd_values, linestyle='-', color='black', label='Estomago')
                    # plt.plot(f_valuesch2, ch2Psd_values, linestyle='-', color='g', label='Ciego')
                    # plt.plot(f_valuesch3, ch3Psd_values, linestyle='-', color='r', label='Ileon')
                    # plt.plot(f_valuesch1Hz, ch1HzPsd_values, linestyle='-', color='blue', label='Ch1')
                    # plt.plot(f_valuesch2Hz, ch2HzPsd_values, linestyle='-', color='orange', label='Ch2')
                    # plt.xlabel('Frequencia [Hz]', fontsize=16)
                    # plt.ylabel('PSD [uV**2 / Hz]', fontsize=16)
                    # plt.title("Espectro Ciego, Estomago e Ileum", fontsize=16)
                    # plt.show()

                    r = pearsonCorr(ch1HzPsd_values, ch2HzPsd_values)

                    # Estomago
                    ch1PsdValVentana = ch1Psd_values[7:26]  # 12 datos
                    varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Ch1Tf, Q3Ch1Tf = estaBasica(ch1PsdValVentana)

                    # Ciego
                    # ch2PsdValVentana = ch2Psd_values[42:68] 26 datos
                    ch2PsdValVentana = ch2Psd_values[0:7]  # 8 datos
                    varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Ch2Tf, Q3Ch2Tf = estaBasica(ch2PsdValVentana)

                    # Ileum
                    ch3PsdValVentana = ch3Psd_values[40:84]
                    varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Ch3Tf, Q3Ch3Tf = estaBasica(ch3PsdValVentana)

                    kurFreCh1 = curtoSis(ch1PsdValVentana)
                    kurFreCh2 = curtoSis(ch2PsdValVentana)
                    kurFreCh3 = curtoSis(ch3PsdValVentana)
                    entFreCh1 = entropia(ch1PsdValVentana)
                    entFreCh2 = entropia(ch2PsdValVentana)
                    entFreCh3 = entropia(ch3PsdValVentana)

                    maxch2, inmaxch2 = maximo(ch2PsdValVentana)
                    #print("Ciego: máximo valor e indice", maxch2, inmaxch2)

                    maxch1, inmaxch1 = maximo(ch1PsdValVentana)
                    #print("Estomago: máximo valor e indice", maxch1, inmaxch1)

                    maxch3, inmaxch3 = maximo(ch3PsdValVentana)
                    #print("Ileum: máximo valor e indice", maxch3, inmaxch3)

                    caracteristicaVentana = np.hstack(
                        [ch1PsdValVentana, kurTiemy0, kurFreCh1, entTemy0, entFreCh1, maxch1, inmaxch1, varY0Tf,
                         meanY0Tf, stdY0Tf, medianY0Tf, varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Y0Tf, Q3Y0Tf,
                         Q1Ch1Tf, Q3Ch1Tf,
                         ch2PsdValVentana, kurTiemy1, kurFreCh2, entTemy1, entFreCh2, maxch2, inmaxch2, varY1Tf,
                         meanY1Tf, stdY1Tf, medianY1Tf, varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Y1Tf, Q3Y1Tf,
                         Q1Ch2Tf, Q3Ch2Tf,
                         ch3PsdValVentana, kurTiemy2, kurFreCh3, entTemy2, entFreCh3, maxch3, inmaxch3, varY2Tf,
                         meanY2Tf, stdY2Tf, medianY2Tf, varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Y2Tf, Q3Y2Tf,
                         Q1Ch3Tf, Q3Ch3Tf,
                         r, etiquetaSenal])
                    #print("caracteristicaVentana.shape")
                    #print(caracteristicaVentana.shape)
                    caracteristicasSenal = np.vstack((caracteristicasSenal, caracteristicaVentana))
                    carFreSenalCh1 = np.vstack((carFreSenalCh1, ch1HzPsd_values))
                    carFreSenalCh2 = np.vstack((carFreSenalCh2, ch2HzPsd_values))

                return caracteristicasSenal, carFreSenalCh1, carFreSenalCh2

            #

            df1 = self.df

            # Filtrado
            nomb = "C2"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            f, carFreSenalCh1, carFreSenalCh2 = extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
            # Acumula verticalmente las caracteristicas de todas las ratitas
            self.caracteristicasSenales = np.vstack([self.caracteristicasSenales, f])
            self.C2Ch1 = np.vstack([self.C2Ch1, carFreSenalCh1])
            self.C2Ch2 = np.vstack([self.C2Ch2, carFreSenalCh2])
            print("caracteristicasSenales")
            print(self.caracteristicasSenales.shape)
            print("C2Ch1")
            print(self.C2Ch1.shape)
            print("C2Ch2")
            print(self.C2Ch2.shape)
    #
    def plotCSV2time(self):
        resultado_st = "plotCSV2time" + "\n"
        self.resultado.setText(resultado_st)
        #C:\Users\Mozart\Mozart\QtDesigner
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/Users/Mozart/Mozart/QtDesigner')
        # Cambiar la ruta
        # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom, xnom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label=y0nom)
                # axs[0].set_ylim(yinf, ysup)
                axs[0].legend(frameon=False, fontsize=10)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)


                axs[1].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label=y2nom)
                # axs[2].set_ylim(yinf, ysup)
                axs[1].legend(frameon=False, fontsize=10)
                #axs[1].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[1].set_xlabel(xnom)
                axs[1].set_ylabel(y2nom)
                axs[1].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):
                # etapaFiltrado(df1, nomb)
                #print("etapaFiltrado")
                #print("Info df1")
                #print(df1.info())
                #print('\n' * 2)
                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()
                #print("Info df6")
                #print(df6.info())
                #print("\n")

                #print("Info df1=df3")
                #print(df3.info())
                #print('\n' * 2)

                #print("****Estadisticas  total****")
                # horizontal_stack = pd.concat([df3, df4], axis=1)
                #print(df3.describe(include=[np.number]))
                #print('\n' * 2)

                #print('Correlación Ch1, Ch2 y Ch3')
                #print(df3.corr())
                #print('\n' * 2)

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estómago", y1nom="Ciego", y2nom="Ciego",xnom="30720 muestras = 1 Min")
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)
                # Señales Estomago, Ciego, Ileum
                # plotDownSamp(y0filPaBaDoSam0, y1CifilPaBaDoSam1, y1filPaBaDoSam1, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="DS Estomago", y1nom="Ds Ciego", y2nom="Ds Ileum Filtro2")
                # Señales sensor Estomago y Ciego con frecuencia corte a 0.75 Hz
                # plotDownSamp(y0HzDoSamPaBaj, y0HzDoSamPaBaj, y1HzDoSamPaBaj, yinf, ysup, inicioDs, finDs, nombre,
                #            y0nom="DS Estomago 0Hz", y1nom="Ds Estomago 0Hz", y2nom="Ds Ileum 1Hz")


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 20-04-20
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Estómago: y0 ", y1nom="Est. y0 Hz", y2nom="Ciego: y1",xnom="120 muestras = 1 Min")

                return 1


            df1 = self.df

            # Filtrado
            nomb = "C2"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            nomb = "C2 Filtrada"
            a0=extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
    #
    def plotmeanPsdC2(self):
        #x=self.df['col1']
        resultado_st = "plotmeanPsdC2" + "\n"
        self.resultado.setText(resultado_st)
        nomb2="C2"
        #
        def estadisticaPotencia(potenciaEnFreqCh1, potenciaEnFreqCh2, nomEP):
            # estadisticaPotencia(caracteristicasFreqBasalCh1, caracteristicasFreqBasalCh2)
            # mediaPotenciaCh1 = np.median(potenciaEnFreqCh1, axis=0)
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            #print(" SSSSSSSS len(potenciaEnFreqCh1) SSSSSSSSSSSSSSSS")
            #print(len(potenciaEnFreqCh1))
            # stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            # stErrCh1=stdPotenciaCh1/(np.sqrt(len(stdPotenciaCh1)))
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))


            #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            # plt.plot(range(1, mediaPotenciaCh1.shape[0] + 1), mediaPotenciaCh1, "b")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots()
            #21-07-20
            #align='center', alpha=0.8, ecolor='black', capsize=2
            #rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width,color="blue", label='y0:Estómago', yerr=stErrCh1)
            rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width, color="blue", label='y0:Estómago', yerr=stErrCh1, align='center', alpha=0.8, ecolor='black', capsize=2)
            rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width,color="r", label='y1:Ciego', yerr=stErrCh2, align='center', alpha=0.8, ecolor='black', capsize=2)
            ax.set_ylabel('PA [uV**2] y stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz')
            #ax.set_xticks(x)
            ax.set_xscale('linear')

            ax.legend()
            plt.show()

            return 0
        #
        estadisticaPotencia(self.C2Ch1, self.C2Ch2, nomb2)
    #
    ##############################CSV3

    def getCSV3(self):
        resultado_st = "getCSV3" + "\n"
        self.resultado.setText(resultado_st)

        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #
            def entropia(X):
                """Devuelve el valor de entropia de una muestra de datos"""
                probs = [np.mean(X == valor) for valor in set(X)]
                return round(np.sum(-p * np.log2(p) for p in probs), 3)
            #
            def curtoSis(y):
                curtosisY = kurtosis(y)
                return curtosisY
            #
            def estaBasica(y):
                estDesy = np.std(y, dtype=np.float64)
                meany = np.mean(y, dtype=np.float64)
                vary = np.var(y, dtype=np.float)
                medianY = np.median(y)
                Q1 = np.quantile(np.sort(y), 0.25)
                Q3 = np.quantile(np.sort(y), 0.75)
                return vary, meany, estDesy, medianY, Q1, Q3
            #
            def maximo(x):
                max = np.amax(x)
                inmax = np.argmax(x)
                return max, inmax
            #
            def pearsonCorr(x, y):
                corrpear = scipy.stats.pearsonr(x, y)
                """
                        The p-value roughly indicates the probability of an uncorrelated system
                        producing datasets that have a Pearson correlation at least as extreme
                        as the one computed from these datasets. The p-values are not entirely
                        reliable but are probably reasonable for datasets larger than 500 or so.
                        Parameters
                """

                return corrpear
            #
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values
            # def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
            #
            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label='PaBaja')
                # axs[0].set_ylim(yinf, ysup)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)

                axs[1].plot(x_valueDowSamy1, y1filPaBaDoSam1, linestyle='-', color='g', label='PaBaja')
                # axs[1].set_ylim(yinf, ysup)
                axs[1].set_xlim(x1, x2)
                axs[1].set_ylabel(y1nom)
                axs[1].grid(True)

                axs[2].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label='PaBaja')
                # axs[2].set_ylim(yinf, ysup)
                axs[2].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[2].set_ylabel(y2nom)
                axs[2].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):

                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                """
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")
                """
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 27-04-20
                """
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Est. y0 Hz", y1nom="Est. y0 Hz", y2nom="Ciego y1 Hz s/Ruido")
                """
                ######################################################
                # y0Guardar = y0filPaBaDoSam[0:longMenorHz]
                # y1Guardar = y1filPaBaDoSam[0:longMenorHz]
                # y0y1HzStack = np.stack((y0Guardar, y1Guardar), axis=-1)
                # guardarArchivo(y0y1HzStack, nombreExCa)



                # longMenor = np.min(np.array([longy0, longy1, longy2]))
                longMenor = np.min(np.array([longy0, longy1, longy2, longy0Hzsr, longy1Hzsr]))
                #print("longMenor: ", longMenor)
                # y0Guardar = y0filPaBaDoSam[0:longMenor]
                # y1Guardar = y1filPaBaDoSam[0:longMenor]
                # y2Guardar = y2filPaBaDoSam[0:longMenor]
                # y0y1y2Stack = np.stack((y0Guardar, y1Guardar, y2Guardar), axis=-1)
                # guardarArchivo(y0y1y2Stack, nombreExCa)

                ventanas0 = longMenor // 130
                #print("ventanas")
                #print(ventanas0)

                inicioDs = 0
                finDs = longMenor
                #print("len(y0filPaBaDoSam): ", len(y0filPaBaDoSam))
                #print("finDs", finDs)
                # Gráfica señales
                # plotDownSamp(y0filPaBaDoSam, y1filPaBaDoSam, y2filPaBaDoSam, yinf, ysup, inicioDs, finDs, nombreExCa,
                #            y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")


                #print("########### ExtractFeature ###############")
                # caracteristicasSenal = np.empty((0, 83))
                caracteristicasSenal = np.empty((0, 127))
                carFreSenalCh1 = np.empty((0, 129))
                carFreSenalCh2 = np.empty((0, 129))

                # for i in range(8):
                for i in range(ventanas0 - 1):
                    #print("i: ", i)
                    dt = 1
                    x_valueDowSam = np.arange(0, longMenor, dt)
                    inicioDs = 130 * i
                    finDs = inicioDs + 260

                    # Ventana de 2 minuto 10 segundos señal down sampled
                    # fs=2 hz         # 2 muestras por segundo
                    # para 60 segundos tengo 120 muestras

                    y0filPaBaDoSamTF = y0filPaBaDoSam[inicioDs:finDs]
                    y1filPaBaDoSamTF = y1filPaBaDoSam[inicioDs:finDs]
                    y2filPaBaDoSamTF = y2filPaBaDoSam[inicioDs:finDs]
                    y0HzfilPaBaDoSamTF = y0HzfilPaBaDoSam[inicioDs:finDs]
                    y1HzfilPaBaDoSamTF = y1HzfilPaBaDoSam[inicioDs:finDs]

                    # Se llama función plotDownSamp()
                    # plotDownSamp(y0filPaBaDoSam, y0filPaBaDoSam, y0filPaBaDoSamTF, yinf, ysup, inicioDs, finDs,
                    #            nombreExCa, y0nom="Estomago", y1nom="Estomago", y2nom="Estomago")

                    # plotDownSamp(y1filPaBaDoSam, y1filPaBaDoSam, y1filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #            y0nom="Ciego", y1nom="Ciego", y2nom="Ciego")

                    # plotDownSamp(y2filPaBaDoSam, y2filPaBaDoSam, y2filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #             y0nom="Ileum", y1nom="Ileum", y2nom="Ileum")

                    kurTiemy0 = curtoSis(y0filPaBaDoSamTF)
                    kurTiemy1 = curtoSis(y1filPaBaDoSamTF)
                    kurTiemy2 = curtoSis(y2filPaBaDoSamTF)

                    entTemy0 = entropia(y0filPaBaDoSamTF)
                    entTemy1 = entropia(y1filPaBaDoSamTF)
                    entTemy2 = entropia(y2filPaBaDoSamTF)

                    varY0Tf, meanY0Tf, stdY0Tf, medianY0Tf, Q1Y0Tf, Q3Y0Tf = estaBasica(y0filPaBaDoSamTF)
                    varY1Tf, meanY1Tf, stdY1Tf, medianY1Tf, Q1Y1Tf, Q3Y1Tf = estaBasica(y1filPaBaDoSamTF)
                    varY2Tf, meanY2Tf, stdY2Tf, medianY2Tf, Q1Y2Tf, Q3Y2Tf = estaBasica(y2filPaBaDoSamTF)

                    f_valuesch1, ch1Psd_values = get_psd_values(y0filPaBaDoSamTF, T, N, f_s)
                    f_valuesch2, ch2Psd_values = get_psd_values(y1filPaBaDoSamTF, T, N, f_s)
                    f_valuesch3, ch3Psd_values = get_psd_values(y2filPaBaDoSamTF, T, N, f_s)
                    # y1HzfilPaBaDoSamTF
                    f_valuesch1Hz, ch1HzPsd_values = get_psd_values(y0HzfilPaBaDoSamTF, T, N, f_s)
                    f_valuesch2Hz, ch2HzPsd_values = get_psd_values(y1HzfilPaBaDoSamTF, T, N, f_s)

                    #print("len(ch2HzPsd_values)")
                    #print(len(ch2HzPsd_values))

                    # plt.plot(f_valuesch1, ch1Psd_values, linestyle='-', color='black', label='Estomago')
                    # plt.plot(f_valuesch2, ch2Psd_values, linestyle='-', color='g', label='Ciego')
                    # plt.plot(f_valuesch3, ch3Psd_values, linestyle='-', color='r', label='Ileon')
                    # plt.plot(f_valuesch1Hz, ch1HzPsd_values, linestyle='-', color='blue', label='Ch1')
                    # plt.plot(f_valuesch2Hz, ch2HzPsd_values, linestyle='-', color='orange', label='Ch2')
                    # plt.xlabel('Frequencia [Hz]', fontsize=16)
                    # plt.ylabel('PSD [uV**2 / Hz]', fontsize=16)
                    # plt.title("Espectro Ciego, Estomago e Ileum", fontsize=16)
                    # plt.show()

                    r = pearsonCorr(ch1HzPsd_values, ch2HzPsd_values)

                    # Estomago
                    ch1PsdValVentana = ch1Psd_values[7:26]  # 12 datos
                    varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Ch1Tf, Q3Ch1Tf = estaBasica(ch1PsdValVentana)

                    # Ciego
                    # ch2PsdValVentana = ch2Psd_values[42:68] 26 datos
                    ch2PsdValVentana = ch2Psd_values[0:7]  # 8 datos
                    varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Ch2Tf, Q3Ch2Tf = estaBasica(ch2PsdValVentana)

                    # Ileum
                    ch3PsdValVentana = ch3Psd_values[40:84]
                    varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Ch3Tf, Q3Ch3Tf = estaBasica(ch3PsdValVentana)

                    kurFreCh1 = curtoSis(ch1PsdValVentana)
                    kurFreCh2 = curtoSis(ch2PsdValVentana)
                    kurFreCh3 = curtoSis(ch3PsdValVentana)
                    entFreCh1 = entropia(ch1PsdValVentana)
                    entFreCh2 = entropia(ch2PsdValVentana)
                    entFreCh3 = entropia(ch3PsdValVentana)

                    maxch2, inmaxch2 = maximo(ch2PsdValVentana)
                    #print("Ciego: máximo valor e indice", maxch2, inmaxch2)

                    maxch1, inmaxch1 = maximo(ch1PsdValVentana)
                    #print("Estomago: máximo valor e indice", maxch1, inmaxch1)

                    maxch3, inmaxch3 = maximo(ch3PsdValVentana)
                    #print("Ileum: máximo valor e indice", maxch3, inmaxch3)

                    caracteristicaVentana = np.hstack(
                        [ch1PsdValVentana, kurTiemy0, kurFreCh1, entTemy0, entFreCh1, maxch1, inmaxch1, varY0Tf,
                         meanY0Tf, stdY0Tf, medianY0Tf, varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Y0Tf, Q3Y0Tf,
                         Q1Ch1Tf, Q3Ch1Tf,
                         ch2PsdValVentana, kurTiemy1, kurFreCh2, entTemy1, entFreCh2, maxch2, inmaxch2, varY1Tf,
                         meanY1Tf, stdY1Tf, medianY1Tf, varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Y1Tf, Q3Y1Tf,
                         Q1Ch2Tf, Q3Ch2Tf,
                         ch3PsdValVentana, kurTiemy2, kurFreCh3, entTemy2, entFreCh3, maxch3, inmaxch3, varY2Tf,
                         meanY2Tf, stdY2Tf, medianY2Tf, varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Y2Tf, Q3Y2Tf,
                         Q1Ch3Tf, Q3Ch3Tf,
                         r, etiquetaSenal])
                    #print("caracteristicaVentana.shape")
                    #print(caracteristicaVentana.shape)
                    caracteristicasSenal = np.vstack((caracteristicasSenal, caracteristicaVentana))
                    carFreSenalCh1 = np.vstack((carFreSenalCh1, ch1HzPsd_values))
                    carFreSenalCh2 = np.vstack((carFreSenalCh2, ch2HzPsd_values))

                return caracteristicasSenal, carFreSenalCh1, carFreSenalCh2

            #

            df1 = self.df

            # Filtrado
            nomb = "C3"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            f, carFreSenalCh1, carFreSenalCh2 = extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
            # Acumula verticalmente las caracteristicas de todas las ratitas
            self.caracteristicasSenales = np.vstack([self.caracteristicasSenales, f])
            self.C3Ch1 = np.vstack([self.C3Ch1, carFreSenalCh1])
            self.C3Ch2 = np.vstack([self.C3Ch2, carFreSenalCh2])
            print("caracteristicasSenales")
            print(self.caracteristicasSenales.shape)
            print("C3Ch1")
            print(self.C3Ch1.shape)
            print("C2Ch2")
            print(self.C3Ch2.shape)
    #
    def plotCSV3time(self):
        resultado_st = "plotCSV3time" + "\n"
        self.resultado.setText(resultado_st)
        #C:\Users\Mozart\Mozart\QtDesigner
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/Users/Mozart/Mozart/QtDesigner')
        # Cambiar la ruta
        # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom, xnom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label=y0nom)
                # axs[0].set_ylim(yinf, ysup)
                axs[0].legend(frameon=False, fontsize=10)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)


                axs[1].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label=y2nom)
                # axs[2].set_ylim(yinf, ysup)
                axs[1].legend(frameon=False, fontsize=10)
                #axs[1].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[1].set_xlabel(xnom)
                axs[1].set_ylabel(y2nom)
                axs[1].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):
                # etapaFiltrado(df1, nomb)
                #print("etapaFiltrado")
                #print("Info df1")
                #print(df1.info())
                #print('\n' * 2)
                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()
                #print("Info df6")
                #print(df6.info())
                #print("\n")

                #print("Info df1=df3")
                #print(df3.info())
                #print('\n' * 2)

                #print("****Estadisticas  total****")
                # horizontal_stack = pd.concat([df3, df4], axis=1)
                #print(df3.describe(include=[np.number]))
                #print('\n' * 2)

                #print('Correlación Ch1, Ch2 y Ch3')
                #print(df3.corr())
                #print('\n' * 2)

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estómago", y1nom="Ciego", y2nom="Ciego",xnom="30720 muestras = 1 Min")
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)
                # Señales Estomago, Ciego, Ileum
                # plotDownSamp(y0filPaBaDoSam0, y1CifilPaBaDoSam1, y1filPaBaDoSam1, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="DS Estomago", y1nom="Ds Ciego", y2nom="Ds Ileum Filtro2")
                # Señales sensor Estomago y Ciego con frecuencia corte a 0.75 Hz
                # plotDownSamp(y0HzDoSamPaBaj, y0HzDoSamPaBaj, y1HzDoSamPaBaj, yinf, ysup, inicioDs, finDs, nombre,
                #            y0nom="DS Estomago 0Hz", y1nom="Ds Estomago 0Hz", y2nom="Ds Ileum 1Hz")


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 20-04-20
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Estómago: y0 ", y1nom="Est. y0 Hz", y2nom="Ciego: y1",xnom="120 muestras = 1 Min")

                return 1


            df1 = self.df

            # Filtrado
            nomb = "C3"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            nomb = "C3 Filtrada"
            a0=extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
    #
    def plotmeanPSDC3(self):
        #x=self.df['col1']
        resultado_st = "plotmeanPSDC3" + "\n"
        self.resultado.setText(resultado_st)
        nomb2="C3"
        #
        def estadisticaPotencia(potenciaEnFreqCh1, potenciaEnFreqCh2, nomEP):
            # estadisticaPotencia(caracteristicasFreqBasalCh1, caracteristicasFreqBasalCh2)
            # mediaPotenciaCh1 = np.median(potenciaEnFreqCh1, axis=0)
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            #print(" SSSSSSSS len(potenciaEnFreqCh1) SSSSSSSSSSSSSSSS")
            #print(len(potenciaEnFreqCh1))
            # stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            # stErrCh1=stdPotenciaCh1/(np.sqrt(len(stdPotenciaCh1)))
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))


            #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            # plt.plot(range(1, mediaPotenciaCh1.shape[0] + 1), mediaPotenciaCh1, "b")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots()
            #21-jul-20
            #align='center', alpha=0.8, ecolor='black', capsize=2
            #rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width,color="blue", label='y0:Estómago', yerr=stErrCh1)
            rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width, color="blue", label='y0:Estómago', yerr=stErrCh1,align='center', alpha=0.8, ecolor='black', capsize=2 )
            rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width,color="r", label='y1:Ciego', yerr=stErrCh2, align='center', alpha=0.8, ecolor='black', capsize=2)
            ax.set_ylabel('PA [uV**2] y stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz')
            #ax.set_xticks(x)
            ax.set_xscale('linear')

            ax.legend()
            plt.show()

            return 0
        #
        estadisticaPotencia(self.C3Ch1, self.C3Ch2, nomb2)
    #
    ############################Resetea contadores
    def resC1(self):
        #
        a = 1
        resultado_st = "resC1" + "\n"
        self.resultado.setText(resultado_st)
        # Matrices archivo CSV Grupo1

        self.C1Ch1 = np.empty((0, 129))
        print("C1Ch1")
        print(self.C1Ch1)

        self.C1Ch2 = np.empty((0, 129))
        print("C1Ch2")
        print(self.C1Ch2)
        #
    #
    def resC2(self):
        a=1
        resultado_st = "resC2" + "\n"
        self.resultado.setText(resultado_st)

        self.C2Ch1 = np.empty((0, 129))
        print("C2Ch1")
        print(self.C2Ch1)

        self.C2Ch2 = np.empty((0, 129))
        print("C2Ch2")
        print(self.C2Ch2)

    def resC3(self):
        a=1
        resultado_st = "resC3" + "\n"
        self.resultado.setText(resultado_st)
        self.C3Ch1 = np.empty((0, 129))
        print("C3Ch1")
        print(self.C3Ch1)

        self.C3Ch2 = np.empty((0, 129))
        print("C3Ch2")
        print(self.C3Ch2)

    ########################### Gráfica 3 condiciones
    def plotPSDch1(self):

        #
        resultado_st = "plotmeanPSDch1" + "\n"
        self.resultado.setText(resultado_st)
        def estadisticaPotencia3Gps(potenciaEnFreqCh1, potenciaEnFreqCh2, potenciaEnFreqCh3, nomEP):
            # https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)
            mediaPotenciaCh3 = np.mean(potenciaEnFreqCh3, axis=0, dtype=np.float64)

            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)
            stdPotenciaCh3 = np.std(potenciaEnFreqCh3, axis=0, dtype=np.float64)
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))
            stErrCh3 = stdPotenciaCh3 / (np.sqrt(len(potenciaEnFreqCh3)))

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "b", label='C1')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='C2')
            plt.plot(x * 0.007813, mediaPotenciaCh3, "g", label='C3')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.3  # the width of the bars
            fig, ax = plt.subplots()
            #21-jul-20
            #align='center', alpha=0.8, ecolor='black', capsize=2
            #rects1 = ax.bar(x - width, mediaPotenciaCh1, width, label='C1', yerr=stErrCh1)
            rects1 = ax.bar(x - width, mediaPotenciaCh1, width, label='C1', yerr=stErrCh1,align='center', alpha=0.8, ecolor='black', capsize=2)
            rects2 = ax.bar(x, mediaPotenciaCh2, width, label='C2', yerr=stErrCh2, align='center', alpha=0.8, ecolor='black', capsize=2)
            rects3 = ax.bar(x + width, mediaPotenciaCh3, width, label='C3', yerr=stErrCh3, align='center', alpha=0.8, ecolor='black', capsize=2)
            ax.set_ylabel('PA [uV**2], stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz.')
            #ax.set_xticks(x)
            ax.set_xscale('linear')
            ax.legend()
            plt.show()

            return 0
        #
        nomEP = "C1, C2, C3. y0"
        estadisticaPotencia3Gps(self.C1Ch1, self.C2Ch1, self.C3Ch1, nomEP)
    #
    def plotPSDch2(self):
        #
        resultado_st = "plotmeanPSDch2" + "\n"
        self.resultado.setText(resultado_st)
        def estadisticaPotencia3Gps(potenciaEnFreqCh1, potenciaEnFreqCh2, potenciaEnFreqCh3, nomEP):
            # https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)
            mediaPotenciaCh3 = np.mean(potenciaEnFreqCh3, axis=0, dtype=np.float64)

            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)
            stdPotenciaCh3 = np.std(potenciaEnFreqCh3, axis=0, dtype=np.float64)
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))
            stErrCh3 = stdPotenciaCh3 / (np.sqrt(len(potenciaEnFreqCh3)))

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "b", label='C1')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='C2')
            plt.plot(x * 0.007813, mediaPotenciaCh3, "g", label='C3')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.3  # the width of the bars
            fig, ax = plt.subplots()
            #21-jul-20
            #align='center', alpha=0.8, ecolor='black', capsize=2
            #rects1 = ax.bar(x - width, mediaPotenciaCh1, width, label='C1', yerr=stErrCh1)
            rects1 = ax.bar(x - width, mediaPotenciaCh1, width, label='C1', yerr=stErrCh1, align='center', alpha=0.8, ecolor='black', capsize=2)
            rects2 = ax.bar(x, mediaPotenciaCh2, width, label='C2', yerr=stErrCh2, align='center', alpha=0.8, ecolor='black', capsize=2)
            rects3 = ax.bar(x + width, mediaPotenciaCh3, width, label='C3', yerr=stErrCh3, align='center', alpha=0.8, ecolor='black', capsize=2)
            ax.set_ylabel('PA [uV**2], stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz.')
            #ax.set_xticks(x)
            ax.set_xscale('linear')
            ax.legend()
            plt.show()

            return 0
        #
        nomEP = "C1, C2, C3. y1"
        estadisticaPotencia3Gps(self.C1Ch2, self.C2Ch2, self.C3Ch2, nomEP)
    #
    # Abre ventana principal.
    def abrirVentanaPrincipal(self):
        self.parent().show()
        self.close()

class ventanaCua(QtWidgets.QMainWindow, Ui_cuatro):
#class ventanaCua(QMainWindow):
    def __init__(self, parent=None):
        super(ventanaCua, self).__init__(parent)
        #loadUi('cuatro.ui', self)
        Ui_cuatro.__init__(self)
        self.setupUi(self)
        self.boton6_4.clicked.connect(self.abrirVentanaPrincipal)

        # Botones Con1
        self.boton1.clicked.connect(self.getCSV1)
        self.boton1_1.clicked.connect(self.plotCSV1time)
        self.boton1_2.clicked.connect(self.plotmeanPsdC1)

        # Aquí va el botón Palm
        self.boton3.clicked.connect(self.getCSV2)
        self.boton3_1.clicked.connect(self.plotCSV2time)
        self.boton3_2.clicked.connect(self.plotmeanPsdC2)

        # Aquí va el botón Pesc
        self.boton4.clicked.connect(self.getCSV3)
        self.boton4_1.clicked.connect(self.plotCSV3time)
        self.boton4_2.clicked.connect(self.plotmeanPsdC3)

        # Botón condición C4
        self.boton5.clicked.connect(self.getCSV4)
        self.boton5_1.clicked.connect(self.plotCSV4time)
        self.boton5_2.clicked.connect(self.plotmeanPsdC4)

        # Botón comparación 4 señales ch1
        self.boton2.clicked.connect(self.plotPSDch1)
        # Botón comparación 4 señales ch2
        self.boton2_1.clicked.connect(self.plotPSDch2)

        # Boton reiniciar captura archivos ctr
        self.boton1_3.clicked.connect(self.resC1)
        # Boton reiniciar captura archivos Palma
        self.boton3_3.clicked.connect(self.resC2)
        # Boton reiniciar captura archivos Pescado
        self.boton4_3.clicked.connect(self.resC3)
        # Boton reiniciar captura archivos C4
        self.boton5_3.clicked.connect(self.resC4)

        # Matrices archivo general de caracteristicas
        self.caracteristicasSenales = np.empty((0, 127))

        # Matrices archivo CSV Grupo1 Ctr
        self.C1Ch1 = np.empty((0, 129))
        self.C1Ch2 = np.empty((0, 129))

        # Matrices archivo CSV Grupo2 Palm
        self.C2Ch1 = np.empty((0, 129))
        self.C2Ch2 = np.empty((0, 129))

        # Matrices archivo CSV Grupo3
        self.C3Ch1 = np.empty((0, 129))
        self.C3Ch2 = np.empty((0, 129))

        # Matrices archivo CSV Grupo4
        self.C4Ch1 = np.empty((0, 129))
        self.C4Ch2 = np.empty((0, 129))


        #

#### Aquí van las funciones

    ######### C1
    #
    def getCSV1(self):
        resultado_st = "getCSV1" + "\n"
        self.resultado.setText(resultado_st)

        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #
            def entropia(X):
                """Devuelve el valor de entropia de una muestra de datos"""
                probs = [np.mean(X == valor) for valor in set(X)]
                return round(np.sum(-p * np.log2(p) for p in probs), 3)
            #
            def curtoSis(y):
                curtosisY = kurtosis(y)
                return curtosisY
            #
            def estaBasica(y):
                estDesy = np.std(y, dtype=np.float64)
                meany = np.mean(y, dtype=np.float64)
                vary = np.var(y, dtype=np.float)
                medianY = np.median(y)
                Q1 = np.quantile(np.sort(y), 0.25)
                Q3 = np.quantile(np.sort(y), 0.75)
                return vary, meany, estDesy, medianY, Q1, Q3
            #
            def maximo(x):
                max = np.amax(x)
                inmax = np.argmax(x)
                return max, inmax
            #
            def pearsonCorr(x, y):
                corrpear = scipy.stats.pearsonr(x, y)
                """
                        The p-value roughly indicates the probability of an uncorrelated system
                        producing datasets that have a Pearson correlation at least as extreme
                        as the one computed from these datasets. The p-values are not entirely
                        reliable but are probably reasonable for datasets larger than 500 or so.
                        Parameters
                """

                return corrpear
            #
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values
            # def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
            #
            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label='PaBaja')
                # axs[0].set_ylim(yinf, ysup)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)

                axs[1].plot(x_valueDowSamy1, y1filPaBaDoSam1, linestyle='-', color='g', label='PaBaja')
                # axs[1].set_ylim(yinf, ysup)
                axs[1].set_xlim(x1, x2)
                axs[1].set_ylabel(y1nom)
                axs[1].grid(True)

                axs[2].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label='PaBaja')
                # axs[2].set_ylim(yinf, ysup)
                axs[2].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[2].set_ylabel(y2nom)
                axs[2].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):

                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                """
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")
                """
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 27-04-20
                """
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Est. y0 Hz", y1nom="Est. y0 Hz", y2nom="Ciego y1 Hz s/Ruido")
                """
                ######################################################
                # y0Guardar = y0filPaBaDoSam[0:longMenorHz]
                # y1Guardar = y1filPaBaDoSam[0:longMenorHz]
                # y0y1HzStack = np.stack((y0Guardar, y1Guardar), axis=-1)
                # guardarArchivo(y0y1HzStack, nombreExCa)



                # longMenor = np.min(np.array([longy0, longy1, longy2]))
                longMenor = np.min(np.array([longy0, longy1, longy2, longy0Hzsr, longy1Hzsr]))
                #print("longMenor: ", longMenor)
                # y0Guardar = y0filPaBaDoSam[0:longMenor]
                # y1Guardar = y1filPaBaDoSam[0:longMenor]
                # y2Guardar = y2filPaBaDoSam[0:longMenor]
                # y0y1y2Stack = np.stack((y0Guardar, y1Guardar, y2Guardar), axis=-1)
                # guardarArchivo(y0y1y2Stack, nombreExCa)

                ventanas0 = longMenor // 130
                #print("ventanas")
                #print(ventanas0)

                inicioDs = 0
                finDs = longMenor
                #print("len(y0filPaBaDoSam): ", len(y0filPaBaDoSam))
                #print("finDs", finDs)
                # Gráfica señales
                # plotDownSamp(y0filPaBaDoSam, y1filPaBaDoSam, y2filPaBaDoSam, yinf, ysup, inicioDs, finDs, nombreExCa,
                #            y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")


                #print("########### ExtractFeature ###############")
                # caracteristicasSenal = np.empty((0, 83))
                caracteristicasSenal = np.empty((0, 127))
                carFreSenalCh1 = np.empty((0, 129))
                carFreSenalCh2 = np.empty((0, 129))

                # for i in range(8):
                for i in range(ventanas0 - 1):
                    #print("i: ", i)
                    dt = 1
                    x_valueDowSam = np.arange(0, longMenor, dt)
                    inicioDs = 130 * i
                    finDs = inicioDs + 260

                    # Ventana de 2 minuto 10 segundos señal down sampled
                    # fs=2 hz         # 2 muestras por segundo
                    # para 60 segundos tengo 120 muestras

                    y0filPaBaDoSamTF = y0filPaBaDoSam[inicioDs:finDs]
                    y1filPaBaDoSamTF = y1filPaBaDoSam[inicioDs:finDs]
                    y2filPaBaDoSamTF = y2filPaBaDoSam[inicioDs:finDs]
                    y0HzfilPaBaDoSamTF = y0HzfilPaBaDoSam[inicioDs:finDs]
                    y1HzfilPaBaDoSamTF = y1HzfilPaBaDoSam[inicioDs:finDs]

                    # Se llama función plotDownSamp()
                    # plotDownSamp(y0filPaBaDoSam, y0filPaBaDoSam, y0filPaBaDoSamTF, yinf, ysup, inicioDs, finDs,
                    #            nombreExCa, y0nom="Estomago", y1nom="Estomago", y2nom="Estomago")

                    # plotDownSamp(y1filPaBaDoSam, y1filPaBaDoSam, y1filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #            y0nom="Ciego", y1nom="Ciego", y2nom="Ciego")

                    # plotDownSamp(y2filPaBaDoSam, y2filPaBaDoSam, y2filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #             y0nom="Ileum", y1nom="Ileum", y2nom="Ileum")

                    kurTiemy0 = curtoSis(y0filPaBaDoSamTF)
                    kurTiemy1 = curtoSis(y1filPaBaDoSamTF)
                    kurTiemy2 = curtoSis(y2filPaBaDoSamTF)

                    entTemy0 = entropia(y0filPaBaDoSamTF)
                    entTemy1 = entropia(y1filPaBaDoSamTF)
                    entTemy2 = entropia(y2filPaBaDoSamTF)

                    varY0Tf, meanY0Tf, stdY0Tf, medianY0Tf, Q1Y0Tf, Q3Y0Tf = estaBasica(y0filPaBaDoSamTF)
                    varY1Tf, meanY1Tf, stdY1Tf, medianY1Tf, Q1Y1Tf, Q3Y1Tf = estaBasica(y1filPaBaDoSamTF)
                    varY2Tf, meanY2Tf, stdY2Tf, medianY2Tf, Q1Y2Tf, Q3Y2Tf = estaBasica(y2filPaBaDoSamTF)

                    f_valuesch1, ch1Psd_values = get_psd_values(y0filPaBaDoSamTF, T, N, f_s)
                    f_valuesch2, ch2Psd_values = get_psd_values(y1filPaBaDoSamTF, T, N, f_s)
                    f_valuesch3, ch3Psd_values = get_psd_values(y2filPaBaDoSamTF, T, N, f_s)
                    # y1HzfilPaBaDoSamTF
                    f_valuesch1Hz, ch1HzPsd_values = get_psd_values(y0HzfilPaBaDoSamTF, T, N, f_s)
                    f_valuesch2Hz, ch2HzPsd_values = get_psd_values(y1HzfilPaBaDoSamTF, T, N, f_s)

                    #print("len(ch2HzPsd_values)")
                    #print(len(ch2HzPsd_values))

                    # plt.plot(f_valuesch1, ch1Psd_values, linestyle='-', color='black', label='Estomago')
                    # plt.plot(f_valuesch2, ch2Psd_values, linestyle='-', color='g', label='Ciego')
                    # plt.plot(f_valuesch3, ch3Psd_values, linestyle='-', color='r', label='Ileon')
                    # plt.plot(f_valuesch1Hz, ch1HzPsd_values, linestyle='-', color='blue', label='Ch1')
                    # plt.plot(f_valuesch2Hz, ch2HzPsd_values, linestyle='-', color='orange', label='Ch2')
                    # plt.xlabel('Frequencia [Hz]', fontsize=16)
                    # plt.ylabel('PSD [uV**2 / Hz]', fontsize=16)
                    # plt.title("Espectro Ciego, Estomago e Ileum", fontsize=16)
                    # plt.show()

                    r = pearsonCorr(ch1HzPsd_values, ch2HzPsd_values)

                    # Estomago
                    ch1PsdValVentana = ch1Psd_values[7:26]  # 12 datos
                    varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Ch1Tf, Q3Ch1Tf = estaBasica(ch1PsdValVentana)

                    # Ciego
                    # ch2PsdValVentana = ch2Psd_values[42:68] 26 datos
                    ch2PsdValVentana = ch2Psd_values[0:7]  # 8 datos
                    varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Ch2Tf, Q3Ch2Tf = estaBasica(ch2PsdValVentana)

                    # Ileum
                    ch3PsdValVentana = ch3Psd_values[40:84]
                    varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Ch3Tf, Q3Ch3Tf = estaBasica(ch3PsdValVentana)

                    kurFreCh1 = curtoSis(ch1PsdValVentana)
                    kurFreCh2 = curtoSis(ch2PsdValVentana)
                    kurFreCh3 = curtoSis(ch3PsdValVentana)
                    entFreCh1 = entropia(ch1PsdValVentana)
                    entFreCh2 = entropia(ch2PsdValVentana)
                    entFreCh3 = entropia(ch3PsdValVentana)

                    maxch2, inmaxch2 = maximo(ch2PsdValVentana)
                    #print("Ciego: máximo valor e indice", maxch2, inmaxch2)

                    maxch1, inmaxch1 = maximo(ch1PsdValVentana)
                    #print("Estomago: máximo valor e indice", maxch1, inmaxch1)

                    maxch3, inmaxch3 = maximo(ch3PsdValVentana)
                    #print("Ileum: máximo valor e indice", maxch3, inmaxch3)

                    caracteristicaVentana = np.hstack(
                        [ch1PsdValVentana, kurTiemy0, kurFreCh1, entTemy0, entFreCh1, maxch1, inmaxch1, varY0Tf,
                         meanY0Tf, stdY0Tf, medianY0Tf, varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Y0Tf, Q3Y0Tf,
                         Q1Ch1Tf, Q3Ch1Tf,
                         ch2PsdValVentana, kurTiemy1, kurFreCh2, entTemy1, entFreCh2, maxch2, inmaxch2, varY1Tf,
                         meanY1Tf, stdY1Tf, medianY1Tf, varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Y1Tf, Q3Y1Tf,
                         Q1Ch2Tf, Q3Ch2Tf,
                         ch3PsdValVentana, kurTiemy2, kurFreCh3, entTemy2, entFreCh3, maxch3, inmaxch3, varY2Tf,
                         meanY2Tf, stdY2Tf, medianY2Tf, varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Y2Tf, Q3Y2Tf,
                         Q1Ch3Tf, Q3Ch3Tf,
                         r, etiquetaSenal])
                    #print("caracteristicaVentana.shape")
                    #print(caracteristicaVentana.shape)
                    caracteristicasSenal = np.vstack((caracteristicasSenal, caracteristicaVentana))
                    carFreSenalCh1 = np.vstack((carFreSenalCh1, ch1HzPsd_values))
                    carFreSenalCh2 = np.vstack((carFreSenalCh2, ch2HzPsd_values))

                return caracteristicasSenal, carFreSenalCh1, carFreSenalCh2

            #

            df1 = self.df

            # Filtrado
            nomb = "C1"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            f, carFreSenalCh1, carFreSenalCh2 = extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
            # Acumula verticalmente las caracteristicas de todas las ratitas
            self.caracteristicasSenales = np.vstack([self.caracteristicasSenales, f])
            self.C1Ch1 = np.vstack([self.C1Ch1, carFreSenalCh1])
            self.C1Ch2 = np.vstack([self.C1Ch2, carFreSenalCh2])
            print("caracteristicasSenales")
            print(self.caracteristicasSenales.shape)
            print("C1Ch1")
            print(self.C1Ch1.shape)
            print("C1Ch2")
            print(self.C1Ch2.shape)
    #
    def plotCSV1time(self):
        resultado_st = "plotCSV1time" + "\n"
        self.resultado.setText(resultado_st)
        #C:\Users\Mozart\Mozart\QtDesigner
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/Users/Mozart/Mozart/QtDesigner')
        # Cambiar la ruta
        # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom, xnom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label=y0nom)
                # axs[0].set_ylim(yinf, ysup)
                axs[0].legend(frameon=False, fontsize=10)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)


                axs[1].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label=y2nom)
                # axs[2].set_ylim(yinf, ysup)
                axs[1].legend(frameon=False, fontsize=10)
                #axs[1].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[1].set_xlabel(xnom)
                axs[1].set_ylabel(y2nom)
                axs[1].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):
                # etapaFiltrado(df1, nomb)
                #print("etapaFiltrado")
                #print("Info df1")
                #print(df1.info())
                #print('\n' * 2)
                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()
                #print("Info df6")
                #print(df6.info())
                #print("\n")

                #print("Info df1=df3")
                #print(df3.info())
                #print('\n' * 2)

                #print("****Estadisticas  total****")
                # horizontal_stack = pd.concat([df3, df4], axis=1)
                #print(df3.describe(include=[np.number]))
                #print('\n' * 2)

                #print('Correlación Ch1, Ch2 y Ch3')
                #print(df3.corr())
                #print('\n' * 2)

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estómago", y1nom="Ciego", y2nom="Ciego",xnom="30720 muestras = 1 Min")
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)
                # Señales Estomago, Ciego, Ileum
                # plotDownSamp(y0filPaBaDoSam0, y1CifilPaBaDoSam1, y1filPaBaDoSam1, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="DS Estomago", y1nom="Ds Ciego", y2nom="Ds Ileum Filtro2")
                # Señales sensor Estomago y Ciego con frecuencia corte a 0.75 Hz
                # plotDownSamp(y0HzDoSamPaBaj, y0HzDoSamPaBaj, y1HzDoSamPaBaj, yinf, ysup, inicioDs, finDs, nombre,
                #            y0nom="DS Estomago 0Hz", y1nom="Ds Estomago 0Hz", y2nom="Ds Ileum 1Hz")


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 20-04-20
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Estómago: y0 ", y1nom="Est. y0 Hz", y2nom="Ciego: y1",xnom="120 muestras = 1 Min")

                return 1


            df1 = self.df

            # Filtrado
            nomb = "C1"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            nomb = "C1 Filtrada"
            a0=extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
    #
    def plotmeanPsdC1(self):
        #x=self.df['col1']
        resultado_st = "plotmeanPsdC1" + "\n"
        self.resultado.setText(resultado_st)
        nomb2="C1"
        #
        def estadisticaPotencia(potenciaEnFreqCh1, potenciaEnFreqCh2, nomEP):
            # estadisticaPotencia(caracteristicasFreqBasalCh1, caracteristicasFreqBasalCh2)
            # mediaPotenciaCh1 = np.median(potenciaEnFreqCh1, axis=0)
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            #print(" SSSSSSSS len(potenciaEnFreqCh1) SSSSSSSSSSSSSSSS")
            #print(len(potenciaEnFreqCh1))
            # stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            # stErrCh1=stdPotenciaCh1/(np.sqrt(len(stdPotenciaCh1)))
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))


            #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            # plt.plot(range(1, mediaPotenciaCh1.shape[0] + 1), mediaPotenciaCh1, "b")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width,color="blue", label='y0:Estómago', yerr=stErrCh1)
            rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width,color="r", label='y1:Ciego', yerr=stErrCh2)
            ax.set_ylabel('PA [uV**2] y stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz')
            #ax.set_xticks(x)
            ax.set_xscale('linear')

            ax.legend()
            plt.show()

            return 0
        #
        estadisticaPotencia(self.C1Ch1, self.C1Ch2, nomb2)

    ######### C2
    #
    def getCSV2(self):
        resultado_st = "getCSV2" + "\n"
        self.resultado.setText(resultado_st)

        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #
            def entropia(X):
                """Devuelve el valor de entropia de una muestra de datos"""
                probs = [np.mean(X == valor) for valor in set(X)]
                return round(np.sum(-p * np.log2(p) for p in probs), 3)
            #
            def curtoSis(y):
                curtosisY = kurtosis(y)
                return curtosisY
            #
            def estaBasica(y):
                estDesy = np.std(y, dtype=np.float64)
                meany = np.mean(y, dtype=np.float64)
                vary = np.var(y, dtype=np.float)
                medianY = np.median(y)
                Q1 = np.quantile(np.sort(y), 0.25)
                Q3 = np.quantile(np.sort(y), 0.75)
                return vary, meany, estDesy, medianY, Q1, Q3
            #
            def maximo(x):
                max = np.amax(x)
                inmax = np.argmax(x)
                return max, inmax
            #
            def pearsonCorr(x, y):
                corrpear = scipy.stats.pearsonr(x, y)
                """
                        The p-value roughly indicates the probability of an uncorrelated system
                        producing datasets that have a Pearson correlation at least as extreme
                        as the one computed from these datasets. The p-values are not entirely
                        reliable but are probably reasonable for datasets larger than 500 or so.
                        Parameters
                """

                return corrpear
            #
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values
            # def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
            #
            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label='PaBaja')
                # axs[0].set_ylim(yinf, ysup)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)

                axs[1].plot(x_valueDowSamy1, y1filPaBaDoSam1, linestyle='-', color='g', label='PaBaja')
                # axs[1].set_ylim(yinf, ysup)
                axs[1].set_xlim(x1, x2)
                axs[1].set_ylabel(y1nom)
                axs[1].grid(True)

                axs[2].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label='PaBaja')
                # axs[2].set_ylim(yinf, ysup)
                axs[2].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[2].set_ylabel(y2nom)
                axs[2].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):

                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                """
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")
                """
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 27-04-20
                """
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Est. y0 Hz", y1nom="Est. y0 Hz", y2nom="Ciego y1 Hz s/Ruido")
                """
                ######################################################
                # y0Guardar = y0filPaBaDoSam[0:longMenorHz]
                # y1Guardar = y1filPaBaDoSam[0:longMenorHz]
                # y0y1HzStack = np.stack((y0Guardar, y1Guardar), axis=-1)
                # guardarArchivo(y0y1HzStack, nombreExCa)



                # longMenor = np.min(np.array([longy0, longy1, longy2]))
                longMenor = np.min(np.array([longy0, longy1, longy2, longy0Hzsr, longy1Hzsr]))
                #print("longMenor: ", longMenor)
                # y0Guardar = y0filPaBaDoSam[0:longMenor]
                # y1Guardar = y1filPaBaDoSam[0:longMenor]
                # y2Guardar = y2filPaBaDoSam[0:longMenor]
                # y0y1y2Stack = np.stack((y0Guardar, y1Guardar, y2Guardar), axis=-1)
                # guardarArchivo(y0y1y2Stack, nombreExCa)

                ventanas0 = longMenor // 130
                #print("ventanas")
                #print(ventanas0)

                inicioDs = 0
                finDs = longMenor
                #print("len(y0filPaBaDoSam): ", len(y0filPaBaDoSam))
                #print("finDs", finDs)
                # Gráfica señales
                # plotDownSamp(y0filPaBaDoSam, y1filPaBaDoSam, y2filPaBaDoSam, yinf, ysup, inicioDs, finDs, nombreExCa,
                #            y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")


                #print("########### ExtractFeature ###############")
                # caracteristicasSenal = np.empty((0, 83))
                caracteristicasSenal = np.empty((0, 127))
                carFreSenalCh1 = np.empty((0, 129))
                carFreSenalCh2 = np.empty((0, 129))

                # for i in range(8):
                for i in range(ventanas0 - 1):
                    #print("i: ", i)
                    dt = 1
                    x_valueDowSam = np.arange(0, longMenor, dt)
                    inicioDs = 130 * i
                    finDs = inicioDs + 260

                    # Ventana de 2 minuto 10 segundos señal down sampled
                    # fs=2 hz         # 2 muestras por segundo
                    # para 60 segundos tengo 120 muestras

                    y0filPaBaDoSamTF = y0filPaBaDoSam[inicioDs:finDs]
                    y1filPaBaDoSamTF = y1filPaBaDoSam[inicioDs:finDs]
                    y2filPaBaDoSamTF = y2filPaBaDoSam[inicioDs:finDs]
                    y0HzfilPaBaDoSamTF = y0HzfilPaBaDoSam[inicioDs:finDs]
                    y1HzfilPaBaDoSamTF = y1HzfilPaBaDoSam[inicioDs:finDs]

                    # Se llama función plotDownSamp()
                    # plotDownSamp(y0filPaBaDoSam, y0filPaBaDoSam, y0filPaBaDoSamTF, yinf, ysup, inicioDs, finDs,
                    #            nombreExCa, y0nom="Estomago", y1nom="Estomago", y2nom="Estomago")

                    # plotDownSamp(y1filPaBaDoSam, y1filPaBaDoSam, y1filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #            y0nom="Ciego", y1nom="Ciego", y2nom="Ciego")

                    # plotDownSamp(y2filPaBaDoSam, y2filPaBaDoSam, y2filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #             y0nom="Ileum", y1nom="Ileum", y2nom="Ileum")

                    kurTiemy0 = curtoSis(y0filPaBaDoSamTF)
                    kurTiemy1 = curtoSis(y1filPaBaDoSamTF)
                    kurTiemy2 = curtoSis(y2filPaBaDoSamTF)

                    entTemy0 = entropia(y0filPaBaDoSamTF)
                    entTemy1 = entropia(y1filPaBaDoSamTF)
                    entTemy2 = entropia(y2filPaBaDoSamTF)

                    varY0Tf, meanY0Tf, stdY0Tf, medianY0Tf, Q1Y0Tf, Q3Y0Tf = estaBasica(y0filPaBaDoSamTF)
                    varY1Tf, meanY1Tf, stdY1Tf, medianY1Tf, Q1Y1Tf, Q3Y1Tf = estaBasica(y1filPaBaDoSamTF)
                    varY2Tf, meanY2Tf, stdY2Tf, medianY2Tf, Q1Y2Tf, Q3Y2Tf = estaBasica(y2filPaBaDoSamTF)

                    f_valuesch1, ch1Psd_values = get_psd_values(y0filPaBaDoSamTF, T, N, f_s)
                    f_valuesch2, ch2Psd_values = get_psd_values(y1filPaBaDoSamTF, T, N, f_s)
                    f_valuesch3, ch3Psd_values = get_psd_values(y2filPaBaDoSamTF, T, N, f_s)
                    # y1HzfilPaBaDoSamTF
                    f_valuesch1Hz, ch1HzPsd_values = get_psd_values(y0HzfilPaBaDoSamTF, T, N, f_s)
                    f_valuesch2Hz, ch2HzPsd_values = get_psd_values(y1HzfilPaBaDoSamTF, T, N, f_s)

                    #print("len(ch2HzPsd_values)")
                    #print(len(ch2HzPsd_values))

                    # plt.plot(f_valuesch1, ch1Psd_values, linestyle='-', color='black', label='Estomago')
                    # plt.plot(f_valuesch2, ch2Psd_values, linestyle='-', color='g', label='Ciego')
                    # plt.plot(f_valuesch3, ch3Psd_values, linestyle='-', color='r', label='Ileon')
                    # plt.plot(f_valuesch1Hz, ch1HzPsd_values, linestyle='-', color='blue', label='Ch1')
                    # plt.plot(f_valuesch2Hz, ch2HzPsd_values, linestyle='-', color='orange', label='Ch2')
                    # plt.xlabel('Frequencia [Hz]', fontsize=16)
                    # plt.ylabel('PSD [uV**2 / Hz]', fontsize=16)
                    # plt.title("Espectro Ciego, Estomago e Ileum", fontsize=16)
                    # plt.show()

                    r = pearsonCorr(ch1HzPsd_values, ch2HzPsd_values)

                    # Estomago
                    ch1PsdValVentana = ch1Psd_values[7:26]  # 12 datos
                    varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Ch1Tf, Q3Ch1Tf = estaBasica(ch1PsdValVentana)

                    # Ciego
                    # ch2PsdValVentana = ch2Psd_values[42:68] 26 datos
                    ch2PsdValVentana = ch2Psd_values[0:7]  # 8 datos
                    varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Ch2Tf, Q3Ch2Tf = estaBasica(ch2PsdValVentana)

                    # Ileum
                    ch3PsdValVentana = ch3Psd_values[40:84]
                    varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Ch3Tf, Q3Ch3Tf = estaBasica(ch3PsdValVentana)

                    kurFreCh1 = curtoSis(ch1PsdValVentana)
                    kurFreCh2 = curtoSis(ch2PsdValVentana)
                    kurFreCh3 = curtoSis(ch3PsdValVentana)
                    entFreCh1 = entropia(ch1PsdValVentana)
                    entFreCh2 = entropia(ch2PsdValVentana)
                    entFreCh3 = entropia(ch3PsdValVentana)

                    maxch2, inmaxch2 = maximo(ch2PsdValVentana)
                    #print("Ciego: máximo valor e indice", maxch2, inmaxch2)

                    maxch1, inmaxch1 = maximo(ch1PsdValVentana)
                    #print("Estomago: máximo valor e indice", maxch1, inmaxch1)

                    maxch3, inmaxch3 = maximo(ch3PsdValVentana)
                    #print("Ileum: máximo valor e indice", maxch3, inmaxch3)

                    caracteristicaVentana = np.hstack(
                        [ch1PsdValVentana, kurTiemy0, kurFreCh1, entTemy0, entFreCh1, maxch1, inmaxch1, varY0Tf,
                         meanY0Tf, stdY0Tf, medianY0Tf, varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Y0Tf, Q3Y0Tf,
                         Q1Ch1Tf, Q3Ch1Tf,
                         ch2PsdValVentana, kurTiemy1, kurFreCh2, entTemy1, entFreCh2, maxch2, inmaxch2, varY1Tf,
                         meanY1Tf, stdY1Tf, medianY1Tf, varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Y1Tf, Q3Y1Tf,
                         Q1Ch2Tf, Q3Ch2Tf,
                         ch3PsdValVentana, kurTiemy2, kurFreCh3, entTemy2, entFreCh3, maxch3, inmaxch3, varY2Tf,
                         meanY2Tf, stdY2Tf, medianY2Tf, varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Y2Tf, Q3Y2Tf,
                         Q1Ch3Tf, Q3Ch3Tf,
                         r, etiquetaSenal])
                    #print("caracteristicaVentana.shape")
                    #print(caracteristicaVentana.shape)
                    caracteristicasSenal = np.vstack((caracteristicasSenal, caracteristicaVentana))
                    carFreSenalCh1 = np.vstack((carFreSenalCh1, ch1HzPsd_values))
                    carFreSenalCh2 = np.vstack((carFreSenalCh2, ch2HzPsd_values))

                return caracteristicasSenal, carFreSenalCh1, carFreSenalCh2

            #

            df1 = self.df

            # Filtrado
            nomb = "C2"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            f, carFreSenalCh1, carFreSenalCh2 = extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
            # Acumula verticalmente las caracteristicas de todas las ratitas
            self.caracteristicasSenales = np.vstack([self.caracteristicasSenales, f])
            self.C2Ch1 = np.vstack([self.C2Ch1, carFreSenalCh1])
            self.C2Ch2 = np.vstack([self.C2Ch2, carFreSenalCh2])
            print("caracteristicasSenales")
            print(self.caracteristicasSenales.shape)
            print("C2Ch1")
            print(self.C2Ch1.shape)
            print("C2Ch2")
            print(self.C2Ch2.shape)
    #
    def plotCSV2time(self):
        resultado_st = "plotCSV2time" + "\n"
        self.resultado.setText(resultado_st)
        #C:\Users\Mozart\Mozart\QtDesigner
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/Users/Mozart/Mozart/QtDesigner')
        # Cambiar la ruta
        # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom, xnom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label=y0nom)
                # axs[0].set_ylim(yinf, ysup)
                axs[0].legend(frameon=False, fontsize=10)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)


                axs[1].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label=y2nom)
                # axs[2].set_ylim(yinf, ysup)
                axs[1].legend(frameon=False, fontsize=10)
                #axs[1].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[1].set_xlabel(xnom)
                axs[1].set_ylabel(y2nom)
                axs[1].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):
                # etapaFiltrado(df1, nomb)
                #print("etapaFiltrado")
                #print("Info df1")
                #print(df1.info())
                #print('\n' * 2)
                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()
                #print("Info df6")
                #print(df6.info())
                #print("\n")

                #print("Info df1=df3")
                #print(df3.info())
                #print('\n' * 2)

                #print("****Estadisticas  total****")
                # horizontal_stack = pd.concat([df3, df4], axis=1)
                #print(df3.describe(include=[np.number]))
                #print('\n' * 2)

                #print('Correlación Ch1, Ch2 y Ch3')
                #print(df3.corr())
                #print('\n' * 2)

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estómago", y1nom="Ciego", y2nom="Ciego",xnom="30720 muestras = 1 Min")
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)
                # Señales Estomago, Ciego, Ileum
                # plotDownSamp(y0filPaBaDoSam0, y1CifilPaBaDoSam1, y1filPaBaDoSam1, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="DS Estomago", y1nom="Ds Ciego", y2nom="Ds Ileum Filtro2")
                # Señales sensor Estomago y Ciego con frecuencia corte a 0.75 Hz
                # plotDownSamp(y0HzDoSamPaBaj, y0HzDoSamPaBaj, y1HzDoSamPaBaj, yinf, ysup, inicioDs, finDs, nombre,
                #            y0nom="DS Estomago 0Hz", y1nom="Ds Estomago 0Hz", y2nom="Ds Ileum 1Hz")


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 20-04-20
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Estómago: y0 ", y1nom="Est. y0 Hz", y2nom="Ciego: y1",xnom="120 muestras = 1 Min")

                return 1


            df1 = self.df

            # Filtrado
            nomb = "C2"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            nomb = "C2 Filtrada"
            a0=extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
    #
    def plotmeanPsdC2(self):
        #x=self.df['col1']
        resultado_st = "plotmeanPsdC2" + "\n"
        self.resultado.setText(resultado_st)
        nomb2="C2"
        #
        def estadisticaPotencia(potenciaEnFreqCh1, potenciaEnFreqCh2, nomEP):
            # estadisticaPotencia(caracteristicasFreqBasalCh1, caracteristicasFreqBasalCh2)
            # mediaPotenciaCh1 = np.median(potenciaEnFreqCh1, axis=0)
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            #print(" SSSSSSSS len(potenciaEnFreqCh1) SSSSSSSSSSSSSSSS")
            #print(len(potenciaEnFreqCh1))
            # stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            # stErrCh1=stdPotenciaCh1/(np.sqrt(len(stdPotenciaCh1)))
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))


            #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            # plt.plot(range(1, mediaPotenciaCh1.shape[0] + 1), mediaPotenciaCh1, "b")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width,color="blue", label='y0:Estómago', yerr=stErrCh1)
            rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width,color="r", label='y1:Ciego', yerr=stErrCh2)
            ax.set_ylabel('PA [uV**2] y stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz')
            #ax.set_xticks(x)
            ax.set_xscale('linear')

            ax.legend()
            plt.show()

            return 0
        #
        estadisticaPotencia(self.C2Ch1, self.C2Ch2, nomb2)
    #
    ##############C3
    #
    def getCSV3(self):
        resultado_st = "getCSV3" + "\n"
        self.resultado.setText(resultado_st)

        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #
            def entropia(X):
                """Devuelve el valor de entropia de una muestra de datos"""
                probs = [np.mean(X == valor) for valor in set(X)]
                return round(np.sum(-p * np.log2(p) for p in probs), 3)
            #
            def curtoSis(y):
                curtosisY = kurtosis(y)
                return curtosisY
            #
            def estaBasica(y):
                estDesy = np.std(y, dtype=np.float64)
                meany = np.mean(y, dtype=np.float64)
                vary = np.var(y, dtype=np.float)
                medianY = np.median(y)
                Q1 = np.quantile(np.sort(y), 0.25)
                Q3 = np.quantile(np.sort(y), 0.75)
                return vary, meany, estDesy, medianY, Q1, Q3
            #
            def maximo(x):
                max = np.amax(x)
                inmax = np.argmax(x)
                return max, inmax
            #
            def pearsonCorr(x, y):
                corrpear = scipy.stats.pearsonr(x, y)
                """
                        The p-value roughly indicates the probability of an uncorrelated system
                        producing datasets that have a Pearson correlation at least as extreme
                        as the one computed from these datasets. The p-values are not entirely
                        reliable but are probably reasonable for datasets larger than 500 or so.
                        Parameters
                """

                return corrpear
            #
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values
            # def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
            #
            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label='PaBaja')
                # axs[0].set_ylim(yinf, ysup)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)

                axs[1].plot(x_valueDowSamy1, y1filPaBaDoSam1, linestyle='-', color='g', label='PaBaja')
                # axs[1].set_ylim(yinf, ysup)
                axs[1].set_xlim(x1, x2)
                axs[1].set_ylabel(y1nom)
                axs[1].grid(True)

                axs[2].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label='PaBaja')
                # axs[2].set_ylim(yinf, ysup)
                axs[2].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[2].set_ylabel(y2nom)
                axs[2].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):

                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                """
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")
                """
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 27-04-20
                """
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Est. y0 Hz", y1nom="Est. y0 Hz", y2nom="Ciego y1 Hz s/Ruido")
                """
                ######################################################
                # y0Guardar = y0filPaBaDoSam[0:longMenorHz]
                # y1Guardar = y1filPaBaDoSam[0:longMenorHz]
                # y0y1HzStack = np.stack((y0Guardar, y1Guardar), axis=-1)
                # guardarArchivo(y0y1HzStack, nombreExCa)



                # longMenor = np.min(np.array([longy0, longy1, longy2]))
                longMenor = np.min(np.array([longy0, longy1, longy2, longy0Hzsr, longy1Hzsr]))
                #print("longMenor: ", longMenor)
                # y0Guardar = y0filPaBaDoSam[0:longMenor]
                # y1Guardar = y1filPaBaDoSam[0:longMenor]
                # y2Guardar = y2filPaBaDoSam[0:longMenor]
                # y0y1y2Stack = np.stack((y0Guardar, y1Guardar, y2Guardar), axis=-1)
                # guardarArchivo(y0y1y2Stack, nombreExCa)

                ventanas0 = longMenor // 130
                #print("ventanas")
                #print(ventanas0)

                inicioDs = 0
                finDs = longMenor
                #print("len(y0filPaBaDoSam): ", len(y0filPaBaDoSam))
                #print("finDs", finDs)
                # Gráfica señales
                # plotDownSamp(y0filPaBaDoSam, y1filPaBaDoSam, y2filPaBaDoSam, yinf, ysup, inicioDs, finDs, nombreExCa,
                #            y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")


                #print("########### ExtractFeature ###############")
                # caracteristicasSenal = np.empty((0, 83))
                caracteristicasSenal = np.empty((0, 127))
                carFreSenalCh1 = np.empty((0, 129))
                carFreSenalCh2 = np.empty((0, 129))

                # for i in range(8):
                for i in range(ventanas0 - 1):
                    #print("i: ", i)
                    dt = 1
                    x_valueDowSam = np.arange(0, longMenor, dt)
                    inicioDs = 130 * i
                    finDs = inicioDs + 260

                    # Ventana de 2 minuto 10 segundos señal down sampled
                    # fs=2 hz         # 2 muestras por segundo
                    # para 60 segundos tengo 120 muestras

                    y0filPaBaDoSamTF = y0filPaBaDoSam[inicioDs:finDs]
                    y1filPaBaDoSamTF = y1filPaBaDoSam[inicioDs:finDs]
                    y2filPaBaDoSamTF = y2filPaBaDoSam[inicioDs:finDs]
                    y0HzfilPaBaDoSamTF = y0HzfilPaBaDoSam[inicioDs:finDs]
                    y1HzfilPaBaDoSamTF = y1HzfilPaBaDoSam[inicioDs:finDs]

                    # Se llama función plotDownSamp()
                    # plotDownSamp(y0filPaBaDoSam, y0filPaBaDoSam, y0filPaBaDoSamTF, yinf, ysup, inicioDs, finDs,
                    #            nombreExCa, y0nom="Estomago", y1nom="Estomago", y2nom="Estomago")

                    # plotDownSamp(y1filPaBaDoSam, y1filPaBaDoSam, y1filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #            y0nom="Ciego", y1nom="Ciego", y2nom="Ciego")

                    # plotDownSamp(y2filPaBaDoSam, y2filPaBaDoSam, y2filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #             y0nom="Ileum", y1nom="Ileum", y2nom="Ileum")

                    kurTiemy0 = curtoSis(y0filPaBaDoSamTF)
                    kurTiemy1 = curtoSis(y1filPaBaDoSamTF)
                    kurTiemy2 = curtoSis(y2filPaBaDoSamTF)

                    entTemy0 = entropia(y0filPaBaDoSamTF)
                    entTemy1 = entropia(y1filPaBaDoSamTF)
                    entTemy2 = entropia(y2filPaBaDoSamTF)

                    varY0Tf, meanY0Tf, stdY0Tf, medianY0Tf, Q1Y0Tf, Q3Y0Tf = estaBasica(y0filPaBaDoSamTF)
                    varY1Tf, meanY1Tf, stdY1Tf, medianY1Tf, Q1Y1Tf, Q3Y1Tf = estaBasica(y1filPaBaDoSamTF)
                    varY2Tf, meanY2Tf, stdY2Tf, medianY2Tf, Q1Y2Tf, Q3Y2Tf = estaBasica(y2filPaBaDoSamTF)

                    f_valuesch1, ch1Psd_values = get_psd_values(y0filPaBaDoSamTF, T, N, f_s)
                    f_valuesch2, ch2Psd_values = get_psd_values(y1filPaBaDoSamTF, T, N, f_s)
                    f_valuesch3, ch3Psd_values = get_psd_values(y2filPaBaDoSamTF, T, N, f_s)
                    # y1HzfilPaBaDoSamTF
                    f_valuesch1Hz, ch1HzPsd_values = get_psd_values(y0HzfilPaBaDoSamTF, T, N, f_s)
                    f_valuesch2Hz, ch2HzPsd_values = get_psd_values(y1HzfilPaBaDoSamTF, T, N, f_s)

                    #print("len(ch2HzPsd_values)")
                    #print(len(ch2HzPsd_values))

                    # plt.plot(f_valuesch1, ch1Psd_values, linestyle='-', color='black', label='Estomago')
                    # plt.plot(f_valuesch2, ch2Psd_values, linestyle='-', color='g', label='Ciego')
                    # plt.plot(f_valuesch3, ch3Psd_values, linestyle='-', color='r', label='Ileon')
                    # plt.plot(f_valuesch1Hz, ch1HzPsd_values, linestyle='-', color='blue', label='Ch1')
                    # plt.plot(f_valuesch2Hz, ch2HzPsd_values, linestyle='-', color='orange', label='Ch2')
                    # plt.xlabel('Frequencia [Hz]', fontsize=16)
                    # plt.ylabel('PSD [uV**2 / Hz]', fontsize=16)
                    # plt.title("Espectro Ciego, Estomago e Ileum", fontsize=16)
                    # plt.show()

                    r = pearsonCorr(ch1HzPsd_values, ch2HzPsd_values)

                    # Estomago
                    ch1PsdValVentana = ch1Psd_values[7:26]  # 12 datos
                    varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Ch1Tf, Q3Ch1Tf = estaBasica(ch1PsdValVentana)

                    # Ciego
                    # ch2PsdValVentana = ch2Psd_values[42:68] 26 datos
                    ch2PsdValVentana = ch2Psd_values[0:7]  # 8 datos
                    varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Ch2Tf, Q3Ch2Tf = estaBasica(ch2PsdValVentana)

                    # Ileum
                    ch3PsdValVentana = ch3Psd_values[40:84]
                    varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Ch3Tf, Q3Ch3Tf = estaBasica(ch3PsdValVentana)

                    kurFreCh1 = curtoSis(ch1PsdValVentana)
                    kurFreCh2 = curtoSis(ch2PsdValVentana)
                    kurFreCh3 = curtoSis(ch3PsdValVentana)
                    entFreCh1 = entropia(ch1PsdValVentana)
                    entFreCh2 = entropia(ch2PsdValVentana)
                    entFreCh3 = entropia(ch3PsdValVentana)

                    maxch2, inmaxch2 = maximo(ch2PsdValVentana)
                    #print("Ciego: máximo valor e indice", maxch2, inmaxch2)

                    maxch1, inmaxch1 = maximo(ch1PsdValVentana)
                    #print("Estomago: máximo valor e indice", maxch1, inmaxch1)

                    maxch3, inmaxch3 = maximo(ch3PsdValVentana)
                    #print("Ileum: máximo valor e indice", maxch3, inmaxch3)

                    caracteristicaVentana = np.hstack(
                        [ch1PsdValVentana, kurTiemy0, kurFreCh1, entTemy0, entFreCh1, maxch1, inmaxch1, varY0Tf,
                         meanY0Tf, stdY0Tf, medianY0Tf, varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Y0Tf, Q3Y0Tf,
                         Q1Ch1Tf, Q3Ch1Tf,
                         ch2PsdValVentana, kurTiemy1, kurFreCh2, entTemy1, entFreCh2, maxch2, inmaxch2, varY1Tf,
                         meanY1Tf, stdY1Tf, medianY1Tf, varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Y1Tf, Q3Y1Tf,
                         Q1Ch2Tf, Q3Ch2Tf,
                         ch3PsdValVentana, kurTiemy2, kurFreCh3, entTemy2, entFreCh3, maxch3, inmaxch3, varY2Tf,
                         meanY2Tf, stdY2Tf, medianY2Tf, varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Y2Tf, Q3Y2Tf,
                         Q1Ch3Tf, Q3Ch3Tf,
                         r, etiquetaSenal])
                    #print("caracteristicaVentana.shape")
                    #print(caracteristicaVentana.shape)
                    caracteristicasSenal = np.vstack((caracteristicasSenal, caracteristicaVentana))
                    carFreSenalCh1 = np.vstack((carFreSenalCh1, ch1HzPsd_values))
                    carFreSenalCh2 = np.vstack((carFreSenalCh2, ch2HzPsd_values))

                return caracteristicasSenal, carFreSenalCh1, carFreSenalCh2

            #

            df1 = self.df

            # Filtrado
            nomb = "C3"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            f, carFreSenalCh1, carFreSenalCh2 = extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
            # Acumula verticalmente las caracteristicas de todas las ratitas
            self.caracteristicasSenales = np.vstack([self.caracteristicasSenales, f])
            self.C3Ch1 = np.vstack([self.C3Ch1, carFreSenalCh1])
            self.C3Ch2 = np.vstack([self.C3Ch2, carFreSenalCh2])
            print("caracteristicasSenales")
            print(self.caracteristicasSenales.shape)
            print("C3Ch1")
            print(self.C3Ch1.shape)
            print("C3Ch2")
            print(self.C3Ch2.shape)
    #
    def plotCSV3time(self):
        resultado_st = "plotCSV3time" + "\n"
        self.resultado.setText(resultado_st)
        #C:\Users\Mozart\Mozart\QtDesigner
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/Users/Mozart/Mozart/QtDesigner')
        # Cambiar la ruta
        # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom, xnom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label=y0nom)
                # axs[0].set_ylim(yinf, ysup)
                axs[0].legend(frameon=False, fontsize=10)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)


                axs[1].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label=y2nom)
                # axs[2].set_ylim(yinf, ysup)
                axs[1].legend(frameon=False, fontsize=10)
                #axs[1].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[1].set_xlabel(xnom)
                axs[1].set_ylabel(y2nom)
                axs[1].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):
                # etapaFiltrado(df1, nomb)
                #print("etapaFiltrado")
                #print("Info df1")
                #print(df1.info())
                #print('\n' * 2)
                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()
                #print("Info df6")
                #print(df6.info())
                #print("\n")

                #print("Info df1=df3")
                #print(df3.info())
                #print('\n' * 2)

                #print("****Estadisticas  total****")
                # horizontal_stack = pd.concat([df3, df4], axis=1)
                #print(df3.describe(include=[np.number]))
                #print('\n' * 2)

                #print('Correlación Ch1, Ch2 y Ch3')
                #print(df3.corr())
                #print('\n' * 2)

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estómago", y1nom="Ciego", y2nom="Ciego",xnom="30720 muestras = 1 Min")
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)
                # Señales Estomago, Ciego, Ileum
                # plotDownSamp(y0filPaBaDoSam0, y1CifilPaBaDoSam1, y1filPaBaDoSam1, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="DS Estomago", y1nom="Ds Ciego", y2nom="Ds Ileum Filtro2")
                # Señales sensor Estomago y Ciego con frecuencia corte a 0.75 Hz
                # plotDownSamp(y0HzDoSamPaBaj, y0HzDoSamPaBaj, y1HzDoSamPaBaj, yinf, ysup, inicioDs, finDs, nombre,
                #            y0nom="DS Estomago 0Hz", y1nom="Ds Estomago 0Hz", y2nom="Ds Ileum 1Hz")


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 20-04-20
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Estómago: y0 ", y1nom="Est. y0 Hz", y2nom="Ciego: y1",xnom="120 muestras = 1 Min")

                return 1


            df1 = self.df

            # Filtrado
            nomb = "C3"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            nomb = "C3 Filtrada"
            a0=extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
    #
    def plotmeanPsdC3(self):
        #x=self.df['col1']
        resultado_st = "plotmeanPSDC3" + "\n"
        self.resultado.setText(resultado_st)
        nomb2="C3"
        #
        def estadisticaPotencia(potenciaEnFreqCh1, potenciaEnFreqCh2, nomEP):
            # estadisticaPotencia(caracteristicasFreqBasalCh1, caracteristicasFreqBasalCh2)
            # mediaPotenciaCh1 = np.median(potenciaEnFreqCh1, axis=0)
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            #print(" SSSSSSSS len(potenciaEnFreqCh1) SSSSSSSSSSSSSSSS")
            #print(len(potenciaEnFreqCh1))
            # stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            # stErrCh1=stdPotenciaCh1/(np.sqrt(len(stdPotenciaCh1)))
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))


            #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            # plt.plot(range(1, mediaPotenciaCh1.shape[0] + 1), mediaPotenciaCh1, "b")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width,color="blue", label='y0:Estómago', yerr=stErrCh1)
            rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width,color="r", label='y1:Ciego', yerr=stErrCh2)
            ax.set_ylabel('PA [uV**2] y stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz')
            #ax.set_xticks(x)
            ax.set_xscale('linear')

            ax.legend()
            plt.show()

            return 0
        #
        estadisticaPotencia(self.C3Ch1, self.C3Ch2, nomb2)
    #
    ############## C4
    #
    def getCSV4(self):
        resultado_st = "getCSV4" + "\n"
        self.resultado.setText(resultado_st)

        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #
            def entropia(X):
                """Devuelve el valor de entropia de una muestra de datos"""
                probs = [np.mean(X == valor) for valor in set(X)]
                return round(np.sum(-p * np.log2(p) for p in probs), 3)
            #
            def curtoSis(y):
                curtosisY = kurtosis(y)
                return curtosisY
            #
            def estaBasica(y):
                estDesy = np.std(y, dtype=np.float64)
                meany = np.mean(y, dtype=np.float64)
                vary = np.var(y, dtype=np.float)
                medianY = np.median(y)
                Q1 = np.quantile(np.sort(y), 0.25)
                Q3 = np.quantile(np.sort(y), 0.75)
                return vary, meany, estDesy, medianY, Q1, Q3
            #
            def maximo(x):
                max = np.amax(x)
                inmax = np.argmax(x)
                return max, inmax
            #
            def pearsonCorr(x, y):
                corrpear = scipy.stats.pearsonr(x, y)
                """
                        The p-value roughly indicates the probability of an uncorrelated system
                        producing datasets that have a Pearson correlation at least as extreme
                        as the one computed from these datasets. The p-values are not entirely
                        reliable but are probably reasonable for datasets larger than 500 or so.
                        Parameters
                """

                return corrpear
            #
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values
            # def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
            #
            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label='PaBaja')
                # axs[0].set_ylim(yinf, ysup)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)

                axs[1].plot(x_valueDowSamy1, y1filPaBaDoSam1, linestyle='-', color='g', label='PaBaja')
                # axs[1].set_ylim(yinf, ysup)
                axs[1].set_xlim(x1, x2)
                axs[1].set_ylabel(y1nom)
                axs[1].grid(True)

                axs[2].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label='PaBaja')
                # axs[2].set_ylim(yinf, ysup)
                axs[2].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[2].set_ylabel(y2nom)
                axs[2].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):

                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                """
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")
                """
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 27-04-20
                """
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Est. y0 Hz", y1nom="Est. y0 Hz", y2nom="Ciego y1 Hz s/Ruido")
                """
                ######################################################
                # y0Guardar = y0filPaBaDoSam[0:longMenorHz]
                # y1Guardar = y1filPaBaDoSam[0:longMenorHz]
                # y0y1HzStack = np.stack((y0Guardar, y1Guardar), axis=-1)
                # guardarArchivo(y0y1HzStack, nombreExCa)



                # longMenor = np.min(np.array([longy0, longy1, longy2]))
                longMenor = np.min(np.array([longy0, longy1, longy2, longy0Hzsr, longy1Hzsr]))
                #print("longMenor: ", longMenor)
                # y0Guardar = y0filPaBaDoSam[0:longMenor]
                # y1Guardar = y1filPaBaDoSam[0:longMenor]
                # y2Guardar = y2filPaBaDoSam[0:longMenor]
                # y0y1y2Stack = np.stack((y0Guardar, y1Guardar, y2Guardar), axis=-1)
                # guardarArchivo(y0y1y2Stack, nombreExCa)

                ventanas0 = longMenor // 130
                #print("ventanas")
                #print(ventanas0)

                inicioDs = 0
                finDs = longMenor
                #print("len(y0filPaBaDoSam): ", len(y0filPaBaDoSam))
                #print("finDs", finDs)
                # Gráfica señales
                # plotDownSamp(y0filPaBaDoSam, y1filPaBaDoSam, y2filPaBaDoSam, yinf, ysup, inicioDs, finDs, nombreExCa,
                #            y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")


                #print("########### ExtractFeature ###############")
                # caracteristicasSenal = np.empty((0, 83))
                caracteristicasSenal = np.empty((0, 127))
                carFreSenalCh1 = np.empty((0, 129))
                carFreSenalCh2 = np.empty((0, 129))

                # for i in range(8):
                for i in range(ventanas0 - 1):
                    #print("i: ", i)
                    dt = 1
                    x_valueDowSam = np.arange(0, longMenor, dt)
                    inicioDs = 130 * i
                    finDs = inicioDs + 260

                    # Ventana de 2 minuto 10 segundos señal down sampled
                    # fs=2 hz         # 2 muestras por segundo
                    # para 60 segundos tengo 120 muestras

                    y0filPaBaDoSamTF = y0filPaBaDoSam[inicioDs:finDs]
                    y1filPaBaDoSamTF = y1filPaBaDoSam[inicioDs:finDs]
                    y2filPaBaDoSamTF = y2filPaBaDoSam[inicioDs:finDs]
                    y0HzfilPaBaDoSamTF = y0HzfilPaBaDoSam[inicioDs:finDs]
                    y1HzfilPaBaDoSamTF = y1HzfilPaBaDoSam[inicioDs:finDs]

                    # Se llama función plotDownSamp()
                    # plotDownSamp(y0filPaBaDoSam, y0filPaBaDoSam, y0filPaBaDoSamTF, yinf, ysup, inicioDs, finDs,
                    #            nombreExCa, y0nom="Estomago", y1nom="Estomago", y2nom="Estomago")

                    # plotDownSamp(y1filPaBaDoSam, y1filPaBaDoSam, y1filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #            y0nom="Ciego", y1nom="Ciego", y2nom="Ciego")

                    # plotDownSamp(y2filPaBaDoSam, y2filPaBaDoSam, y2filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #             y0nom="Ileum", y1nom="Ileum", y2nom="Ileum")

                    kurTiemy0 = curtoSis(y0filPaBaDoSamTF)
                    kurTiemy1 = curtoSis(y1filPaBaDoSamTF)
                    kurTiemy2 = curtoSis(y2filPaBaDoSamTF)

                    entTemy0 = entropia(y0filPaBaDoSamTF)
                    entTemy1 = entropia(y1filPaBaDoSamTF)
                    entTemy2 = entropia(y2filPaBaDoSamTF)

                    varY0Tf, meanY0Tf, stdY0Tf, medianY0Tf, Q1Y0Tf, Q3Y0Tf = estaBasica(y0filPaBaDoSamTF)
                    varY1Tf, meanY1Tf, stdY1Tf, medianY1Tf, Q1Y1Tf, Q3Y1Tf = estaBasica(y1filPaBaDoSamTF)
                    varY2Tf, meanY2Tf, stdY2Tf, medianY2Tf, Q1Y2Tf, Q3Y2Tf = estaBasica(y2filPaBaDoSamTF)

                    f_valuesch1, ch1Psd_values = get_psd_values(y0filPaBaDoSamTF, T, N, f_s)
                    f_valuesch2, ch2Psd_values = get_psd_values(y1filPaBaDoSamTF, T, N, f_s)
                    f_valuesch3, ch3Psd_values = get_psd_values(y2filPaBaDoSamTF, T, N, f_s)
                    # y1HzfilPaBaDoSamTF
                    f_valuesch1Hz, ch1HzPsd_values = get_psd_values(y0HzfilPaBaDoSamTF, T, N, f_s)
                    f_valuesch2Hz, ch2HzPsd_values = get_psd_values(y1HzfilPaBaDoSamTF, T, N, f_s)

                    #print("len(ch2HzPsd_values)")
                    #print(len(ch2HzPsd_values))

                    # plt.plot(f_valuesch1, ch1Psd_values, linestyle='-', color='black', label='Estomago')
                    # plt.plot(f_valuesch2, ch2Psd_values, linestyle='-', color='g', label='Ciego')
                    # plt.plot(f_valuesch3, ch3Psd_values, linestyle='-', color='r', label='Ileon')
                    # plt.plot(f_valuesch1Hz, ch1HzPsd_values, linestyle='-', color='blue', label='Ch1')
                    # plt.plot(f_valuesch2Hz, ch2HzPsd_values, linestyle='-', color='orange', label='Ch2')
                    # plt.xlabel('Frequencia [Hz]', fontsize=16)
                    # plt.ylabel('PSD [uV**2 / Hz]', fontsize=16)
                    # plt.title("Espectro Ciego, Estomago e Ileum", fontsize=16)
                    # plt.show()

                    r = pearsonCorr(ch1HzPsd_values, ch2HzPsd_values)

                    # Estomago
                    ch1PsdValVentana = ch1Psd_values[7:26]  # 12 datos
                    varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Ch1Tf, Q3Ch1Tf = estaBasica(ch1PsdValVentana)

                    # Ciego
                    # ch2PsdValVentana = ch2Psd_values[42:68] 26 datos
                    ch2PsdValVentana = ch2Psd_values[0:7]  # 8 datos
                    varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Ch2Tf, Q3Ch2Tf = estaBasica(ch2PsdValVentana)

                    # Ileum
                    ch3PsdValVentana = ch3Psd_values[40:84]
                    varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Ch3Tf, Q3Ch3Tf = estaBasica(ch3PsdValVentana)

                    kurFreCh1 = curtoSis(ch1PsdValVentana)
                    kurFreCh2 = curtoSis(ch2PsdValVentana)
                    kurFreCh3 = curtoSis(ch3PsdValVentana)
                    entFreCh1 = entropia(ch1PsdValVentana)
                    entFreCh2 = entropia(ch2PsdValVentana)
                    entFreCh3 = entropia(ch3PsdValVentana)

                    maxch2, inmaxch2 = maximo(ch2PsdValVentana)
                    #print("Ciego: máximo valor e indice", maxch2, inmaxch2)

                    maxch1, inmaxch1 = maximo(ch1PsdValVentana)
                    #print("Estomago: máximo valor e indice", maxch1, inmaxch1)

                    maxch3, inmaxch3 = maximo(ch3PsdValVentana)
                    #print("Ileum: máximo valor e indice", maxch3, inmaxch3)

                    caracteristicaVentana = np.hstack(
                        [ch1PsdValVentana, kurTiemy0, kurFreCh1, entTemy0, entFreCh1, maxch1, inmaxch1, varY0Tf,
                         meanY0Tf, stdY0Tf, medianY0Tf, varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Y0Tf, Q3Y0Tf,
                         Q1Ch1Tf, Q3Ch1Tf,
                         ch2PsdValVentana, kurTiemy1, kurFreCh2, entTemy1, entFreCh2, maxch2, inmaxch2, varY1Tf,
                         meanY1Tf, stdY1Tf, medianY1Tf, varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Y1Tf, Q3Y1Tf,
                         Q1Ch2Tf, Q3Ch2Tf,
                         ch3PsdValVentana, kurTiemy2, kurFreCh3, entTemy2, entFreCh3, maxch3, inmaxch3, varY2Tf,
                         meanY2Tf, stdY2Tf, medianY2Tf, varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Y2Tf, Q3Y2Tf,
                         Q1Ch3Tf, Q3Ch3Tf,
                         r, etiquetaSenal])
                    #print("caracteristicaVentana.shape")
                    #print(caracteristicaVentana.shape)
                    caracteristicasSenal = np.vstack((caracteristicasSenal, caracteristicaVentana))
                    carFreSenalCh1 = np.vstack((carFreSenalCh1, ch1HzPsd_values))
                    carFreSenalCh2 = np.vstack((carFreSenalCh2, ch2HzPsd_values))

                return caracteristicasSenal, carFreSenalCh1, carFreSenalCh2

            #

            df1 = self.df

            # Filtrado
            nomb = "C4"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            f, carFreSenalCh1, carFreSenalCh2 = extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
            # Acumula verticalmente las caracteristicas de todas las ratitas
            self.caracteristicasSenales = np.vstack([self.caracteristicasSenales, f])
            self.C4Ch1 = np.vstack([self.C4Ch1, carFreSenalCh1])
            self.C4Ch2 = np.vstack([self.C4Ch2, carFreSenalCh2])
            print("caracteristicasSenales")
            print(self.caracteristicasSenales.shape)
            print("C4Ch1")
            print(self.C4Ch1.shape)
            print("C4Ch2")
            print(self.C4Ch2.shape)
    #
    def plotCSV4time(self):
        resultado_st = "plotCSV4time" + "\n"
        self.resultado.setText(resultado_st)
        #C:\Users\Mozart\Mozart\QtDesigner
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/Users/Mozart/Mozart/QtDesigner')
        # Cambiar la ruta
        # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom, xnom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label=y0nom)
                # axs[0].set_ylim(yinf, ysup)
                axs[0].legend(frameon=False, fontsize=10)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)


                axs[1].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label=y2nom)
                # axs[2].set_ylim(yinf, ysup)
                axs[1].legend(frameon=False, fontsize=10)
                #axs[1].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[1].set_xlabel(xnom)
                axs[1].set_ylabel(y2nom)
                axs[1].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):
                # etapaFiltrado(df1, nomb)
                #print("etapaFiltrado")
                #print("Info df1")
                #print(df1.info())
                #print('\n' * 2)
                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()
                #print("Info df6")
                #print(df6.info())
                #print("\n")

                #print("Info df1=df3")
                #print(df3.info())
                #print('\n' * 2)

                #print("****Estadisticas  total****")
                # horizontal_stack = pd.concat([df3, df4], axis=1)
                #print(df3.describe(include=[np.number]))
                #print('\n' * 2)

                #print('Correlación Ch1, Ch2 y Ch3')
                #print(df3.corr())
                #print('\n' * 2)

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estómago", y1nom="Ciego", y2nom="Ciego",xnom="30720 muestras = 1 Min")
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)
                # Señales Estomago, Ciego, Ileum
                # plotDownSamp(y0filPaBaDoSam0, y1CifilPaBaDoSam1, y1filPaBaDoSam1, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="DS Estomago", y1nom="Ds Ciego", y2nom="Ds Ileum Filtro2")
                # Señales sensor Estomago y Ciego con frecuencia corte a 0.75 Hz
                # plotDownSamp(y0HzDoSamPaBaj, y0HzDoSamPaBaj, y1HzDoSamPaBaj, yinf, ysup, inicioDs, finDs, nombre,
                #            y0nom="DS Estomago 0Hz", y1nom="Ds Estomago 0Hz", y2nom="Ds Ileum 1Hz")


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 20-04-20
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Estómago: y0 ", y1nom="Est. y0 Hz", y2nom="Ciego: y1",xnom="120 muestras = 1 Min")

                return 1


            df1 = self.df

            # Filtrado
            nomb = "C4"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            nomb = "C4 Filtrada"
            a0=extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
    #
    def plotmeanPsdC4(self):
        #x=self.df['col1']
        resultado_st = "plotmeanPsdC4" + "\n"
        self.resultado.setText(resultado_st)
        nomb2="C4"
        #
        def estadisticaPotencia(potenciaEnFreqCh1, potenciaEnFreqCh2, nomEP):
            # estadisticaPotencia(caracteristicasFreqBasalCh1, caracteristicasFreqBasalCh2)
            # mediaPotenciaCh1 = np.median(potenciaEnFreqCh1, axis=0)
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            #print(" SSSSSSSS len(potenciaEnFreqCh1) SSSSSSSSSSSSSSSS")
            #print(len(potenciaEnFreqCh1))
            # stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            # stErrCh1=stdPotenciaCh1/(np.sqrt(len(stdPotenciaCh1)))
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))


            #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            # plt.plot(range(1, mediaPotenciaCh1.shape[0] + 1), mediaPotenciaCh1, "b")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width,color="blue", label='y0:Estómago', yerr=stErrCh1)
            rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width,color="r", label='y1:Ciego', yerr=stErrCh2)
            ax.set_ylabel('PA [uV**2] y stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz')
            #ax.set_xticks(x)
            ax.set_xscale('linear')

            ax.legend()
            plt.show()

            return 0
        #
        estadisticaPotencia(self.C4Ch1, self.C4Ch2, nomb2)
    #
    ##########################
    #mean Psd 4 condiciones y0
    #
    def plotPSDch1(self):

        #
        resultado_st = "plotmeanPSDch1" + "\n"
        self.resultado.setText(resultado_st)
        def estadisticaPotencia3Gps(potenciaEnFreqCh1, potenciaEnFreqCh2, potenciaEnFreqCh3, potenciaEnFreqCh4, nomEP):
            # https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)
            mediaPotenciaCh3 = np.mean(potenciaEnFreqCh3, axis=0, dtype=np.float64)
            mediaPotenciaCh4 = np.mean(potenciaEnFreqCh4, axis=0, dtype=np.float64)

            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)
            stdPotenciaCh3 = np.std(potenciaEnFreqCh3, axis=0, dtype=np.float64)
            stdPotenciaCh4 = np.std(potenciaEnFreqCh4, axis=0, dtype=np.float64)

            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))
            stErrCh3 = stdPotenciaCh3 / (np.sqrt(len(potenciaEnFreqCh3)))
            stErrCh4 = stdPotenciaCh4 / (np.sqrt(len(potenciaEnFreqCh4)))

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "b", label='C1')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='C2')
            plt.plot(x * 0.007813, mediaPotenciaCh3, "g", label='C3')
            plt.plot(x * 0.007813, mediaPotenciaCh4, "orange", label='C4')

            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.25  # the width of the bars
            fig, ax = plt.subplots()
            #rects1 = ax.bar(x - width, mediaPotenciaCh1, width, label='C1', yerr=stErrCh1)
            rects1 = ax.bar(x - 1.45*width, mediaPotenciaCh1, width-0.03, label='C1', yerr=stErrCh1)
            #rects2 = ax.bar(x, mediaPotenciaCh2, width, label='C2', yerr=stErrCh2)
            rects2 = ax.bar(x-width/2, mediaPotenciaCh2, width, label='C2', yerr=stErrCh2)
            #rects3 = ax.bar(x + width, mediaPotenciaCh3, width, label='C3', yerr=stErrCh3)
            rects3 = ax.bar(x+width/2, mediaPotenciaCh3, width, label='C3', yerr=stErrCh3)
            #Se agrega rects 4
            rects4 = ax.bar(x + width*1.5, mediaPotenciaCh4, width, label='C4', yerr=stErrCh4)

            ax.set_ylabel('PA [uV**2], stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz.')
            #ax.set_xticks(x)
            ax.set_xscale('linear')
            ax.legend()
            plt.show()

            return 0
        #
        nomEP = "C1, C2, C3, C4, y0"
        estadisticaPotencia3Gps(self.C1Ch1, self.C2Ch1, self.C3Ch1, self.C4Ch1, nomEP)
    #
    # mean Psd 4 condiciones y1
    def plotPSDch2(self):
        #
        resultado_st = "plotmeanPSDch2" + "\n"
        self.resultado.setText(resultado_st)
        def estadisticaPotencia3Gps(potenciaEnFreqCh1, potenciaEnFreqCh2, potenciaEnFreqCh3, potenciaEnFreqCh4, nomEP):
            # https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)
            mediaPotenciaCh3 = np.mean(potenciaEnFreqCh3, axis=0, dtype=np.float64)
            mediaPotenciaCh4 = np.mean(potenciaEnFreqCh4, axis=0, dtype=np.float64)

            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)
            stdPotenciaCh3 = np.std(potenciaEnFreqCh3, axis=0, dtype=np.float64)
            stdPotenciaCh4 = np.std(potenciaEnFreqCh4, axis=0, dtype=np.float64)

            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))
            stErrCh3 = stdPotenciaCh3 / (np.sqrt(len(potenciaEnFreqCh3)))
            stErrCh4 = stdPotenciaCh4 / (np.sqrt(len(potenciaEnFreqCh4)))

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "b", label='C1')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='C2')
            plt.plot(x * 0.007813, mediaPotenciaCh3, "g", label='C3')
            plt.plot(x * 0.007813, mediaPotenciaCh4, "orange", label='C4')

            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.25  # the width of the bars
            fig, ax = plt.subplots()
            #rects1 = ax.bar(x - width, mediaPotenciaCh1, width, label='C1', yerr=stErrCh1)
            rects1 = ax.bar(x - 1.45*width, mediaPotenciaCh1, width-0.03, label='C1', yerr=stErrCh1)
            #rects2 = ax.bar(x, mediaPotenciaCh2, width, label='C2', yerr=stErrCh2)
            rects2 = ax.bar(x-width/2, mediaPotenciaCh2, width, label='C2', yerr=stErrCh2)
            #rects3 = ax.bar(x + width, mediaPotenciaCh3, width, label='C3', yerr=stErrCh3)
            rects3 = ax.bar(x+width/2, mediaPotenciaCh3, width, label='C3', yerr=stErrCh3)
            #Se agrega rects 4
            rects4 = ax.bar(x + width*1.5, mediaPotenciaCh4, width, label='C4', yerr=stErrCh4)

            ax.set_ylabel('PA [uV**2], stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz.')
            #ax.set_xticks(x)
            ax.set_xscale('linear')
            ax.legend()
            plt.show()

            return 0
        #
        nomEP = "C1, C2, C3, C4, y1"
        estadisticaPotencia3Gps(self.C1Ch2, self.C2Ch2, self.C3Ch2, self.C4Ch2, nomEP)


    # Restablece memorias
    def resC1(self):
        resultado_st = "resC1" + "\n"
        self.resultado.setText(resultado_st)

        self.C1Ch1 = np.empty((0, 129))
        print("C1Ch1")
        print(self.C1Ch1)

        self.C1Ch2 = np.empty((0, 129))
        print("C1Ch2")
        print(self.C1Ch2)

    def resC2(self):
        resultado_st = "resC2" + "\n"
        self.resultado.setText(resultado_st)

        self.C2Ch1 = np.empty((0, 129))
        print("C2Ch1")
        print(self.C2Ch1)

        self.C2Ch2 = np.empty((0, 129))
        print("C2Ch2")
        print(self.C2Ch2)

    def resC3(self):
        resultado_st = "resC3" + "\n"
        self.resultado.setText(resultado_st)

        self.C3Ch1 = np.empty((0, 129))
        print("C3Ch1")
        print(self.C3Ch1)

        self.C3Ch2 = np.empty((0, 129))
        print("C3Ch2")
        print(self.C3Ch2)

    def resC4(self):
        resultado_st = "resC4" + "\n"
        self.resultado.setText(resultado_st)

        self.C4Ch1 = np.empty((0, 129))
        print("C4Ch1")
        print(self.C4Ch1)

        self.C4Ch2 = np.empty((0, 129))
        print("C4Ch2")
        print(self.C4Ch2)



    ###############
    def abrirVentanaPrincipal(self):
        self.parent().show()
        self.close()


class ventanaCin(QtWidgets.QMainWindow, Ui_cinco):
#class ventanaCin(QMainWindow):
    def __init__(self, parent=None):
        super(ventanaCin, self).__init__(parent)
        #loadUi('cinco.ui', self)
        Ui_cinco.__init__(self)
        self.setupUi(self)
        self.boton6_4.clicked.connect(self.abrirVentanaPrincipal)


        #6May20
        # Botones Con1
        self.boton1.clicked.connect(self.getCSV1)
        self.boton1_1.clicked.connect(self.plotCSV1time)
        self.boton1_2.clicked.connect(self.plotmeanPsdC1)

        # Aquí va el botón Palm
        self.boton3.clicked.connect(self.getCSV2)
        self.boton3_1.clicked.connect(self.plotCSV2time)
        self.boton3_2.clicked.connect(self.plotmeanPsdC2)

        # Aquí va el botón Pesc
        self.boton4.clicked.connect(self.getCSV3)
        self.boton4_1.clicked.connect(self.plotCSV3time)
        self.boton4_2.clicked.connect(self.plotmeanPsdC3)

        # Botón condición C4
        self.boton5.clicked.connect(self.getCSV4)
        self.boton5_1.clicked.connect(self.plotCSV4time)
        self.boton5_2.clicked.connect(self.plotmeanPsdC4)

        # Botón condición C5
        self.boton6.clicked.connect(self.getCSV5)
        self.boton6_1.clicked.connect(self.plotCSV5time)
        self.boton6_2.clicked.connect(self.plotmeanPsdC5)

        # Botón comparación 5 señales ch1
        self.boton2.clicked.connect(self.plotPSDch1)
        # Botón comparación 5 señales ch2
        self.boton2_1.clicked.connect(self.plotPSDch2)

        # Boton reiniciar captura archivos ctr
        self.boton1_3.clicked.connect(self.resC1)
        # Boton reiniciar captura archivos Palma
        self.boton3_3.clicked.connect(self.resC2)
        # Boton reiniciar captura archivos Pescado
        self.boton4_3.clicked.connect(self.resC3)
        # Boton reiniciar captura archivos C4
        self.boton5_3.clicked.connect(self.resC4)
        # Boton reiniciar captura archivos C5
        self.boton6_3.clicked.connect(self.resC5)


        # Matrices archivo general de caracteristicas
        self.caracteristicasSenales = np.empty((0, 127))

        # Matrices archivo CSV Grupo1 Ctr
        self.C1Ch1 = np.empty((0, 129))
        self.C1Ch2 = np.empty((0, 129))

        # Matrices archivo CSV Grupo2 Palm
        self.C2Ch1 = np.empty((0, 129))
        self.C2Ch2 = np.empty((0, 129))

        # Matrices archivo CSV Grupo3
        self.C3Ch1 = np.empty((0, 129))
        self.C3Ch2 = np.empty((0, 129))

        # Matrices archivo CSV Grupo4
        self.C4Ch1 = np.empty((0, 129))
        self.C4Ch2 = np.empty((0, 129))

        # Matrices archivo CSV Grupo5
        self.C5Ch1 = np.empty((0, 129))
        self.C5Ch2 = np.empty((0, 129))

        #
    #C1
    def getCSV1(self):
        resultado_st = "getCSV1" + "\n"
        self.resultado.setText(resultado_st)

        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #
            def entropia(X):
                """Devuelve el valor de entropia de una muestra de datos"""
                probs = [np.mean(X == valor) for valor in set(X)]
                return round(np.sum(-p * np.log2(p) for p in probs), 3)
            #
            def curtoSis(y):
                curtosisY = kurtosis(y)
                return curtosisY
            #
            def estaBasica(y):
                estDesy = np.std(y, dtype=np.float64)
                meany = np.mean(y, dtype=np.float64)
                vary = np.var(y, dtype=np.float)
                medianY = np.median(y)
                Q1 = np.quantile(np.sort(y), 0.25)
                Q3 = np.quantile(np.sort(y), 0.75)
                return vary, meany, estDesy, medianY, Q1, Q3
            #
            def maximo(x):
                max = np.amax(x)
                inmax = np.argmax(x)
                return max, inmax
            #
            def pearsonCorr(x, y):
                corrpear = scipy.stats.pearsonr(x, y)
                """
                        The p-value roughly indicates the probability of an uncorrelated system
                        producing datasets that have a Pearson correlation at least as extreme
                        as the one computed from these datasets. The p-values are not entirely
                        reliable but are probably reasonable for datasets larger than 500 or so.
                        Parameters
                """

                return corrpear
            #
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values
            # def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
            #
            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label='PaBaja')
                # axs[0].set_ylim(yinf, ysup)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)

                axs[1].plot(x_valueDowSamy1, y1filPaBaDoSam1, linestyle='-', color='g', label='PaBaja')
                # axs[1].set_ylim(yinf, ysup)
                axs[1].set_xlim(x1, x2)
                axs[1].set_ylabel(y1nom)
                axs[1].grid(True)

                axs[2].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label='PaBaja')
                # axs[2].set_ylim(yinf, ysup)
                axs[2].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[2].set_ylabel(y2nom)
                axs[2].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):

                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                """
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")
                """
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 27-04-20
                """
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Est. y0 Hz", y1nom="Est. y0 Hz", y2nom="Ciego y1 Hz s/Ruido")
                """
                ######################################################
                # y0Guardar = y0filPaBaDoSam[0:longMenorHz]
                # y1Guardar = y1filPaBaDoSam[0:longMenorHz]
                # y0y1HzStack = np.stack((y0Guardar, y1Guardar), axis=-1)
                # guardarArchivo(y0y1HzStack, nombreExCa)



                # longMenor = np.min(np.array([longy0, longy1, longy2]))
                longMenor = np.min(np.array([longy0, longy1, longy2, longy0Hzsr, longy1Hzsr]))
                #print("longMenor: ", longMenor)
                # y0Guardar = y0filPaBaDoSam[0:longMenor]
                # y1Guardar = y1filPaBaDoSam[0:longMenor]
                # y2Guardar = y2filPaBaDoSam[0:longMenor]
                # y0y1y2Stack = np.stack((y0Guardar, y1Guardar, y2Guardar), axis=-1)
                # guardarArchivo(y0y1y2Stack, nombreExCa)

                ventanas0 = longMenor // 130
                #print("ventanas")
                #print(ventanas0)

                inicioDs = 0
                finDs = longMenor
                #print("len(y0filPaBaDoSam): ", len(y0filPaBaDoSam))
                #print("finDs", finDs)
                # Gráfica señales
                # plotDownSamp(y0filPaBaDoSam, y1filPaBaDoSam, y2filPaBaDoSam, yinf, ysup, inicioDs, finDs, nombreExCa,
                #            y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")


                #print("########### ExtractFeature ###############")
                # caracteristicasSenal = np.empty((0, 83))
                caracteristicasSenal = np.empty((0, 127))
                carFreSenalCh1 = np.empty((0, 129))
                carFreSenalCh2 = np.empty((0, 129))

                # for i in range(8):
                for i in range(ventanas0 - 1):
                    #print("i: ", i)
                    dt = 1
                    x_valueDowSam = np.arange(0, longMenor, dt)
                    inicioDs = 130 * i
                    finDs = inicioDs + 260

                    # Ventana de 2 minuto 10 segundos señal down sampled
                    # fs=2 hz         # 2 muestras por segundo
                    # para 60 segundos tengo 120 muestras

                    y0filPaBaDoSamTF = y0filPaBaDoSam[inicioDs:finDs]
                    y1filPaBaDoSamTF = y1filPaBaDoSam[inicioDs:finDs]
                    y2filPaBaDoSamTF = y2filPaBaDoSam[inicioDs:finDs]
                    y0HzfilPaBaDoSamTF = y0HzfilPaBaDoSam[inicioDs:finDs]
                    y1HzfilPaBaDoSamTF = y1HzfilPaBaDoSam[inicioDs:finDs]

                    # Se llama función plotDownSamp()
                    # plotDownSamp(y0filPaBaDoSam, y0filPaBaDoSam, y0filPaBaDoSamTF, yinf, ysup, inicioDs, finDs,
                    #            nombreExCa, y0nom="Estomago", y1nom="Estomago", y2nom="Estomago")

                    # plotDownSamp(y1filPaBaDoSam, y1filPaBaDoSam, y1filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #            y0nom="Ciego", y1nom="Ciego", y2nom="Ciego")

                    # plotDownSamp(y2filPaBaDoSam, y2filPaBaDoSam, y2filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #             y0nom="Ileum", y1nom="Ileum", y2nom="Ileum")

                    kurTiemy0 = curtoSis(y0filPaBaDoSamTF)
                    kurTiemy1 = curtoSis(y1filPaBaDoSamTF)
                    kurTiemy2 = curtoSis(y2filPaBaDoSamTF)

                    entTemy0 = entropia(y0filPaBaDoSamTF)
                    entTemy1 = entropia(y1filPaBaDoSamTF)
                    entTemy2 = entropia(y2filPaBaDoSamTF)

                    varY0Tf, meanY0Tf, stdY0Tf, medianY0Tf, Q1Y0Tf, Q3Y0Tf = estaBasica(y0filPaBaDoSamTF)
                    varY1Tf, meanY1Tf, stdY1Tf, medianY1Tf, Q1Y1Tf, Q3Y1Tf = estaBasica(y1filPaBaDoSamTF)
                    varY2Tf, meanY2Tf, stdY2Tf, medianY2Tf, Q1Y2Tf, Q3Y2Tf = estaBasica(y2filPaBaDoSamTF)

                    f_valuesch1, ch1Psd_values = get_psd_values(y0filPaBaDoSamTF, T, N, f_s)
                    f_valuesch2, ch2Psd_values = get_psd_values(y1filPaBaDoSamTF, T, N, f_s)
                    f_valuesch3, ch3Psd_values = get_psd_values(y2filPaBaDoSamTF, T, N, f_s)
                    # y1HzfilPaBaDoSamTF
                    f_valuesch1Hz, ch1HzPsd_values = get_psd_values(y0HzfilPaBaDoSamTF, T, N, f_s)
                    f_valuesch2Hz, ch2HzPsd_values = get_psd_values(y1HzfilPaBaDoSamTF, T, N, f_s)

                    #print("len(ch2HzPsd_values)")
                    #print(len(ch2HzPsd_values))

                    # plt.plot(f_valuesch1, ch1Psd_values, linestyle='-', color='black', label='Estomago')
                    # plt.plot(f_valuesch2, ch2Psd_values, linestyle='-', color='g', label='Ciego')
                    # plt.plot(f_valuesch3, ch3Psd_values, linestyle='-', color='r', label='Ileon')
                    # plt.plot(f_valuesch1Hz, ch1HzPsd_values, linestyle='-', color='blue', label='Ch1')
                    # plt.plot(f_valuesch2Hz, ch2HzPsd_values, linestyle='-', color='orange', label='Ch2')
                    # plt.xlabel('Frequencia [Hz]', fontsize=16)
                    # plt.ylabel('PSD [uV**2 / Hz]', fontsize=16)
                    # plt.title("Espectro Ciego, Estomago e Ileum", fontsize=16)
                    # plt.show()

                    r = pearsonCorr(ch1HzPsd_values, ch2HzPsd_values)

                    # Estomago
                    ch1PsdValVentana = ch1Psd_values[7:26]  # 12 datos
                    varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Ch1Tf, Q3Ch1Tf = estaBasica(ch1PsdValVentana)

                    # Ciego
                    # ch2PsdValVentana = ch2Psd_values[42:68] 26 datos
                    ch2PsdValVentana = ch2Psd_values[0:7]  # 8 datos
                    varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Ch2Tf, Q3Ch2Tf = estaBasica(ch2PsdValVentana)

                    # Ileum
                    ch3PsdValVentana = ch3Psd_values[40:84]
                    varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Ch3Tf, Q3Ch3Tf = estaBasica(ch3PsdValVentana)

                    kurFreCh1 = curtoSis(ch1PsdValVentana)
                    kurFreCh2 = curtoSis(ch2PsdValVentana)
                    kurFreCh3 = curtoSis(ch3PsdValVentana)
                    entFreCh1 = entropia(ch1PsdValVentana)
                    entFreCh2 = entropia(ch2PsdValVentana)
                    entFreCh3 = entropia(ch3PsdValVentana)

                    maxch2, inmaxch2 = maximo(ch2PsdValVentana)
                    #print("Ciego: máximo valor e indice", maxch2, inmaxch2)

                    maxch1, inmaxch1 = maximo(ch1PsdValVentana)
                    #print("Estomago: máximo valor e indice", maxch1, inmaxch1)

                    maxch3, inmaxch3 = maximo(ch3PsdValVentana)
                    #print("Ileum: máximo valor e indice", maxch3, inmaxch3)

                    caracteristicaVentana = np.hstack(
                        [ch1PsdValVentana, kurTiemy0, kurFreCh1, entTemy0, entFreCh1, maxch1, inmaxch1, varY0Tf,
                         meanY0Tf, stdY0Tf, medianY0Tf, varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Y0Tf, Q3Y0Tf,
                         Q1Ch1Tf, Q3Ch1Tf,
                         ch2PsdValVentana, kurTiemy1, kurFreCh2, entTemy1, entFreCh2, maxch2, inmaxch2, varY1Tf,
                         meanY1Tf, stdY1Tf, medianY1Tf, varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Y1Tf, Q3Y1Tf,
                         Q1Ch2Tf, Q3Ch2Tf,
                         ch3PsdValVentana, kurTiemy2, kurFreCh3, entTemy2, entFreCh3, maxch3, inmaxch3, varY2Tf,
                         meanY2Tf, stdY2Tf, medianY2Tf, varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Y2Tf, Q3Y2Tf,
                         Q1Ch3Tf, Q3Ch3Tf,
                         r, etiquetaSenal])
                    #print("caracteristicaVentana.shape")
                    #print(caracteristicaVentana.shape)
                    caracteristicasSenal = np.vstack((caracteristicasSenal, caracteristicaVentana))
                    carFreSenalCh1 = np.vstack((carFreSenalCh1, ch1HzPsd_values))
                    carFreSenalCh2 = np.vstack((carFreSenalCh2, ch2HzPsd_values))

                return caracteristicasSenal, carFreSenalCh1, carFreSenalCh2

            #

            df1 = self.df

            # Filtrado
            nomb = "C1"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            f, carFreSenalCh1, carFreSenalCh2 = extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
            # Acumula verticalmente las caracteristicas de todas las ratitas
            self.caracteristicasSenales = np.vstack([self.caracteristicasSenales, f])
            self.C1Ch1 = np.vstack([self.C1Ch1, carFreSenalCh1])
            self.C1Ch2 = np.vstack([self.C1Ch2, carFreSenalCh2])
            print("caracteristicasSenales")
            print(self.caracteristicasSenales.shape)
            print("C1Ch1")
            print(self.C1Ch1.shape)
            print("C1Ch2")
            print(self.C1Ch2.shape)
    #
    def plotCSV1time(self):
        resultado_st = "plotCSV1time" + "\n"
        self.resultado.setText(resultado_st)
        #C:\Users\Mozart\Mozart\QtDesigner
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/Users/Mozart/Mozart/QtDesigner')
        # Cambiar la ruta
        # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom, xnom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label=y0nom)
                # axs[0].set_ylim(yinf, ysup)
                axs[0].legend(frameon=False, fontsize=10)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)


                axs[1].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label=y2nom)
                # axs[2].set_ylim(yinf, ysup)
                axs[1].legend(frameon=False, fontsize=10)
                #axs[1].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[1].set_xlabel(xnom)
                axs[1].set_ylabel(y2nom)
                axs[1].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):
                # etapaFiltrado(df1, nomb)
                #print("etapaFiltrado")
                #print("Info df1")
                #print(df1.info())
                #print('\n' * 2)
                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()
                #print("Info df6")
                #print(df6.info())
                #print("\n")

                #print("Info df1=df3")
                #print(df3.info())
                #print('\n' * 2)

                #print("****Estadisticas  total****")
                # horizontal_stack = pd.concat([df3, df4], axis=1)
                #print(df3.describe(include=[np.number]))
                #print('\n' * 2)

                #print('Correlación Ch1, Ch2 y Ch3')
                #print(df3.corr())
                #print('\n' * 2)

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estómago", y1nom="Ciego", y2nom="Ciego",xnom="30720 muestras = 1 Min")
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)
                # Señales Estomago, Ciego, Ileum
                # plotDownSamp(y0filPaBaDoSam0, y1CifilPaBaDoSam1, y1filPaBaDoSam1, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="DS Estomago", y1nom="Ds Ciego", y2nom="Ds Ileum Filtro2")
                # Señales sensor Estomago y Ciego con frecuencia corte a 0.75 Hz
                # plotDownSamp(y0HzDoSamPaBaj, y0HzDoSamPaBaj, y1HzDoSamPaBaj, yinf, ysup, inicioDs, finDs, nombre,
                #            y0nom="DS Estomago 0Hz", y1nom="Ds Estomago 0Hz", y2nom="Ds Ileum 1Hz")


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 20-04-20
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Estómago: y0 ", y1nom="Est. y0 Hz", y2nom="Ciego: y1",xnom="120 muestras = 1 Min")

                return 1


            df1 = self.df

            # Filtrado
            nomb = "C1"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            nomb = "C1 Filtrada"
            a0=extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
    #
    def plotmeanPsdC1(self):
        #x=self.df['col1']
        resultado_st = "plotmeanPsdC1" + "\n"
        self.resultado.setText(resultado_st)
        nomb2="C1"
        #
        def estadisticaPotencia(potenciaEnFreqCh1, potenciaEnFreqCh2, nomEP):
            # estadisticaPotencia(caracteristicasFreqBasalCh1, caracteristicasFreqBasalCh2)
            # mediaPotenciaCh1 = np.median(potenciaEnFreqCh1, axis=0)
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            #print(" SSSSSSSS len(potenciaEnFreqCh1) SSSSSSSSSSSSSSSS")
            #print(len(potenciaEnFreqCh1))
            # stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            # stErrCh1=stdPotenciaCh1/(np.sqrt(len(stdPotenciaCh1)))
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))


            #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            # plt.plot(range(1, mediaPotenciaCh1.shape[0] + 1), mediaPotenciaCh1, "b")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width,color="blue", label='y0:Estómago', yerr=stErrCh1)
            rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width,color="r", label='y1:Ciego', yerr=stErrCh2)
            ax.set_ylabel('PA [uV**2] y stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz')
            #ax.set_xticks(x)
            ax.set_xscale('linear')

            ax.legend()
            plt.show()

            return 0
        #
        estadisticaPotencia(self.C1Ch1, self.C1Ch2, nomb2)
    #C2
    def getCSV2(self):
        resultado_st = "getCSV2" + "\n"
        self.resultado.setText(resultado_st)

        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #
            def entropia(X):
                """Devuelve el valor de entropia de una muestra de datos"""
                probs = [np.mean(X == valor) for valor in set(X)]
                return round(np.sum(-p * np.log2(p) for p in probs), 3)
            #
            def curtoSis(y):
                curtosisY = kurtosis(y)
                return curtosisY
            #
            def estaBasica(y):
                estDesy = np.std(y, dtype=np.float64)
                meany = np.mean(y, dtype=np.float64)
                vary = np.var(y, dtype=np.float)
                medianY = np.median(y)
                Q1 = np.quantile(np.sort(y), 0.25)
                Q3 = np.quantile(np.sort(y), 0.75)
                return vary, meany, estDesy, medianY, Q1, Q3
            #
            def maximo(x):
                max = np.amax(x)
                inmax = np.argmax(x)
                return max, inmax
            #
            def pearsonCorr(x, y):
                corrpear = scipy.stats.pearsonr(x, y)
                """
                        The p-value roughly indicates the probability of an uncorrelated system
                        producing datasets that have a Pearson correlation at least as extreme
                        as the one computed from these datasets. The p-values are not entirely
                        reliable but are probably reasonable for datasets larger than 500 or so.
                        Parameters
                """

                return corrpear
            #
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values
            # def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
            #
            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label='PaBaja')
                # axs[0].set_ylim(yinf, ysup)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)

                axs[1].plot(x_valueDowSamy1, y1filPaBaDoSam1, linestyle='-', color='g', label='PaBaja')
                # axs[1].set_ylim(yinf, ysup)
                axs[1].set_xlim(x1, x2)
                axs[1].set_ylabel(y1nom)
                axs[1].grid(True)

                axs[2].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label='PaBaja')
                # axs[2].set_ylim(yinf, ysup)
                axs[2].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[2].set_ylabel(y2nom)
                axs[2].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):

                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                """
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")
                """
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 27-04-20
                """
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Est. y0 Hz", y1nom="Est. y0 Hz", y2nom="Ciego y1 Hz s/Ruido")
                """
                ######################################################
                # y0Guardar = y0filPaBaDoSam[0:longMenorHz]
                # y1Guardar = y1filPaBaDoSam[0:longMenorHz]
                # y0y1HzStack = np.stack((y0Guardar, y1Guardar), axis=-1)
                # guardarArchivo(y0y1HzStack, nombreExCa)



                # longMenor = np.min(np.array([longy0, longy1, longy2]))
                longMenor = np.min(np.array([longy0, longy1, longy2, longy0Hzsr, longy1Hzsr]))
                #print("longMenor: ", longMenor)
                # y0Guardar = y0filPaBaDoSam[0:longMenor]
                # y1Guardar = y1filPaBaDoSam[0:longMenor]
                # y2Guardar = y2filPaBaDoSam[0:longMenor]
                # y0y1y2Stack = np.stack((y0Guardar, y1Guardar, y2Guardar), axis=-1)
                # guardarArchivo(y0y1y2Stack, nombreExCa)

                ventanas0 = longMenor // 130
                #print("ventanas")
                #print(ventanas0)

                inicioDs = 0
                finDs = longMenor
                #print("len(y0filPaBaDoSam): ", len(y0filPaBaDoSam))
                #print("finDs", finDs)
                # Gráfica señales
                # plotDownSamp(y0filPaBaDoSam, y1filPaBaDoSam, y2filPaBaDoSam, yinf, ysup, inicioDs, finDs, nombreExCa,
                #            y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")


                #print("########### ExtractFeature ###############")
                # caracteristicasSenal = np.empty((0, 83))
                caracteristicasSenal = np.empty((0, 127))
                carFreSenalCh1 = np.empty((0, 129))
                carFreSenalCh2 = np.empty((0, 129))

                # for i in range(8):
                for i in range(ventanas0 - 1):
                    #print("i: ", i)
                    dt = 1
                    x_valueDowSam = np.arange(0, longMenor, dt)
                    inicioDs = 130 * i
                    finDs = inicioDs + 260

                    # Ventana de 2 minuto 10 segundos señal down sampled
                    # fs=2 hz         # 2 muestras por segundo
                    # para 60 segundos tengo 120 muestras

                    y0filPaBaDoSamTF = y0filPaBaDoSam[inicioDs:finDs]
                    y1filPaBaDoSamTF = y1filPaBaDoSam[inicioDs:finDs]
                    y2filPaBaDoSamTF = y2filPaBaDoSam[inicioDs:finDs]
                    y0HzfilPaBaDoSamTF = y0HzfilPaBaDoSam[inicioDs:finDs]
                    y1HzfilPaBaDoSamTF = y1HzfilPaBaDoSam[inicioDs:finDs]

                    # Se llama función plotDownSamp()
                    # plotDownSamp(y0filPaBaDoSam, y0filPaBaDoSam, y0filPaBaDoSamTF, yinf, ysup, inicioDs, finDs,
                    #            nombreExCa, y0nom="Estomago", y1nom="Estomago", y2nom="Estomago")

                    # plotDownSamp(y1filPaBaDoSam, y1filPaBaDoSam, y1filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #            y0nom="Ciego", y1nom="Ciego", y2nom="Ciego")

                    # plotDownSamp(y2filPaBaDoSam, y2filPaBaDoSam, y2filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #             y0nom="Ileum", y1nom="Ileum", y2nom="Ileum")

                    kurTiemy0 = curtoSis(y0filPaBaDoSamTF)
                    kurTiemy1 = curtoSis(y1filPaBaDoSamTF)
                    kurTiemy2 = curtoSis(y2filPaBaDoSamTF)

                    entTemy0 = entropia(y0filPaBaDoSamTF)
                    entTemy1 = entropia(y1filPaBaDoSamTF)
                    entTemy2 = entropia(y2filPaBaDoSamTF)

                    varY0Tf, meanY0Tf, stdY0Tf, medianY0Tf, Q1Y0Tf, Q3Y0Tf = estaBasica(y0filPaBaDoSamTF)
                    varY1Tf, meanY1Tf, stdY1Tf, medianY1Tf, Q1Y1Tf, Q3Y1Tf = estaBasica(y1filPaBaDoSamTF)
                    varY2Tf, meanY2Tf, stdY2Tf, medianY2Tf, Q1Y2Tf, Q3Y2Tf = estaBasica(y2filPaBaDoSamTF)

                    f_valuesch1, ch1Psd_values = get_psd_values(y0filPaBaDoSamTF, T, N, f_s)
                    f_valuesch2, ch2Psd_values = get_psd_values(y1filPaBaDoSamTF, T, N, f_s)
                    f_valuesch3, ch3Psd_values = get_psd_values(y2filPaBaDoSamTF, T, N, f_s)
                    # y1HzfilPaBaDoSamTF
                    f_valuesch1Hz, ch1HzPsd_values = get_psd_values(y0HzfilPaBaDoSamTF, T, N, f_s)
                    f_valuesch2Hz, ch2HzPsd_values = get_psd_values(y1HzfilPaBaDoSamTF, T, N, f_s)

                    #print("len(ch2HzPsd_values)")
                    #print(len(ch2HzPsd_values))

                    # plt.plot(f_valuesch1, ch1Psd_values, linestyle='-', color='black', label='Estomago')
                    # plt.plot(f_valuesch2, ch2Psd_values, linestyle='-', color='g', label='Ciego')
                    # plt.plot(f_valuesch3, ch3Psd_values, linestyle='-', color='r', label='Ileon')
                    # plt.plot(f_valuesch1Hz, ch1HzPsd_values, linestyle='-', color='blue', label='Ch1')
                    # plt.plot(f_valuesch2Hz, ch2HzPsd_values, linestyle='-', color='orange', label='Ch2')
                    # plt.xlabel('Frequencia [Hz]', fontsize=16)
                    # plt.ylabel('PSD [uV**2 / Hz]', fontsize=16)
                    # plt.title("Espectro Ciego, Estomago e Ileum", fontsize=16)
                    # plt.show()

                    r = pearsonCorr(ch1HzPsd_values, ch2HzPsd_values)

                    # Estomago
                    ch1PsdValVentana = ch1Psd_values[7:26]  # 12 datos
                    varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Ch1Tf, Q3Ch1Tf = estaBasica(ch1PsdValVentana)

                    # Ciego
                    # ch2PsdValVentana = ch2Psd_values[42:68] 26 datos
                    ch2PsdValVentana = ch2Psd_values[0:7]  # 8 datos
                    varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Ch2Tf, Q3Ch2Tf = estaBasica(ch2PsdValVentana)

                    # Ileum
                    ch3PsdValVentana = ch3Psd_values[40:84]
                    varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Ch3Tf, Q3Ch3Tf = estaBasica(ch3PsdValVentana)

                    kurFreCh1 = curtoSis(ch1PsdValVentana)
                    kurFreCh2 = curtoSis(ch2PsdValVentana)
                    kurFreCh3 = curtoSis(ch3PsdValVentana)
                    entFreCh1 = entropia(ch1PsdValVentana)
                    entFreCh2 = entropia(ch2PsdValVentana)
                    entFreCh3 = entropia(ch3PsdValVentana)

                    maxch2, inmaxch2 = maximo(ch2PsdValVentana)
                    #print("Ciego: máximo valor e indice", maxch2, inmaxch2)

                    maxch1, inmaxch1 = maximo(ch1PsdValVentana)
                    #print("Estomago: máximo valor e indice", maxch1, inmaxch1)

                    maxch3, inmaxch3 = maximo(ch3PsdValVentana)
                    #print("Ileum: máximo valor e indice", maxch3, inmaxch3)

                    caracteristicaVentana = np.hstack(
                        [ch1PsdValVentana, kurTiemy0, kurFreCh1, entTemy0, entFreCh1, maxch1, inmaxch1, varY0Tf,
                         meanY0Tf, stdY0Tf, medianY0Tf, varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Y0Tf, Q3Y0Tf,
                         Q1Ch1Tf, Q3Ch1Tf,
                         ch2PsdValVentana, kurTiemy1, kurFreCh2, entTemy1, entFreCh2, maxch2, inmaxch2, varY1Tf,
                         meanY1Tf, stdY1Tf, medianY1Tf, varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Y1Tf, Q3Y1Tf,
                         Q1Ch2Tf, Q3Ch2Tf,
                         ch3PsdValVentana, kurTiemy2, kurFreCh3, entTemy2, entFreCh3, maxch3, inmaxch3, varY2Tf,
                         meanY2Tf, stdY2Tf, medianY2Tf, varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Y2Tf, Q3Y2Tf,
                         Q1Ch3Tf, Q3Ch3Tf,
                         r, etiquetaSenal])
                    #print("caracteristicaVentana.shape")
                    #print(caracteristicaVentana.shape)
                    caracteristicasSenal = np.vstack((caracteristicasSenal, caracteristicaVentana))
                    carFreSenalCh1 = np.vstack((carFreSenalCh1, ch1HzPsd_values))
                    carFreSenalCh2 = np.vstack((carFreSenalCh2, ch2HzPsd_values))

                return caracteristicasSenal, carFreSenalCh1, carFreSenalCh2

            #

            df1 = self.df

            # Filtrado
            nomb = "C2"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            f, carFreSenalCh1, carFreSenalCh2 = extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
            # Acumula verticalmente las caracteristicas de todas las ratitas
            self.caracteristicasSenales = np.vstack([self.caracteristicasSenales, f])
            self.C2Ch1 = np.vstack([self.C2Ch1, carFreSenalCh1])
            self.C2Ch2 = np.vstack([self.C2Ch2, carFreSenalCh2])
            print("caracteristicasSenales")
            print(self.caracteristicasSenales.shape)
            print("C2Ch1")
            print(self.C2Ch1.shape)
            print("C2Ch2")
            print(self.C2Ch2.shape)
    #
    def plotCSV2time(self):
        resultado_st = "plotCSV2time" + "\n"
        self.resultado.setText(resultado_st)
        #C:\Users\Mozart\Mozart\QtDesigner
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/Users/Mozart/Mozart/QtDesigner')
        # Cambiar la ruta
        # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom, xnom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label=y0nom)
                # axs[0].set_ylim(yinf, ysup)
                axs[0].legend(frameon=False, fontsize=10)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)


                axs[1].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label=y2nom)
                # axs[2].set_ylim(yinf, ysup)
                axs[1].legend(frameon=False, fontsize=10)
                #axs[1].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[1].set_xlabel(xnom)
                axs[1].set_ylabel(y2nom)
                axs[1].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):
                # etapaFiltrado(df1, nomb)
                #print("etapaFiltrado")
                #print("Info df1")
                #print(df1.info())
                #print('\n' * 2)
                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()
                #print("Info df6")
                #print(df6.info())
                #print("\n")

                #print("Info df1=df3")
                #print(df3.info())
                #print('\n' * 2)

                #print("****Estadisticas  total****")
                # horizontal_stack = pd.concat([df3, df4], axis=1)
                #print(df3.describe(include=[np.number]))
                #print('\n' * 2)

                #print('Correlación Ch1, Ch2 y Ch3')
                #print(df3.corr())
                #print('\n' * 2)

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estómago", y1nom="Ciego", y2nom="Ciego",xnom="30720 muestras = 1 Min")
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)
                # Señales Estomago, Ciego, Ileum
                # plotDownSamp(y0filPaBaDoSam0, y1CifilPaBaDoSam1, y1filPaBaDoSam1, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="DS Estomago", y1nom="Ds Ciego", y2nom="Ds Ileum Filtro2")
                # Señales sensor Estomago y Ciego con frecuencia corte a 0.75 Hz
                # plotDownSamp(y0HzDoSamPaBaj, y0HzDoSamPaBaj, y1HzDoSamPaBaj, yinf, ysup, inicioDs, finDs, nombre,
                #            y0nom="DS Estomago 0Hz", y1nom="Ds Estomago 0Hz", y2nom="Ds Ileum 1Hz")


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 20-04-20
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Estómago: y0 ", y1nom="Est. y0 Hz", y2nom="Ciego: y1",xnom="120 muestras = 1 Min")

                return 1


            df1 = self.df

            # Filtrado
            nomb = "C2"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            nomb = "C2 Filtrada"
            a0=extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
    #
    def plotmeanPsdC2(self):
        #x=self.df['col1']
        resultado_st = "plotmeanPsdC2" + "\n"
        self.resultado.setText(resultado_st)
        nomb2="C2"
        #
        def estadisticaPotencia(potenciaEnFreqCh1, potenciaEnFreqCh2, nomEP):
            # estadisticaPotencia(caracteristicasFreqBasalCh1, caracteristicasFreqBasalCh2)
            # mediaPotenciaCh1 = np.median(potenciaEnFreqCh1, axis=0)
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            #print(" SSSSSSSS len(potenciaEnFreqCh1) SSSSSSSSSSSSSSSS")
            #print(len(potenciaEnFreqCh1))
            # stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            # stErrCh1=stdPotenciaCh1/(np.sqrt(len(stdPotenciaCh1)))
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))


            #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            # plt.plot(range(1, mediaPotenciaCh1.shape[0] + 1), mediaPotenciaCh1, "b")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width,color="blue", label='y0:Estómago', yerr=stErrCh1)
            rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width,color="r", label='y1:Ciego', yerr=stErrCh2)
            ax.set_ylabel('PA [uV**2] y stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz')
            #ax.set_xticks(x)
            ax.set_xscale('linear')

            ax.legend()
            plt.show()

            return 0
        #
        estadisticaPotencia(self.C2Ch1, self.C2Ch2, nomb2)
    #C3
    #
    def getCSV3(self):
        resultado_st = "getCSV3" + "\n"
        self.resultado.setText(resultado_st)

        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #
            def entropia(X):
                """Devuelve el valor de entropia de una muestra de datos"""
                probs = [np.mean(X == valor) for valor in set(X)]
                return round(np.sum(-p * np.log2(p) for p in probs), 3)
            #
            def curtoSis(y):
                curtosisY = kurtosis(y)
                return curtosisY
            #
            def estaBasica(y):
                estDesy = np.std(y, dtype=np.float64)
                meany = np.mean(y, dtype=np.float64)
                vary = np.var(y, dtype=np.float)
                medianY = np.median(y)
                Q1 = np.quantile(np.sort(y), 0.25)
                Q3 = np.quantile(np.sort(y), 0.75)
                return vary, meany, estDesy, medianY, Q1, Q3
            #
            def maximo(x):
                max = np.amax(x)
                inmax = np.argmax(x)
                return max, inmax
            #
            def pearsonCorr(x, y):
                corrpear = scipy.stats.pearsonr(x, y)
                """
                        The p-value roughly indicates the probability of an uncorrelated system
                        producing datasets that have a Pearson correlation at least as extreme
                        as the one computed from these datasets. The p-values are not entirely
                        reliable but are probably reasonable for datasets larger than 500 or so.
                        Parameters
                """

                return corrpear
            #
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values
            # def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
            #
            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label='PaBaja')
                # axs[0].set_ylim(yinf, ysup)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)

                axs[1].plot(x_valueDowSamy1, y1filPaBaDoSam1, linestyle='-', color='g', label='PaBaja')
                # axs[1].set_ylim(yinf, ysup)
                axs[1].set_xlim(x1, x2)
                axs[1].set_ylabel(y1nom)
                axs[1].grid(True)

                axs[2].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label='PaBaja')
                # axs[2].set_ylim(yinf, ysup)
                axs[2].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[2].set_ylabel(y2nom)
                axs[2].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):

                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                """
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")
                """
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 27-04-20
                """
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Est. y0 Hz", y1nom="Est. y0 Hz", y2nom="Ciego y1 Hz s/Ruido")
                """
                ######################################################
                # y0Guardar = y0filPaBaDoSam[0:longMenorHz]
                # y1Guardar = y1filPaBaDoSam[0:longMenorHz]
                # y0y1HzStack = np.stack((y0Guardar, y1Guardar), axis=-1)
                # guardarArchivo(y0y1HzStack, nombreExCa)



                # longMenor = np.min(np.array([longy0, longy1, longy2]))
                longMenor = np.min(np.array([longy0, longy1, longy2, longy0Hzsr, longy1Hzsr]))
                #print("longMenor: ", longMenor)
                # y0Guardar = y0filPaBaDoSam[0:longMenor]
                # y1Guardar = y1filPaBaDoSam[0:longMenor]
                # y2Guardar = y2filPaBaDoSam[0:longMenor]
                # y0y1y2Stack = np.stack((y0Guardar, y1Guardar, y2Guardar), axis=-1)
                # guardarArchivo(y0y1y2Stack, nombreExCa)

                ventanas0 = longMenor // 130
                #print("ventanas")
                #print(ventanas0)

                inicioDs = 0
                finDs = longMenor
                #print("len(y0filPaBaDoSam): ", len(y0filPaBaDoSam))
                #print("finDs", finDs)
                # Gráfica señales
                # plotDownSamp(y0filPaBaDoSam, y1filPaBaDoSam, y2filPaBaDoSam, yinf, ysup, inicioDs, finDs, nombreExCa,
                #            y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")


                #print("########### ExtractFeature ###############")
                # caracteristicasSenal = np.empty((0, 83))
                caracteristicasSenal = np.empty((0, 127))
                carFreSenalCh1 = np.empty((0, 129))
                carFreSenalCh2 = np.empty((0, 129))

                # for i in range(8):
                for i in range(ventanas0 - 1):
                    #print("i: ", i)
                    dt = 1
                    x_valueDowSam = np.arange(0, longMenor, dt)
                    inicioDs = 130 * i
                    finDs = inicioDs + 260

                    # Ventana de 2 minuto 10 segundos señal down sampled
                    # fs=2 hz         # 2 muestras por segundo
                    # para 60 segundos tengo 120 muestras

                    y0filPaBaDoSamTF = y0filPaBaDoSam[inicioDs:finDs]
                    y1filPaBaDoSamTF = y1filPaBaDoSam[inicioDs:finDs]
                    y2filPaBaDoSamTF = y2filPaBaDoSam[inicioDs:finDs]
                    y0HzfilPaBaDoSamTF = y0HzfilPaBaDoSam[inicioDs:finDs]
                    y1HzfilPaBaDoSamTF = y1HzfilPaBaDoSam[inicioDs:finDs]

                    # Se llama función plotDownSamp()
                    # plotDownSamp(y0filPaBaDoSam, y0filPaBaDoSam, y0filPaBaDoSamTF, yinf, ysup, inicioDs, finDs,
                    #            nombreExCa, y0nom="Estomago", y1nom="Estomago", y2nom="Estomago")

                    # plotDownSamp(y1filPaBaDoSam, y1filPaBaDoSam, y1filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #            y0nom="Ciego", y1nom="Ciego", y2nom="Ciego")

                    # plotDownSamp(y2filPaBaDoSam, y2filPaBaDoSam, y2filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #             y0nom="Ileum", y1nom="Ileum", y2nom="Ileum")

                    kurTiemy0 = curtoSis(y0filPaBaDoSamTF)
                    kurTiemy1 = curtoSis(y1filPaBaDoSamTF)
                    kurTiemy2 = curtoSis(y2filPaBaDoSamTF)

                    entTemy0 = entropia(y0filPaBaDoSamTF)
                    entTemy1 = entropia(y1filPaBaDoSamTF)
                    entTemy2 = entropia(y2filPaBaDoSamTF)

                    varY0Tf, meanY0Tf, stdY0Tf, medianY0Tf, Q1Y0Tf, Q3Y0Tf = estaBasica(y0filPaBaDoSamTF)
                    varY1Tf, meanY1Tf, stdY1Tf, medianY1Tf, Q1Y1Tf, Q3Y1Tf = estaBasica(y1filPaBaDoSamTF)
                    varY2Tf, meanY2Tf, stdY2Tf, medianY2Tf, Q1Y2Tf, Q3Y2Tf = estaBasica(y2filPaBaDoSamTF)

                    f_valuesch1, ch1Psd_values = get_psd_values(y0filPaBaDoSamTF, T, N, f_s)
                    f_valuesch2, ch2Psd_values = get_psd_values(y1filPaBaDoSamTF, T, N, f_s)
                    f_valuesch3, ch3Psd_values = get_psd_values(y2filPaBaDoSamTF, T, N, f_s)
                    # y1HzfilPaBaDoSamTF
                    f_valuesch1Hz, ch1HzPsd_values = get_psd_values(y0HzfilPaBaDoSamTF, T, N, f_s)
                    f_valuesch2Hz, ch2HzPsd_values = get_psd_values(y1HzfilPaBaDoSamTF, T, N, f_s)

                    #print("len(ch2HzPsd_values)")
                    #print(len(ch2HzPsd_values))

                    # plt.plot(f_valuesch1, ch1Psd_values, linestyle='-', color='black', label='Estomago')
                    # plt.plot(f_valuesch2, ch2Psd_values, linestyle='-', color='g', label='Ciego')
                    # plt.plot(f_valuesch3, ch3Psd_values, linestyle='-', color='r', label='Ileon')
                    # plt.plot(f_valuesch1Hz, ch1HzPsd_values, linestyle='-', color='blue', label='Ch1')
                    # plt.plot(f_valuesch2Hz, ch2HzPsd_values, linestyle='-', color='orange', label='Ch2')
                    # plt.xlabel('Frequencia [Hz]', fontsize=16)
                    # plt.ylabel('PSD [uV**2 / Hz]', fontsize=16)
                    # plt.title("Espectro Ciego, Estomago e Ileum", fontsize=16)
                    # plt.show()

                    r = pearsonCorr(ch1HzPsd_values, ch2HzPsd_values)

                    # Estomago
                    ch1PsdValVentana = ch1Psd_values[7:26]  # 12 datos
                    varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Ch1Tf, Q3Ch1Tf = estaBasica(ch1PsdValVentana)

                    # Ciego
                    # ch2PsdValVentana = ch2Psd_values[42:68] 26 datos
                    ch2PsdValVentana = ch2Psd_values[0:7]  # 8 datos
                    varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Ch2Tf, Q3Ch2Tf = estaBasica(ch2PsdValVentana)

                    # Ileum
                    ch3PsdValVentana = ch3Psd_values[40:84]
                    varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Ch3Tf, Q3Ch3Tf = estaBasica(ch3PsdValVentana)

                    kurFreCh1 = curtoSis(ch1PsdValVentana)
                    kurFreCh2 = curtoSis(ch2PsdValVentana)
                    kurFreCh3 = curtoSis(ch3PsdValVentana)
                    entFreCh1 = entropia(ch1PsdValVentana)
                    entFreCh2 = entropia(ch2PsdValVentana)
                    entFreCh3 = entropia(ch3PsdValVentana)

                    maxch2, inmaxch2 = maximo(ch2PsdValVentana)
                    #print("Ciego: máximo valor e indice", maxch2, inmaxch2)

                    maxch1, inmaxch1 = maximo(ch1PsdValVentana)
                    #print("Estomago: máximo valor e indice", maxch1, inmaxch1)

                    maxch3, inmaxch3 = maximo(ch3PsdValVentana)
                    #print("Ileum: máximo valor e indice", maxch3, inmaxch3)

                    caracteristicaVentana = np.hstack(
                        [ch1PsdValVentana, kurTiemy0, kurFreCh1, entTemy0, entFreCh1, maxch1, inmaxch1, varY0Tf,
                         meanY0Tf, stdY0Tf, medianY0Tf, varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Y0Tf, Q3Y0Tf,
                         Q1Ch1Tf, Q3Ch1Tf,
                         ch2PsdValVentana, kurTiemy1, kurFreCh2, entTemy1, entFreCh2, maxch2, inmaxch2, varY1Tf,
                         meanY1Tf, stdY1Tf, medianY1Tf, varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Y1Tf, Q3Y1Tf,
                         Q1Ch2Tf, Q3Ch2Tf,
                         ch3PsdValVentana, kurTiemy2, kurFreCh3, entTemy2, entFreCh3, maxch3, inmaxch3, varY2Tf,
                         meanY2Tf, stdY2Tf, medianY2Tf, varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Y2Tf, Q3Y2Tf,
                         Q1Ch3Tf, Q3Ch3Tf,
                         r, etiquetaSenal])
                    #print("caracteristicaVentana.shape")
                    #print(caracteristicaVentana.shape)
                    caracteristicasSenal = np.vstack((caracteristicasSenal, caracteristicaVentana))
                    carFreSenalCh1 = np.vstack((carFreSenalCh1, ch1HzPsd_values))
                    carFreSenalCh2 = np.vstack((carFreSenalCh2, ch2HzPsd_values))

                return caracteristicasSenal, carFreSenalCh1, carFreSenalCh2

            #

            df1 = self.df

            # Filtrado
            nomb = "C3"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            f, carFreSenalCh1, carFreSenalCh2 = extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
            # Acumula verticalmente las caracteristicas de todas las ratitas
            self.caracteristicasSenales = np.vstack([self.caracteristicasSenales, f])
            self.C3Ch1 = np.vstack([self.C3Ch1, carFreSenalCh1])
            self.C3Ch2 = np.vstack([self.C3Ch2, carFreSenalCh2])
            print("caracteristicasSenales")
            print(self.caracteristicasSenales.shape)
            print("C3Ch1")
            print(self.C3Ch1.shape)
            print("C3Ch2")
            print(self.C3Ch2.shape)
    #
    def plotCSV3time(self):
        resultado_st = "plotCSV3time" + "\n"
        self.resultado.setText(resultado_st)
        #C:\Users\Mozart\Mozart\QtDesigner
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/Users/Mozart/Mozart/QtDesigner')
        # Cambiar la ruta
        # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom, xnom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label=y0nom)
                # axs[0].set_ylim(yinf, ysup)
                axs[0].legend(frameon=False, fontsize=10)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)


                axs[1].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label=y2nom)
                # axs[2].set_ylim(yinf, ysup)
                axs[1].legend(frameon=False, fontsize=10)
                #axs[1].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[1].set_xlabel(xnom)
                axs[1].set_ylabel(y2nom)
                axs[1].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):
                # etapaFiltrado(df1, nomb)
                #print("etapaFiltrado")
                #print("Info df1")
                #print(df1.info())
                #print('\n' * 2)
                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()
                #print("Info df6")
                #print(df6.info())
                #print("\n")

                #print("Info df1=df3")
                #print(df3.info())
                #print('\n' * 2)

                #print("****Estadisticas  total****")
                # horizontal_stack = pd.concat([df3, df4], axis=1)
                #print(df3.describe(include=[np.number]))
                #print('\n' * 2)

                #print('Correlación Ch1, Ch2 y Ch3')
                #print(df3.corr())
                #print('\n' * 2)

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estómago", y1nom="Ciego", y2nom="Ciego",xnom="30720 muestras = 1 Min")
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)
                # Señales Estomago, Ciego, Ileum
                # plotDownSamp(y0filPaBaDoSam0, y1CifilPaBaDoSam1, y1filPaBaDoSam1, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="DS Estomago", y1nom="Ds Ciego", y2nom="Ds Ileum Filtro2")
                # Señales sensor Estomago y Ciego con frecuencia corte a 0.75 Hz
                # plotDownSamp(y0HzDoSamPaBaj, y0HzDoSamPaBaj, y1HzDoSamPaBaj, yinf, ysup, inicioDs, finDs, nombre,
                #            y0nom="DS Estomago 0Hz", y1nom="Ds Estomago 0Hz", y2nom="Ds Ileum 1Hz")


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 20-04-20
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Estómago: y0 ", y1nom="Est. y0 Hz", y2nom="Ciego: y1",xnom="120 muestras = 1 Min")

                return 1


            df1 = self.df

            # Filtrado
            nomb = "C3"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            nomb = "C3 Filtrada"
            a0=extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
    #
    def plotmeanPsdC3(self):
        #x=self.df['col1']
        resultado_st = "plotmeanPSDC3" + "\n"
        self.resultado.setText(resultado_st)
        nomb2="C3"
        #
        def estadisticaPotencia(potenciaEnFreqCh1, potenciaEnFreqCh2, nomEP):
            # estadisticaPotencia(caracteristicasFreqBasalCh1, caracteristicasFreqBasalCh2)
            # mediaPotenciaCh1 = np.median(potenciaEnFreqCh1, axis=0)
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            #print(" SSSSSSSS len(potenciaEnFreqCh1) SSSSSSSSSSSSSSSS")
            #print(len(potenciaEnFreqCh1))
            # stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            # stErrCh1=stdPotenciaCh1/(np.sqrt(len(stdPotenciaCh1)))
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))


            #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            # plt.plot(range(1, mediaPotenciaCh1.shape[0] + 1), mediaPotenciaCh1, "b")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width,color="blue", label='y0:Estómago', yerr=stErrCh1)
            rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width,color="r", label='y1:Ciego', yerr=stErrCh2)
            ax.set_ylabel('PA [uV**2] y stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz')
            #ax.set_xticks(x)
            ax.set_xscale('linear')

            ax.legend()
            plt.show()

            return 0
        #
        estadisticaPotencia(self.C3Ch1, self.C3Ch2, nomb2)
    #C4
    def getCSV4(self):
        resultado_st = "getCSV4" + "\n"
        self.resultado.setText(resultado_st)

        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #
            def entropia(X):
                """Devuelve el valor de entropia de una muestra de datos"""
                probs = [np.mean(X == valor) for valor in set(X)]
                return round(np.sum(-p * np.log2(p) for p in probs), 3)
            #
            def curtoSis(y):
                curtosisY = kurtosis(y)
                return curtosisY
            #
            def estaBasica(y):
                estDesy = np.std(y, dtype=np.float64)
                meany = np.mean(y, dtype=np.float64)
                vary = np.var(y, dtype=np.float)
                medianY = np.median(y)
                Q1 = np.quantile(np.sort(y), 0.25)
                Q3 = np.quantile(np.sort(y), 0.75)
                return vary, meany, estDesy, medianY, Q1, Q3
            #
            def maximo(x):
                max = np.amax(x)
                inmax = np.argmax(x)
                return max, inmax
            #
            def pearsonCorr(x, y):
                corrpear = scipy.stats.pearsonr(x, y)
                """
                        The p-value roughly indicates the probability of an uncorrelated system
                        producing datasets that have a Pearson correlation at least as extreme
                        as the one computed from these datasets. The p-values are not entirely
                        reliable but are probably reasonable for datasets larger than 500 or so.
                        Parameters
                """

                return corrpear
            #
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values
            # def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
            #
            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label='PaBaja')
                # axs[0].set_ylim(yinf, ysup)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)

                axs[1].plot(x_valueDowSamy1, y1filPaBaDoSam1, linestyle='-', color='g', label='PaBaja')
                # axs[1].set_ylim(yinf, ysup)
                axs[1].set_xlim(x1, x2)
                axs[1].set_ylabel(y1nom)
                axs[1].grid(True)

                axs[2].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label='PaBaja')
                # axs[2].set_ylim(yinf, ysup)
                axs[2].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[2].set_ylabel(y2nom)
                axs[2].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):

                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                """
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")
                """
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 27-04-20
                """
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Est. y0 Hz", y1nom="Est. y0 Hz", y2nom="Ciego y1 Hz s/Ruido")
                """
                ######################################################
                # y0Guardar = y0filPaBaDoSam[0:longMenorHz]
                # y1Guardar = y1filPaBaDoSam[0:longMenorHz]
                # y0y1HzStack = np.stack((y0Guardar, y1Guardar), axis=-1)
                # guardarArchivo(y0y1HzStack, nombreExCa)



                # longMenor = np.min(np.array([longy0, longy1, longy2]))
                longMenor = np.min(np.array([longy0, longy1, longy2, longy0Hzsr, longy1Hzsr]))
                #print("longMenor: ", longMenor)
                # y0Guardar = y0filPaBaDoSam[0:longMenor]
                # y1Guardar = y1filPaBaDoSam[0:longMenor]
                # y2Guardar = y2filPaBaDoSam[0:longMenor]
                # y0y1y2Stack = np.stack((y0Guardar, y1Guardar, y2Guardar), axis=-1)
                # guardarArchivo(y0y1y2Stack, nombreExCa)

                ventanas0 = longMenor // 130
                #print("ventanas")
                #print(ventanas0)

                inicioDs = 0
                finDs = longMenor
                #print("len(y0filPaBaDoSam): ", len(y0filPaBaDoSam))
                #print("finDs", finDs)
                # Gráfica señales
                # plotDownSamp(y0filPaBaDoSam, y1filPaBaDoSam, y2filPaBaDoSam, yinf, ysup, inicioDs, finDs, nombreExCa,
                #            y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")


                #print("########### ExtractFeature ###############")
                # caracteristicasSenal = np.empty((0, 83))
                caracteristicasSenal = np.empty((0, 127))
                carFreSenalCh1 = np.empty((0, 129))
                carFreSenalCh2 = np.empty((0, 129))

                # for i in range(8):
                for i in range(ventanas0 - 1):
                    #print("i: ", i)
                    dt = 1
                    x_valueDowSam = np.arange(0, longMenor, dt)
                    inicioDs = 130 * i
                    finDs = inicioDs + 260

                    # Ventana de 2 minuto 10 segundos señal down sampled
                    # fs=2 hz         # 2 muestras por segundo
                    # para 60 segundos tengo 120 muestras

                    y0filPaBaDoSamTF = y0filPaBaDoSam[inicioDs:finDs]
                    y1filPaBaDoSamTF = y1filPaBaDoSam[inicioDs:finDs]
                    y2filPaBaDoSamTF = y2filPaBaDoSam[inicioDs:finDs]
                    y0HzfilPaBaDoSamTF = y0HzfilPaBaDoSam[inicioDs:finDs]
                    y1HzfilPaBaDoSamTF = y1HzfilPaBaDoSam[inicioDs:finDs]

                    # Se llama función plotDownSamp()
                    # plotDownSamp(y0filPaBaDoSam, y0filPaBaDoSam, y0filPaBaDoSamTF, yinf, ysup, inicioDs, finDs,
                    #            nombreExCa, y0nom="Estomago", y1nom="Estomago", y2nom="Estomago")

                    # plotDownSamp(y1filPaBaDoSam, y1filPaBaDoSam, y1filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #            y0nom="Ciego", y1nom="Ciego", y2nom="Ciego")

                    # plotDownSamp(y2filPaBaDoSam, y2filPaBaDoSam, y2filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #             y0nom="Ileum", y1nom="Ileum", y2nom="Ileum")

                    kurTiemy0 = curtoSis(y0filPaBaDoSamTF)
                    kurTiemy1 = curtoSis(y1filPaBaDoSamTF)
                    kurTiemy2 = curtoSis(y2filPaBaDoSamTF)

                    entTemy0 = entropia(y0filPaBaDoSamTF)
                    entTemy1 = entropia(y1filPaBaDoSamTF)
                    entTemy2 = entropia(y2filPaBaDoSamTF)

                    varY0Tf, meanY0Tf, stdY0Tf, medianY0Tf, Q1Y0Tf, Q3Y0Tf = estaBasica(y0filPaBaDoSamTF)
                    varY1Tf, meanY1Tf, stdY1Tf, medianY1Tf, Q1Y1Tf, Q3Y1Tf = estaBasica(y1filPaBaDoSamTF)
                    varY2Tf, meanY2Tf, stdY2Tf, medianY2Tf, Q1Y2Tf, Q3Y2Tf = estaBasica(y2filPaBaDoSamTF)

                    f_valuesch1, ch1Psd_values = get_psd_values(y0filPaBaDoSamTF, T, N, f_s)
                    f_valuesch2, ch2Psd_values = get_psd_values(y1filPaBaDoSamTF, T, N, f_s)
                    f_valuesch3, ch3Psd_values = get_psd_values(y2filPaBaDoSamTF, T, N, f_s)
                    # y1HzfilPaBaDoSamTF
                    f_valuesch1Hz, ch1HzPsd_values = get_psd_values(y0HzfilPaBaDoSamTF, T, N, f_s)
                    f_valuesch2Hz, ch2HzPsd_values = get_psd_values(y1HzfilPaBaDoSamTF, T, N, f_s)

                    #print("len(ch2HzPsd_values)")
                    #print(len(ch2HzPsd_values))

                    # plt.plot(f_valuesch1, ch1Psd_values, linestyle='-', color='black', label='Estomago')
                    # plt.plot(f_valuesch2, ch2Psd_values, linestyle='-', color='g', label='Ciego')
                    # plt.plot(f_valuesch3, ch3Psd_values, linestyle='-', color='r', label='Ileon')
                    # plt.plot(f_valuesch1Hz, ch1HzPsd_values, linestyle='-', color='blue', label='Ch1')
                    # plt.plot(f_valuesch2Hz, ch2HzPsd_values, linestyle='-', color='orange', label='Ch2')
                    # plt.xlabel('Frequencia [Hz]', fontsize=16)
                    # plt.ylabel('PSD [uV**2 / Hz]', fontsize=16)
                    # plt.title("Espectro Ciego, Estomago e Ileum", fontsize=16)
                    # plt.show()

                    r = pearsonCorr(ch1HzPsd_values, ch2HzPsd_values)

                    # Estomago
                    ch1PsdValVentana = ch1Psd_values[7:26]  # 12 datos
                    varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Ch1Tf, Q3Ch1Tf = estaBasica(ch1PsdValVentana)

                    # Ciego
                    # ch2PsdValVentana = ch2Psd_values[42:68] 26 datos
                    ch2PsdValVentana = ch2Psd_values[0:7]  # 8 datos
                    varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Ch2Tf, Q3Ch2Tf = estaBasica(ch2PsdValVentana)

                    # Ileum
                    ch3PsdValVentana = ch3Psd_values[40:84]
                    varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Ch3Tf, Q3Ch3Tf = estaBasica(ch3PsdValVentana)

                    kurFreCh1 = curtoSis(ch1PsdValVentana)
                    kurFreCh2 = curtoSis(ch2PsdValVentana)
                    kurFreCh3 = curtoSis(ch3PsdValVentana)
                    entFreCh1 = entropia(ch1PsdValVentana)
                    entFreCh2 = entropia(ch2PsdValVentana)
                    entFreCh3 = entropia(ch3PsdValVentana)

                    maxch2, inmaxch2 = maximo(ch2PsdValVentana)
                    #print("Ciego: máximo valor e indice", maxch2, inmaxch2)

                    maxch1, inmaxch1 = maximo(ch1PsdValVentana)
                    #print("Estomago: máximo valor e indice", maxch1, inmaxch1)

                    maxch3, inmaxch3 = maximo(ch3PsdValVentana)
                    #print("Ileum: máximo valor e indice", maxch3, inmaxch3)

                    caracteristicaVentana = np.hstack(
                        [ch1PsdValVentana, kurTiemy0, kurFreCh1, entTemy0, entFreCh1, maxch1, inmaxch1, varY0Tf,
                         meanY0Tf, stdY0Tf, medianY0Tf, varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Y0Tf, Q3Y0Tf,
                         Q1Ch1Tf, Q3Ch1Tf,
                         ch2PsdValVentana, kurTiemy1, kurFreCh2, entTemy1, entFreCh2, maxch2, inmaxch2, varY1Tf,
                         meanY1Tf, stdY1Tf, medianY1Tf, varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Y1Tf, Q3Y1Tf,
                         Q1Ch2Tf, Q3Ch2Tf,
                         ch3PsdValVentana, kurTiemy2, kurFreCh3, entTemy2, entFreCh3, maxch3, inmaxch3, varY2Tf,
                         meanY2Tf, stdY2Tf, medianY2Tf, varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Y2Tf, Q3Y2Tf,
                         Q1Ch3Tf, Q3Ch3Tf,
                         r, etiquetaSenal])
                    #print("caracteristicaVentana.shape")
                    #print(caracteristicaVentana.shape)
                    caracteristicasSenal = np.vstack((caracteristicasSenal, caracteristicaVentana))
                    carFreSenalCh1 = np.vstack((carFreSenalCh1, ch1HzPsd_values))
                    carFreSenalCh2 = np.vstack((carFreSenalCh2, ch2HzPsd_values))

                return caracteristicasSenal, carFreSenalCh1, carFreSenalCh2

            #

            df1 = self.df

            # Filtrado
            nomb = "C4"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            f, carFreSenalCh1, carFreSenalCh2 = extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
            # Acumula verticalmente las caracteristicas de todas las ratitas
            self.caracteristicasSenales = np.vstack([self.caracteristicasSenales, f])
            self.C4Ch1 = np.vstack([self.C4Ch1, carFreSenalCh1])
            self.C4Ch2 = np.vstack([self.C4Ch2, carFreSenalCh2])
            print("caracteristicasSenales")
            print(self.caracteristicasSenales.shape)
            print("C4Ch1")
            print(self.C4Ch1.shape)
            print("C4Ch2")
            print(self.C4Ch2.shape)
    #
    def plotCSV4time(self):
        resultado_st = "plotCSV4time" + "\n"
        self.resultado.setText(resultado_st)
        #C:\Users\Mozart\Mozart\QtDesigner
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/Users/Mozart/Mozart/QtDesigner')
        # Cambiar la ruta
        # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom, xnom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label=y0nom)
                # axs[0].set_ylim(yinf, ysup)
                axs[0].legend(frameon=False, fontsize=10)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)


                axs[1].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label=y2nom)
                # axs[2].set_ylim(yinf, ysup)
                axs[1].legend(frameon=False, fontsize=10)
                #axs[1].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[1].set_xlabel(xnom)
                axs[1].set_ylabel(y2nom)
                axs[1].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):
                # etapaFiltrado(df1, nomb)
                #print("etapaFiltrado")
                #print("Info df1")
                #print(df1.info())
                #print('\n' * 2)
                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()
                #print("Info df6")
                #print(df6.info())
                #print("\n")

                #print("Info df1=df3")
                #print(df3.info())
                #print('\n' * 2)

                #print("****Estadisticas  total****")
                # horizontal_stack = pd.concat([df3, df4], axis=1)
                #print(df3.describe(include=[np.number]))
                #print('\n' * 2)

                #print('Correlación Ch1, Ch2 y Ch3')
                #print(df3.corr())
                #print('\n' * 2)

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estómago", y1nom="Ciego", y2nom="Ciego",xnom="30720 muestras = 1 Min")
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)
                # Señales Estomago, Ciego, Ileum
                # plotDownSamp(y0filPaBaDoSam0, y1CifilPaBaDoSam1, y1filPaBaDoSam1, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="DS Estomago", y1nom="Ds Ciego", y2nom="Ds Ileum Filtro2")
                # Señales sensor Estomago y Ciego con frecuencia corte a 0.75 Hz
                # plotDownSamp(y0HzDoSamPaBaj, y0HzDoSamPaBaj, y1HzDoSamPaBaj, yinf, ysup, inicioDs, finDs, nombre,
                #            y0nom="DS Estomago 0Hz", y1nom="Ds Estomago 0Hz", y2nom="Ds Ileum 1Hz")


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 20-04-20
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Estómago: y0 ", y1nom="Est. y0 Hz", y2nom="Ciego: y1",xnom="120 muestras = 1 Min")

                return 1


            df1 = self.df

            # Filtrado
            nomb = "C4"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            nomb = "C4 Filtrada"
            a0=extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
    #
    def plotmeanPsdC4(self):
        #x=self.df['col1']
        resultado_st = "plotmeanPsdC4" + "\n"
        self.resultado.setText(resultado_st)
        nomb2="C4"
        #
        def estadisticaPotencia(potenciaEnFreqCh1, potenciaEnFreqCh2, nomEP):
            # estadisticaPotencia(caracteristicasFreqBasalCh1, caracteristicasFreqBasalCh2)
            # mediaPotenciaCh1 = np.median(potenciaEnFreqCh1, axis=0)
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            #print(" SSSSSSSS len(potenciaEnFreqCh1) SSSSSSSSSSSSSSSS")
            #print(len(potenciaEnFreqCh1))
            # stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            # stErrCh1=stdPotenciaCh1/(np.sqrt(len(stdPotenciaCh1)))
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))


            #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            # plt.plot(range(1, mediaPotenciaCh1.shape[0] + 1), mediaPotenciaCh1, "b")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width,color="blue", label='y0:Estómago', yerr=stErrCh1)
            rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width,color="r", label='y1:Ciego', yerr=stErrCh2)
            ax.set_ylabel('PA [uV**2] y stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz')
            #ax.set_xticks(x)
            ax.set_xscale('linear')

            ax.legend()
            plt.show()

            return 0
        #
        estadisticaPotencia(self.C4Ch1, self.C4Ch2, nomb2)
    #C5
    def getCSV5(self):
        resultado_st = "getCSV5" + "\n"
        self.resultado.setText(resultado_st)

        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            #
            def entropia(X):
                """Devuelve el valor de entropia de una muestra de datos"""
                probs = [np.mean(X == valor) for valor in set(X)]
                return round(np.sum(-p * np.log2(p) for p in probs), 3)
            #
            def curtoSis(y):
                curtosisY = kurtosis(y)
                return curtosisY
            #
            def estaBasica(y):
                estDesy = np.std(y, dtype=np.float64)
                meany = np.mean(y, dtype=np.float64)
                vary = np.var(y, dtype=np.float)
                medianY = np.median(y)
                Q1 = np.quantile(np.sort(y), 0.25)
                Q3 = np.quantile(np.sort(y), 0.75)
                return vary, meany, estDesy, medianY, Q1, Q3
            #
            def maximo(x):
                max = np.amax(x)
                inmax = np.argmax(x)
                return max, inmax
            #
            def pearsonCorr(x, y):
                corrpear = scipy.stats.pearsonr(x, y)
                """
                        The p-value roughly indicates the probability of an uncorrelated system
                        producing datasets that have a Pearson correlation at least as extreme
                        as the one computed from these datasets. The p-values are not entirely
                        reliable but are probably reasonable for datasets larger than 500 or so.
                        Parameters
                """

                return corrpear
            #
            def get_psd_values(y_values, T, N, f_s):
                f_values, psd_values = welch(y_values, fs=f_s)
                return f_values, psd_values
            # def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
            #
            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label='PaBaja')
                # axs[0].set_ylim(yinf, ysup)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)

                axs[1].plot(x_valueDowSamy1, y1filPaBaDoSam1, linestyle='-', color='g', label='PaBaja')
                # axs[1].set_ylim(yinf, ysup)
                axs[1].set_xlim(x1, x2)
                axs[1].set_ylabel(y1nom)
                axs[1].grid(True)

                axs[2].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label='PaBaja')
                # axs[2].set_ylim(yinf, ysup)
                axs[2].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[2].set_ylabel(y2nom)
                axs[2].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):

                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                """
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")
                """
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 27-04-20
                """
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Est. y0 Hz", y1nom="Est. y0 Hz", y2nom="Ciego y1 Hz s/Ruido")
                """
                ######################################################
                # y0Guardar = y0filPaBaDoSam[0:longMenorHz]
                # y1Guardar = y1filPaBaDoSam[0:longMenorHz]
                # y0y1HzStack = np.stack((y0Guardar, y1Guardar), axis=-1)
                # guardarArchivo(y0y1HzStack, nombreExCa)



                # longMenor = np.min(np.array([longy0, longy1, longy2]))
                longMenor = np.min(np.array([longy0, longy1, longy2, longy0Hzsr, longy1Hzsr]))
                #print("longMenor: ", longMenor)
                # y0Guardar = y0filPaBaDoSam[0:longMenor]
                # y1Guardar = y1filPaBaDoSam[0:longMenor]
                # y2Guardar = y2filPaBaDoSam[0:longMenor]
                # y0y1y2Stack = np.stack((y0Guardar, y1Guardar, y2Guardar), axis=-1)
                # guardarArchivo(y0y1y2Stack, nombreExCa)

                ventanas0 = longMenor // 130
                #print("ventanas")
                #print(ventanas0)

                inicioDs = 0
                finDs = longMenor
                #print("len(y0filPaBaDoSam): ", len(y0filPaBaDoSam))
                #print("finDs", finDs)
                # Gráfica señales
                # plotDownSamp(y0filPaBaDoSam, y1filPaBaDoSam, y2filPaBaDoSam, yinf, ysup, inicioDs, finDs, nombreExCa,
                #            y0nom="Estomago", y1nom="Ciego", y2nom="Ileum")


                #print("########### ExtractFeature ###############")
                # caracteristicasSenal = np.empty((0, 83))
                caracteristicasSenal = np.empty((0, 127))
                carFreSenalCh1 = np.empty((0, 129))
                carFreSenalCh2 = np.empty((0, 129))

                # for i in range(8):
                for i in range(ventanas0 - 1):
                    #print("i: ", i)
                    dt = 1
                    x_valueDowSam = np.arange(0, longMenor, dt)
                    inicioDs = 130 * i
                    finDs = inicioDs + 260

                    # Ventana de 2 minuto 10 segundos señal down sampled
                    # fs=2 hz         # 2 muestras por segundo
                    # para 60 segundos tengo 120 muestras

                    y0filPaBaDoSamTF = y0filPaBaDoSam[inicioDs:finDs]
                    y1filPaBaDoSamTF = y1filPaBaDoSam[inicioDs:finDs]
                    y2filPaBaDoSamTF = y2filPaBaDoSam[inicioDs:finDs]
                    y0HzfilPaBaDoSamTF = y0HzfilPaBaDoSam[inicioDs:finDs]
                    y1HzfilPaBaDoSamTF = y1HzfilPaBaDoSam[inicioDs:finDs]

                    # Se llama función plotDownSamp()
                    # plotDownSamp(y0filPaBaDoSam, y0filPaBaDoSam, y0filPaBaDoSamTF, yinf, ysup, inicioDs, finDs,
                    #            nombreExCa, y0nom="Estomago", y1nom="Estomago", y2nom="Estomago")

                    # plotDownSamp(y1filPaBaDoSam, y1filPaBaDoSam, y1filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #            y0nom="Ciego", y1nom="Ciego", y2nom="Ciego")

                    # plotDownSamp(y2filPaBaDoSam, y2filPaBaDoSam, y2filPaBaDoSamTF, yinf, ysup, inicioDs, finDs, nombreExCa,
                    #             y0nom="Ileum", y1nom="Ileum", y2nom="Ileum")

                    kurTiemy0 = curtoSis(y0filPaBaDoSamTF)
                    kurTiemy1 = curtoSis(y1filPaBaDoSamTF)
                    kurTiemy2 = curtoSis(y2filPaBaDoSamTF)

                    entTemy0 = entropia(y0filPaBaDoSamTF)
                    entTemy1 = entropia(y1filPaBaDoSamTF)
                    entTemy2 = entropia(y2filPaBaDoSamTF)

                    varY0Tf, meanY0Tf, stdY0Tf, medianY0Tf, Q1Y0Tf, Q3Y0Tf = estaBasica(y0filPaBaDoSamTF)
                    varY1Tf, meanY1Tf, stdY1Tf, medianY1Tf, Q1Y1Tf, Q3Y1Tf = estaBasica(y1filPaBaDoSamTF)
                    varY2Tf, meanY2Tf, stdY2Tf, medianY2Tf, Q1Y2Tf, Q3Y2Tf = estaBasica(y2filPaBaDoSamTF)

                    f_valuesch1, ch1Psd_values = get_psd_values(y0filPaBaDoSamTF, T, N, f_s)
                    f_valuesch2, ch2Psd_values = get_psd_values(y1filPaBaDoSamTF, T, N, f_s)
                    f_valuesch3, ch3Psd_values = get_psd_values(y2filPaBaDoSamTF, T, N, f_s)
                    # y1HzfilPaBaDoSamTF
                    f_valuesch1Hz, ch1HzPsd_values = get_psd_values(y0HzfilPaBaDoSamTF, T, N, f_s)
                    f_valuesch2Hz, ch2HzPsd_values = get_psd_values(y1HzfilPaBaDoSamTF, T, N, f_s)

                    #print("len(ch2HzPsd_values)")
                    #print(len(ch2HzPsd_values))

                    # plt.plot(f_valuesch1, ch1Psd_values, linestyle='-', color='black', label='Estomago')
                    # plt.plot(f_valuesch2, ch2Psd_values, linestyle='-', color='g', label='Ciego')
                    # plt.plot(f_valuesch3, ch3Psd_values, linestyle='-', color='r', label='Ileon')
                    # plt.plot(f_valuesch1Hz, ch1HzPsd_values, linestyle='-', color='blue', label='Ch1')
                    # plt.plot(f_valuesch2Hz, ch2HzPsd_values, linestyle='-', color='orange', label='Ch2')
                    # plt.xlabel('Frequencia [Hz]', fontsize=16)
                    # plt.ylabel('PSD [uV**2 / Hz]', fontsize=16)
                    # plt.title("Espectro Ciego, Estomago e Ileum", fontsize=16)
                    # plt.show()

                    r = pearsonCorr(ch1HzPsd_values, ch2HzPsd_values)

                    # Estomago
                    ch1PsdValVentana = ch1Psd_values[7:26]  # 12 datos
                    varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Ch1Tf, Q3Ch1Tf = estaBasica(ch1PsdValVentana)

                    # Ciego
                    # ch2PsdValVentana = ch2Psd_values[42:68] 26 datos
                    ch2PsdValVentana = ch2Psd_values[0:7]  # 8 datos
                    varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Ch2Tf, Q3Ch2Tf = estaBasica(ch2PsdValVentana)

                    # Ileum
                    ch3PsdValVentana = ch3Psd_values[40:84]
                    varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Ch3Tf, Q3Ch3Tf = estaBasica(ch3PsdValVentana)

                    kurFreCh1 = curtoSis(ch1PsdValVentana)
                    kurFreCh2 = curtoSis(ch2PsdValVentana)
                    kurFreCh3 = curtoSis(ch3PsdValVentana)
                    entFreCh1 = entropia(ch1PsdValVentana)
                    entFreCh2 = entropia(ch2PsdValVentana)
                    entFreCh3 = entropia(ch3PsdValVentana)

                    maxch2, inmaxch2 = maximo(ch2PsdValVentana)
                    #print("Ciego: máximo valor e indice", maxch2, inmaxch2)

                    maxch1, inmaxch1 = maximo(ch1PsdValVentana)
                    #print("Estomago: máximo valor e indice", maxch1, inmaxch1)

                    maxch3, inmaxch3 = maximo(ch3PsdValVentana)
                    #print("Ileum: máximo valor e indice", maxch3, inmaxch3)

                    caracteristicaVentana = np.hstack(
                        [ch1PsdValVentana, kurTiemy0, kurFreCh1, entTemy0, entFreCh1, maxch1, inmaxch1, varY0Tf,
                         meanY0Tf, stdY0Tf, medianY0Tf, varCh1Tf, meanCh1Tf, stCh1Tf, medianCh1Tf, Q1Y0Tf, Q3Y0Tf,
                         Q1Ch1Tf, Q3Ch1Tf,
                         ch2PsdValVentana, kurTiemy1, kurFreCh2, entTemy1, entFreCh2, maxch2, inmaxch2, varY1Tf,
                         meanY1Tf, stdY1Tf, medianY1Tf, varCh2Tf, meanCh2Tf, stCh2Tf, medianCh2Tf, Q1Y1Tf, Q3Y1Tf,
                         Q1Ch2Tf, Q3Ch2Tf,
                         ch3PsdValVentana, kurTiemy2, kurFreCh3, entTemy2, entFreCh3, maxch3, inmaxch3, varY2Tf,
                         meanY2Tf, stdY2Tf, medianY2Tf, varCh3Tf, meanCh3Tf, stCh3Tf, medianCh3Tf, Q1Y2Tf, Q3Y2Tf,
                         Q1Ch3Tf, Q3Ch3Tf,
                         r, etiquetaSenal])
                    #print("caracteristicaVentana.shape")
                    #print(caracteristicaVentana.shape)
                    caracteristicasSenal = np.vstack((caracteristicasSenal, caracteristicaVentana))
                    carFreSenalCh1 = np.vstack((carFreSenalCh1, ch1HzPsd_values))
                    carFreSenalCh2 = np.vstack((carFreSenalCh2, ch2HzPsd_values))

                return caracteristicasSenal, carFreSenalCh1, carFreSenalCh2

            #

            df1 = self.df

            # Filtrado
            nomb = "C5"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            f, carFreSenalCh1, carFreSenalCh2 = extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
            # Acumula verticalmente las caracteristicas de todas las ratitas
            self.caracteristicasSenales = np.vstack([self.caracteristicasSenales, f])
            self.C5Ch1 = np.vstack([self.C5Ch1, carFreSenalCh1])
            self.C5Ch2 = np.vstack([self.C5Ch2, carFreSenalCh2])
            print("caracteristicasSenales")
            print(self.caracteristicasSenales.shape)
            print("C5Ch1")
            print(self.C5Ch1.shape)
            print("C5Ch2")
            print(self.C5Ch2.shape)
    #
    def plotCSV5time(self):
        resultado_st = "plotCSV5time" + "\n"
        self.resultado.setText(resultado_st)
        #C:\Users\Mozart\Mozart\QtDesigner
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        #filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/Users/Mozart/Mozart/QtDesigner')
        # Cambiar la ruta
        # filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
        filePath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if filePath != "":
            print("Dirección", filePath)  # Opcional imprimir la dirección del archivo
            self.df = pd.read_csv(str(filePath))

            def plotDownSamp(y0, y1, y2, yinf, ysup, x1, x2, nombre, y0nom, y1nom, y2nom, xnom):
                y0filPaBaDoSam0 = y0
                y1filPaBaDoSam1 = y1
                y2filPaBaDoSam2 = y2
                longDownSamy0 = len(y0filPaBaDoSam0)
                longDownSamy1 = len(y1filPaBaDoSam1)
                longDownSamy2 = len(y2filPaBaDoSam2)

                dt = 1
                x_valueDowSamy0 = np.arange(0, longDownSamy0, dt)
                x_valueDowSamy1 = np.arange(0, longDownSamy1, dt)
                x_valueDowSamy2 = np.arange(0, longDownSamy2, dt)

                fig, axs = plt.subplots(2, 1)
                axs[0].plot(x_valueDowSamy0, y0filPaBaDoSam0, linestyle='-', color='black', label=y0nom)
                # axs[0].set_ylim(yinf, ysup)
                axs[0].legend(frameon=False, fontsize=10)
                axs[0].set_title(nombre)
                axs[0].set_ylabel(y0nom)
                axs[0].grid(True)


                axs[1].plot(x_valueDowSamy2, y2filPaBaDoSam2, linestyle='-', color='red', label=y2nom)
                # axs[2].set_ylim(yinf, ysup)
                axs[1].legend(frameon=False, fontsize=10)
                #axs[1].set_xlabel('Muestras (120 muestras = 60 Seg)')
                axs[1].set_xlabel(xnom)
                axs[1].set_ylabel(y2nom)
                axs[1].grid(True)
                plt.show()

                return 1

            def eliminaRuido(y0, y0Hz, y1Ci, y1Hz, y1, y2):
                # Se revisa una ventana de 30 seg= 60 muestras
                # fs=2 hz
                # 2 muestras por segundo

                longitud = len(y0)
                #print("longitud", longitud)
                ventanas = longitud // 30

                accStaDes0 = np.array([])
                accStaDes0Hz = np.array([])
                accStaDes1Ci = np.array([])
                accStaDes1Hz = np.array([])
                accStaDes1 = np.array([])
                accStaDes2 = np.array([])
                # Se divide la señal en segmentos
                for i in range(ventanas - 1):
                    a = 0
                    inicio = 30 * i
                    fin = inicio + 60

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]
                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]
                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    # Se acumula la desviación estandar de cada segmento
                    accStaDes0 = np.append(accStaDes0, estDes0)
                    accStaDes0Hz = np.append(accStaDes0Hz, estDes0Hz)
                    accStaDes1Ci = np.append(accStaDes1Ci, estDes1Ci)
                    accStaDes1Hz = np.append(accStaDes1Hz, estDes1Hz)
                    accStaDes1 = np.append(accStaDes1, estDes1)
                    accStaDes2 = np.append(accStaDes2, estDes2)
                # Se calcula el promedio de las desviaciones estandar de todos los segmentos
                proAcStDes0 = np.mean(accStaDes0)
                proAcStDes0Hz = np.mean(accStaDes0Hz)

                proAcStDes1Ci = np.mean(accStaDes1Ci)
                proAcStDes1Hz = np.mean(accStaDes1Hz)

                proAcStDes1 = np.mean(accStaDes1)
                proAcStDes2 = np.mean(accStaDes2)
                # Se calcula el ruido ruidoThreshold
                ruidoThreshold0 = proAcStDes0 * 1.15
                ruidoThreshold0Hz = proAcStDes0Hz * 1.15

                ruidoThreshold1Ci = proAcStDes1Ci * 1.15
                ruidoThreshold1Hz = proAcStDes1Hz * 1.15

                ruidoThreshold1 = proAcStDes1 * 1.15
                ruidoThreshold2 = proAcStDes2 * 1.15

                # Comparar contra ruido y eliminar si es mayor
                # ventanas2 = longitud // 50
                ventanas2 = longitud // 20
                y0SinRuido = np.array([])
                y0HzSinRuido = np.array([])

                y1CiSinRuido = np.array([])
                y1HzSinRuido = np.array([])

                y1SinRuido = np.array([])
                y2SinRuido = np.array([])
                for j in range(ventanas2):
                    a = 0
                    inicio = 20 * j
                    fin = inicio + 20

                    y0Analisis = y0[inicio:fin]
                    y0HzAnalisis = y0Hz[inicio:fin]

                    y1CiAnalisis = y1Ci[inicio:fin]
                    y1HzAnalisis = y1Hz[inicio:fin]

                    y1Analisis = y1[inicio:fin]
                    y2Analisis = y2[inicio:fin]

                    # Se calcula la desviación estandar del segmento
                    # y se compara con el ruidoThreshold
                    estDes0 = np.std(y0Analisis, dtype=np.float64)
                    if estDes0 < ruidoThreshold0:
                        y0SinRuido = np.append(y0SinRuido, y0Analisis)

                    estDes0Hz = np.std(y0HzAnalisis, dtype=np.float64)
                    if estDes0Hz < ruidoThreshold0Hz:
                        y0HzSinRuido = np.append(y0HzSinRuido, y0HzAnalisis)

                    estDes1Ci = np.std(y1CiAnalisis, dtype=np.float64)
                    if estDes1Ci < ruidoThreshold1Ci:
                        y1CiSinRuido = np.append(y1CiSinRuido, y1CiAnalisis)

                    estDes1Hz = np.std(y1HzAnalisis, dtype=np.float64)
                    if estDes1Hz < ruidoThreshold1Hz:
                        y1HzSinRuido = np.append(y1HzSinRuido, y1HzAnalisis)

                    estDes1 = np.std(y1Analisis, dtype=np.float64)
                    if estDes1 < ruidoThreshold1:
                        y1SinRuido = np.append(y1SinRuido, y1Analisis)

                    estDes2 = np.std(y2Analisis, dtype=np.float64)
                    if estDes2 < ruidoThreshold2:
                        y2SinRuido = np.append(y2SinRuido, y2Analisis)

                #print("Len(y0): ", len(y0SinRuido))
                #print("Len(y1): ", len(y1CiSinRuido))
                #print("Len(y1): ", len(y1SinRuido))
                #print("Len(y2): ", len(y2SinRuido))

                return y0SinRuido, y0HzSinRuido, y1CiSinRuido, y1HzSinRuido, y1SinRuido, y2SinRuido

            #
            def etapaFiltrado(df1, nombre):
                # etapaFiltrado(df1, nomb)
                #print("etapaFiltrado")
                #print("Info df1")
                #print(df1.info())
                #print('\n' * 2)
                renglon = df1.shape[0]

                #print("Renglones: ", df1.shape[0])
                df3 = df1.iloc[0:renglon, 0:3]
                #print("****Imprimiedo df3=nuevo sin nan****")
                # print(df3)
                #print('\n' * 2)
                # Eliminar columna ch3
                df6 = df3[df3.columns.difference(['ch3'])]
                #print("Info df6")
                #print(df6.info())
                # Eliminar datos nulos
                df6 = df6.dropna()
                # Eliminar duplicados
                df6 = df6.drop_duplicates()
                #print("Info df6")
                #print(df6.info())
                #print("\n")

                #print("Info df1=df3")
                #print(df3.info())
                #print('\n' * 2)

                #print("****Estadisticas  total****")
                # horizontal_stack = pd.concat([df3, df4], axis=1)
                #print(df3.describe(include=[np.number]))
                #print('\n' * 2)

                #print('Correlación Ch1, Ch2 y Ch3')
                #print(df3.corr())
                #print('\n' * 2)

                t_n = 1
                N = 512
                T = t_n / N
                # T= 1/512=0.001953
                f_s = 1 / T
                # f_s =1/0.001953=512 hz

                dt = 1
                x_value = np.arange(0, renglon, dt)

                y0 = df3.iloc[0:renglon, 0]
                y0Hz = df3.iloc[0:renglon, 0]
                y1Ci = df3.iloc[0:renglon, 1]
                y1 = df3.iloc[0:renglon, 1]
                y1Hz = df3.iloc[0:renglon, 1]
                y2 = df3.iloc[0:renglon, 2]

                yinf = -210
                ysup = 210
                inicioDs = 0
                finDs = len(y0)

                #############################################################################
                ###Se quita el 27-04-20
                plotDownSamp(y0, y1Ci, y1, yinf, ysup, inicioDs, finDs, nombre,
                             y0nom="Estómago", y1nom="Ciego", y2nom="Ciego",xnom="30720 muestras = 1 Min")
                ###
                #############################################3
                # filtrado de señal

                #  scipy and numpy have too many future warnings
                import warnings

                warnings.simplefilter(action='ignore', category=FutureWarning)
                from scipy.signal import butter, filtfilt

                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                # Correct the cutoff frequency for the number of passes in the filter

                cieFcBa = 0.008
                # cieFcAl = 0.05
                cieFcAl = 0.7

                # estFcBa = 0.01
                estFcBa = 0.008
                # estFcAl = 0.2
                estFcAl = 0.7

                # ilFcBa = 0.33
                # ilFcBa = 0.01
                ilFcBa = 0.008
                ilFcAl = 0.7

                C = 0.802
                ######################Filtro1
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0paAlEs = filtfilt(bb, aa, y0)

                # Ciego
                bb, aa = butter(2, (cieFcBa / C) / (f_s / 2), btype='high')
                y1paAlCi = filtfilt(bb, aa, y1Ci)

                # Ileon
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1paAlIl = filtfilt(bb, aa, y1)

                # Filtro pasa altas 0.6 hz para señal respiración y ECG
                # Este filtro pasa banda busca tomar la señal de respiración
                bb, aa = butter(2, (0.8 / C) / (f_s / 2), btype='high')
                y2paAl = filtfilt(bb, aa, y2)

                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filtradoPban = filtfilt(b, a, y0paAlEs)  # filter with phase shift correction
                # y0Hz
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0HzfilPbaj = filtfilt(b, a, y0Hz)  # filter with phase shift correction
                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifiltradoPban = filtfilt(b, a, y1paAlCi)  # filter with phase shift correction
                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filtradoPban = filtfilt(b, a, y1paAlIl)  # filter with phase shift correction
                # y1hz
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzfilPbaj = filtfilt(b, a, y1Hz)  # filter with phase shift correction

                # Filtro pasabajas 1 hz para ECG
                b, a = butter(2, (1.7 / C) / (f_s / 2), btype='low')
                y2filtradoPban = filtfilt(b, a, y2paAl)  # filter with phase shift correction

                # Estomago
                y0filtradoPban = y0filtradoPban * 1
                # y0Hz
                y0HzfilPbaj = y0HzfilPbaj * 1
                # Ciego
                y1CifiltradoPban = y1CifiltradoPban * 1
                # y1Hz
                y1HzfilPbaj = y1HzfilPbaj * 1
                # Ileum
                y1filtradoPban = y1filtradoPban * 1
                # Artefactos
                y2filtradoPban = y2filtradoPban * 1

                # plotDownSamp(y0filtradoPban, y1CifiltradoPban, y1filtradoPban, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="Estomago", y1nom="Ciego", y2nom="Ileum Filtro1")

                ######################33
                # Filtro pasa altas
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/Electromyography.ipynb
                # https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
                from scipy.signal import butter, filtfilt

                # Se baja frecuencia de muestreo
                #####Para 2 hz
                t_n = 1  # tiempo en segundos
                N = 2  # Cantidad de muestras en 1 segundo: 2 muestras
                # N = 120  # Cantidad de muestras en 60 segundos
                T = t_n / N
                f_s = 1 / T

                #print("y0filtradoPban")
                longitudy0 = len(y0filtradoPban)
                #print(len(y0filtradoPban))

                y0DoSam = np.array([])
                y0HzDoSam = np.array([])
                y1CiDoSam = np.array([])
                y1HzDoSam = np.array([])
                y1DoSam = np.array([])
                y2DoSam = np.array([])

                for i in range(0, longitudy0, 256):
                    y0DoSam = np.append(y0DoSam, y0filtradoPban[i])
                    y0HzDoSam = np.append(y0HzDoSam, y0HzfilPbaj[i])
                    y1CiDoSam = np.append(y1CiDoSam, y1CifiltradoPban[i])
                    y1HzDoSam = np.append(y1HzDoSam, y1HzfilPbaj[i])
                    y1DoSam = np.append(y1DoSam, y1filtradoPban[i])
                    y2DoSam = np.append(y2DoSam, y2filtradoPban[i])

                #print("len(y0DoSam)")
                longDownSam = len(y0DoSam)
                #print(len(y0DoSam))

                C = 0.802
                ######################Filtro2 con señal down sampled

                # Se agrega filtro pasa altas para tomar en la señal del estomago las frecuencias del estomago;
                # En la señal de ciego las frecuencia de ciego, e ileum
                # Estomago
                bb, aa = butter(2, (estFcBa / C) / (f_s / 2), btype='high')
                y0filPaAltDoSam = filtfilt(bb, aa, y0DoSam)

                # Ileum
                bb, aa = butter(2, (ilFcBa / C) / (f_s / 2), btype='high')
                y1filPaAltDoSam = filtfilt(bb, aa, y1DoSam)

                # filtro pasabajas
                # Estomago
                b, a = butter(2, (estFcAl / C) / (f_s / 2), btype='low')
                y0filPaBaDoSam0 = filtfilt(b, a, y0filPaAltDoSam)  # filter with phase shift correction

                # y0HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y0HzDoSamPaBaj = filtfilt(b, a, y0HzDoSam)  # filter with phase shift correction

                # Ciego
                b, a = butter(2, (cieFcAl / C) / (f_s / 2), btype='low')
                y1CifilPaBaDoSam1 = filtfilt(b, a, y1CiDoSam)  # filter with phase shift correction
                # y1HzDosamp
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1HzDoSamPaBaj = filtfilt(b, a, y1HzDoSam)  # filter with phase shift correction

                # Ileum
                b, a = butter(2, (ilFcAl / C) / (f_s / 2), btype='low')
                y1filPaBaDoSam1 = filtfilt(b, a, y1filPaAltDoSam)  # filter with phase shift correction

                # Artefactos
                b, a = butter(2, (.75 / C) / (f_s / 2), btype='low')
                y2filPaBaDoSam = filtfilt(b, a, y2DoSam)  # filter with phase shift correction

                inicioDs = 0
                finDs = longDownSam
                # Se corta la señal en amplitud
                # Estomago
                np.clip(y0filPaBaDoSam0, -450, 450, out=y0filPaBaDoSam0)
                # Sensor Estomago con pasabajas de 0.75 hz
                np.clip(y0HzDoSamPaBaj, -450, 450, out=y0HzDoSamPaBaj)
                # Frecuencias Ciego
                np.clip(y1CifilPaBaDoSam1, -450, 450, out=y1CifilPaBaDoSam1)
                # Sensor Ciego con pasabajas de 0.75 hz
                np.clip(y1HzDoSamPaBaj, -450, 450, out=y1HzDoSamPaBaj)
                # Ileon
                np.clip(y1filPaBaDoSam1, -450, 450, out=y1filPaBaDoSam1)
                # Artefactos
                np.clip(y2filPaBaDoSam, -450, 450, out=y2filPaBaDoSam)
                # Señales Estomago, Ciego, Ileum
                # plotDownSamp(y0filPaBaDoSam0, y1CifilPaBaDoSam1, y1filPaBaDoSam1, yinf, ysup, inicioDs, finDs, nombre,
                #             y0nom="DS Estomago", y1nom="Ds Ciego", y2nom="Ds Ileum Filtro2")
                # Señales sensor Estomago y Ciego con frecuencia corte a 0.75 Hz
                # plotDownSamp(y0HzDoSamPaBaj, y0HzDoSamPaBaj, y1HzDoSamPaBaj, yinf, ysup, inicioDs, finDs, nombre,
                #            y0nom="DS Estomago 0Hz", y1nom="Ds Estomago 0Hz", y2nom="Ds Ileum 1Hz")


                y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui = eliminaRuido(y0filPaBaDoSam0,
                                                                                                y0HzDoSamPaBaj,
                                                                                                y1CifilPaBaDoSam1,
                                                                                                y1HzDoSamPaBaj,
                                                                                                y1filPaBaDoSam1,
                                                                                                y2filPaBaDoSam)

                longy0SinR = len(y0SinRui)
                longy0HzSinR = len(y0HzSinRui)

                longy1CiSinR = len(y1CiSinRui)
                longy1HzSinR = len(y1HzSinRui)

                longy1SinR = len(y1SinRui)
                longy2SinR = len(y2SinRui)
                # Gurada archivo y0
                # guardarArchivo(y0SinRui, 'y0sinRuido.txt')

                # Gurada archivo y1
                # guardarArchivo(y1SinRui, 'y1sinRuido.txt')

                # Gurada archivo y2
                # guardarArchivo(y2SinRui, 'y2sinRuido.txt')

                return y0SinRui, y0HzSinRui, y1CiSinRui, y1HzSinRui, y1SinRui, y2SinRui, longy0SinR, longy0HzSinR, longy1CiSinR, longy1HzSinR, longy1SinR, longy2SinR, T, N, f_s, t_n

            #
            def extractFeature(signalch0, signalch0Hz, signalch1, signalch1Hz, signalch2, f_s, longy0, longy0Hzsr,
                               longy1, longy1Hzsr, longy2, etiquetaSenal, nombreExCa):

                # signalch0: Estomago : longy0
                # signalch1: Ciego: longy1
                # signalch2: Ileon: longy2

                #print("################# nombreExCa   #########################")
                #print(nombreExCa)
                #print("etiquetaSenal")
                #print(etiquetaSenal)

                #print("f_s")
                #print(f_s)
                yinf = -28
                ysup = 28
                # Estomago
                y0filPaBaDoSam = signalch0
                y0HzfilPaBaDoSam = signalch0Hz

                # Ciego
                y1filPaBaDoSam = signalch1
                y1HzfilPaBaDoSam = signalch1Hz

                # Ileum
                y2filPaBaDoSam = signalch2

                np.clip(y0filPaBaDoSam, -150, 150, out=y0filPaBaDoSam)
                np.clip(y0HzfilPaBaDoSam, -150, 150, out=y0HzfilPaBaDoSam)

                np.clip(y1filPaBaDoSam, -150, 150, out=y1filPaBaDoSam)
                np.clip(y1HzfilPaBaDoSam, -150, 150, out=y1HzfilPaBaDoSam)

                np.clip(y2filPaBaDoSam, -150, 150, out=y2filPaBaDoSam)
                # Señales filtras para tato: Filtro 1, Down Sampled, Filttro2

                # Gráfica señales
                inicioDsHz = 0
                longMenorHz = np.min(np.array([longy0Hzsr, longy1Hzsr]))

                finDsHz = longMenorHz
                #################################################3
                #### Se quita el 20-04-20
                plotDownSamp(y0HzfilPaBaDoSam, y0HzfilPaBaDoSam, y1HzfilPaBaDoSam, yinf, ysup, inicioDsHz, finDsHz,
                             nombreExCa,
                             y0nom="Estómago: y0 ", y1nom="Est. y0 Hz", y2nom="Ciego: y1",xnom="120 muestras = 1 Min")

                return 1


            df1 = self.df

            # Filtrado
            nomb = "C5"
            y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam, y1HzFilSinRui, y1filPaBaDoSam, y2filPaBaDoSam, longy0, longy0HzSR, longy1Ci, longy1HzSR, longy1, longy2, T, N, f_s, t_n = etapaFiltrado(
            df1, nomb)
            # *************************** Llama funcion extracción de características
            nomb = "C5 Filtrada"
            a0=extractFeature(y0filPaBaDoSam, y0HzFilSinRui, y1CifilPaBaDoSam,
                                                           y1HzFilSinRui, y1filPaBaDoSam, f_s, longy0, longy0HzSR,
                                                           longy1Ci, longy1HzSR, longy1, 6, nomb)
    #
    def plotmeanPsdC5(self):
        #x=self.df['col1']
        resultado_st = "plotmeanPsdC5" + "\n"
        self.resultado.setText(resultado_st)
        nomb2="C5"
        #
        def estadisticaPotencia(potenciaEnFreqCh1, potenciaEnFreqCh2, nomEP):
            # estadisticaPotencia(caracteristicasFreqBasalCh1, caracteristicasFreqBasalCh2)
            # mediaPotenciaCh1 = np.median(potenciaEnFreqCh1, axis=0)
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            #print(" SSSSSSSS len(potenciaEnFreqCh1) SSSSSSSSSSSSSSSS")
            #print(len(potenciaEnFreqCh1))
            # stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)

            # stErrCh1=stdPotenciaCh1/(np.sqrt(len(stdPotenciaCh1)))
            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))


            #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            # plt.plot(range(1, mediaPotenciaCh1.shape[0] + 1), mediaPotenciaCh1, "b")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "black", label='y0:Estómago')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='y1:Ciego')
            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.35  # the width of the bars
            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width / 2, mediaPotenciaCh1, width,color="blue", label='y0:Estómago', yerr=stErrCh1)
            rects2 = ax.bar(x + width / 2, mediaPotenciaCh2, width,color="r", label='y1:Ciego', yerr=stErrCh2)
            ax.set_ylabel('PA [uV**2] y stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz')
            #ax.set_xticks(x)
            ax.set_xscale('linear')

            ax.legend()
            plt.show()

            return 0
        #
        estadisticaPotencia(self.C5Ch1, self.C5Ch2, nomb2)

    #mean Psd 5 condiciones y0
    def plotPSDch1(self):

        #
        resultado_st = "C5plotmeanPSDy0" + "\n"
        self.resultado.setText(resultado_st)
        def estadisticaPotencia3Gps(potenciaEnFreqCh1, potenciaEnFreqCh2, potenciaEnFreqCh3, potenciaEnFreqCh4,potenciaEnFreqCh5, nomEP):
            # https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)
            mediaPotenciaCh3 = np.mean(potenciaEnFreqCh3, axis=0, dtype=np.float64)
            mediaPotenciaCh4 = np.mean(potenciaEnFreqCh4, axis=0, dtype=np.float64)
            mediaPotenciaCh5 = np.mean(potenciaEnFreqCh5, axis=0, dtype=np.float64)

            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)
            stdPotenciaCh3 = np.std(potenciaEnFreqCh3, axis=0, dtype=np.float64)
            stdPotenciaCh4 = np.std(potenciaEnFreqCh4, axis=0, dtype=np.float64)
            stdPotenciaCh5 = np.std(potenciaEnFreqCh5, axis=0, dtype=np.float64)

            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))
            stErrCh3 = stdPotenciaCh3 / (np.sqrt(len(potenciaEnFreqCh3)))
            stErrCh4 = stdPotenciaCh4 / (np.sqrt(len(potenciaEnFreqCh4)))
            stErrCh5 = stdPotenciaCh5 / (np.sqrt(len(potenciaEnFreqCh5)))

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "b", label='C1')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='C2')
            plt.plot(x * 0.007813, mediaPotenciaCh3, "g", label='C3')
            plt.plot(x * 0.007813, mediaPotenciaCh4, "yellow", label='C4')
            plt.plot(x * 0.007813, mediaPotenciaCh5, "orange", label='C5')

            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()

            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.20  # the width of the bars
            fig, ax = plt.subplots()
            #rects1 = ax.bar(x - width, mediaPotenciaCh1, width, label='C1', yerr=stErrCh1)
            rects1 = ax.bar(x - 2*width, mediaPotenciaCh1, width-0.003,color="blue", label='C1', yerr=stErrCh1)
            #rects2 = ax.bar(x, mediaPotenciaCh2, width, label='C2', yerr=stErrCh2)
            rects2 = ax.bar(x - width, mediaPotenciaCh2, width,color="r", label='C2', yerr=stErrCh2)
            #rects3 = ax.bar(x + width, mediaPotenciaCh3, width, label='C3', yerr=stErrCh3)
            rects3 = ax.bar(x, mediaPotenciaCh3, width,color="g", label='C3', yerr=stErrCh3)
            #Se agrega rects 4
            rects4 = ax.bar(x + width, mediaPotenciaCh4, width,color="yellow", label='C4', yerr=stErrCh4)
            # Se agrega rects 5
            rects5 = ax.bar(x + width*2, mediaPotenciaCh5, width-0.003,color="orange", label='C5', yerr=stErrCh5)

            ax.set_ylabel('PA [uV**2], stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz.')
            #ax.set_xticks(x)
            ax.set_xscale('linear')
            ax.legend()
            plt.show()

            return 0
        #
        nomEP = "C1, ..., C5 y0"
        estadisticaPotencia3Gps(self.C1Ch1, self.C2Ch1, self.C3Ch1, self.C4Ch1,self.C5Ch1, nomEP)
    # mean Psd 5 condiciones y1
    def plotPSDch2(self):
        #
        resultado_st = "C5plotmeanPSDy1" + "\n"
        self.resultado.setText(resultado_st)
        def estadisticaPotencia3Gps(potenciaEnFreqCh1, potenciaEnFreqCh2, potenciaEnFreqCh3, potenciaEnFreqCh4,potenciaEnFreqCh5, nomEP):
            # https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
            mediaPotenciaCh1 = np.mean(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            mediaPotenciaCh2 = np.mean(potenciaEnFreqCh2, axis=0, dtype=np.float64)
            mediaPotenciaCh3 = np.mean(potenciaEnFreqCh3, axis=0, dtype=np.float64)
            mediaPotenciaCh4 = np.mean(potenciaEnFreqCh4, axis=0, dtype=np.float64)
            mediaPotenciaCh5 = np.mean(potenciaEnFreqCh5, axis=0, dtype=np.float64)

            stdPotenciaCh1 = np.std(potenciaEnFreqCh1, axis=0, dtype=np.float64)
            stdPotenciaCh2 = np.std(potenciaEnFreqCh2, axis=0, dtype=np.float64)
            stdPotenciaCh3 = np.std(potenciaEnFreqCh3, axis=0, dtype=np.float64)
            stdPotenciaCh4 = np.std(potenciaEnFreqCh4, axis=0, dtype=np.float64)
            stdPotenciaCh5 = np.std(potenciaEnFreqCh5, axis=0, dtype=np.float64)

            stErrCh1 = stdPotenciaCh1 / (np.sqrt(len(potenciaEnFreqCh1)))
            stErrCh2 = stdPotenciaCh2 / (np.sqrt(len(potenciaEnFreqCh2)))
            stErrCh3 = stdPotenciaCh3 / (np.sqrt(len(potenciaEnFreqCh3)))
            stErrCh4 = stdPotenciaCh4 / (np.sqrt(len(potenciaEnFreqCh4)))
            stErrCh5 = stdPotenciaCh5 / (np.sqrt(len(potenciaEnFreqCh5)))

            x = np.arange(len(mediaPotenciaCh1))
            figura = plt.figure()
            plt.title(nomEP)
            plt.ylim(0, 6000)
            plt.xlabel("Frecuencia 0-1 Hz.")
            plt.ylabel("mean PSD [uV**2 / Hz]")
            plt.plot(x * 0.007813, mediaPotenciaCh1, "b", label='C1')
            plt.plot(x * 0.007813, mediaPotenciaCh2, "r", label='C2')
            plt.plot(x * 0.007813, mediaPotenciaCh3, "g", label='C3')
            plt.plot(x * 0.007813, mediaPotenciaCh4, "yellow", label='C4')
            plt.plot(x * 0.007813, mediaPotenciaCh5, "orange", label='C5')

            plt.legend(frameon=False, fontsize=10)
            plt.grid(True)
            #plt.xticks(x * 0.007813)
            plt.xscale('linear')
            plt.show()


            x = np.arange(len(mediaPotenciaCh1))  # the label locations
            width = 0.2  # the width of the bars
            fig, ax = plt.subplots()
            #rects1 = ax.bar(x - width, mediaPotenciaCh1, width, label='C1', yerr=stErrCh1)
            rects1 = ax.bar(x - 2*width, mediaPotenciaCh1, width-0.003,color="blue", label='C1', yerr=stErrCh1)
            #rects2 = ax.bar(x, mediaPotenciaCh2, width, label='C2', yerr=stErrCh2)
            rects2 = ax.bar(x-width, mediaPotenciaCh2, width,color="r", label='C2', yerr=stErrCh2)
            #rects3 = ax.bar(x + width, mediaPotenciaCh3, width, label='C3', yerr=stErrCh3)
            rects3 = ax.bar(x, mediaPotenciaCh3, width,color="g", label='C3', yerr=stErrCh3)
            #Se agrega rects 4
            rects4 = ax.bar(x + width, mediaPotenciaCh4, width,color="yellow", label='C4', yerr=stErrCh4)
            rects5 = ax.bar(x + width * 2, mediaPotenciaCh5, width-0.003,color="orange", label='C5', yerr=stErrCh5)

            ax.set_ylabel('PA [uV**2], stdErr')
            ax.set_title(nomEP)
            ax.set_xlabel('Frecuencia 0-1 Hz.')
            #ax.set_xticks(x)
            ax.set_xscale('linear')
            ax.legend()
            plt.show()

            return 0
        #
        nomEP = "C1, ..., C5, y1"
        estadisticaPotencia3Gps(self.C1Ch2, self.C2Ch2, self.C3Ch2, self.C4Ch2,self.C5Ch2, nomEP)

    #Restablece memorias
    def resC1(self):
        resultado_st = "resC1" + "\n"
        self.resultado.setText(resultado_st)

        self.C1Ch1 = np.empty((0, 129))
        print("C1Ch1")
        print(self.C1Ch1)

        self.C1Ch2 = np.empty((0, 129))
        print("C1Ch2")
        print(self.C1Ch2)

    def resC2(self):
        resultado_st = "resC2" + "\n"
        self.resultado.setText(resultado_st)

        self.C2Ch1 = np.empty((0, 129))
        print("C2Ch1")
        print(self.C2Ch1)

        self.C2Ch2 = np.empty((0, 129))
        print("C2Ch2")
        print(self.C2Ch2)

    def resC3(self):
        resultado_st = "resC3" + "\n"
        self.resultado.setText(resultado_st)

        self.C3Ch1 = np.empty((0, 129))
        print("C3Ch1")
        print(self.C3Ch1)

        self.C3Ch2 = np.empty((0, 129))
        print("C3Ch2")
        print(self.C3Ch2)

    def resC4(self):
        resultado_st = "resC4" + "\n"
        self.resultado.setText(resultado_st)

        self.C4Ch1 = np.empty((0, 129))
        print("C4Ch1")
        print(self.C4Ch1)

        self.C4Ch2 = np.empty((0, 129))
        print("C4Ch2")
        print(self.C4Ch2)

    def resC5(self):
        resultado_st = "resC5" + "\n"
        self.resultado.setText(resultado_st)

        self.C5Ch1 = np.empty((0, 129))
        print("C5Ch1")
        print(self.C5Ch1)

        self.C5Ch2 = np.empty((0, 129))
        print("C5Ch2")
        print(self.C5Ch2)
    #

    def abrirVentanaPrincipal(self):
        self.parent().show()
        self.close()

class ventanaAyu(QtWidgets.QMainWindow, Ui_ayuda):
#class ventanaAyu(QMainWindow):
    def __init__(self, parent=None):
        super(ventanaAyu, self).__init__(parent)
        #loadUi('ayuda.ui', self)
        Ui_ayuda.__init__(self)
        self.setupUi(self)
        self.boton6_4.clicked.connect(self.abrirVentanaPrincipal)

    def conversion(self):
        b = 1

    def abrirVentanaPrincipal(self):
        self.parent().show()
        self.close()



app = QApplication(sys.argv)
main = VentanaPrincipal()
main.show()

#main3= ventanaTres()
#main3.show()

sys.exit(app.exec_())
