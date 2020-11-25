# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ayuda.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ayuda(object):
    def setupUi(self, ayuda):
        ayuda.setObjectName("ayuda")
        ayuda.resize(740, 592)
        ayuda.setStyleSheet("background-color: rgb(218, 218, 218);")
        self.centralwidget = QtWidgets.QWidget(ayuda)
        self.centralwidget.setObjectName("centralwidget")
        self.line_19 = QtWidgets.QFrame(self.centralwidget)
        self.line_19.setGeometry(QtCore.QRect(330, 550, 70, 3))
        self.line_19.setStyleSheet("background-color: rgb(255, 170, 0);")
        self.line_19.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_19.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_19.setObjectName("line_19")
        self.boton6_4 = QtWidgets.QPushButton(self.centralwidget)
        self.boton6_4.setGeometry(QtCore.QRect(330, 520, 71, 23))
        self.boton6_4.setStyleSheet("background-color: rgb(255, 186, 129);")
        self.boton6_4.setObjectName("boton6_4")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(50, -10, 651, 481))
        self.label_7.setStyleSheet("background-color: rgb(255, 245, 166);\n"
"font: 9pt \"MS Shell Dlg 2\";\n"
"background-color: rgb(244, 238, 233);")
        self.label_7.setObjectName("label_7")
        ayuda.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ayuda)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 740, 21))
        self.menubar.setObjectName("menubar")
        ayuda.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ayuda)
        self.statusbar.setObjectName("statusbar")
        ayuda.setStatusBar(self.statusbar)

        self.retranslateUi(ayuda)
        QtCore.QMetaObject.connectSlotsByName(ayuda)

    def retranslateUi(self, ayuda):
        _translate = QtCore.QCoreApplication.translate
        ayuda.setWindowTitle(_translate("ayuda", "MainWindow"))
        self.boton6_4.setText(_translate("ayuda", "Regresar"))
        self.label_7.setText(_translate("ayuda", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:1.8%;\"><span style=\" font-family:\'Arial\'; font-weight:600; color:#000000; background-color:#ffffff;\">Programa para evaluar la actividad electromiográfica del sistema digestivo.</span></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:1.8%; font-family:\'Arial\'; font-weight:600; color:#000000;\"><br /></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:1.8%; font-family:\'Arial\'; font-weight:600; color:#000000;\"><br /></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; font-weight:600; color:#000000; background-color:#ffffff;\"><br /></span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; font-weight:600; color:#000000; background-color:#ffffff;\">Los pasos para realizar el análisis son:</span></p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Arial\'; font-weight:600; color:#000000;\"><br /></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">1.- Seleccionar cantidad de condiciones para analizar en la GUI principal.</span></p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">2.-  Cargar archivo CSV de condición 1 dando click en “Importar CSV”.</span></p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:1.8%;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">2.- Cargar archivos CSV de condición 1 dando click en “Importar CSV”.</span></p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:1.8%; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:1.8%; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:1.8%;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">3.- Si desea graficar los datos del archivo importado dar click en “Graficar”. </span></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\"><br /></span></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">4- Continuar cargando archivos de condición hasta se se terminen (paso 2).</span></p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">5.- Se desea graficar la meanPSD y la PA de todos la archivos cargado en una condición</span></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">dar click en “meanPSD”.</span></p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">6.- Cargar archivos CSV de la siguiente codicione siguiendo paso 2.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:1.8%; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:1.8%; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; line-height:1.8%;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">7.- Si desea graficar cada archivo después de cargado puede repetir el paso 3.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">8.- Cuando finalice la carga de archivos en todas las condiciones, con el botón “Graficar meanPSD y</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">PA CH1” puede graficar la meanPSD y la PA del sensor 1 (estómago) de todas las condiciones elegidas.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">9.- El botón “Reiniciar CSVs” le permite reiniciar la carga de archivos CSV del grupo donde se encuentre el botón.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">10.- El botón “Regresar”  le permite regresar a la pantalla principal.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Arial\'; color:#000000;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\">11.- El botón ayuda le muestra ayuda para realizar la evaluación de las señales.</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\'; color:#000000; background-color:#ffffff;\"> </span></p></body></html>"))
