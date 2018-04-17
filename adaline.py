from math import fabs
import numpy as np


class Adaline(object):
    def __init__(self, taxa_de_apredizagem=0.05, precisao=0.000001, max_epocas=1000, pesos=None):
        self.__taxa_de_apredizagem = taxa_de_apredizagem
        self.__precisao = precisao
        self.__max_epocas = max_epocas
        self.__pesos = pesos
        
    @property
    def pesos(self):
        return self.__pesos
    
    @property
    def percentual_de_acertos(self):
        return self.__percentual_de_acertos
    
    def __EQM(self):
        EQM = 0
        
        for i in range(self.__X_treino.shape[0]):            
            u = self.__pesos[0]

            for j in range(self.__X_treino.shape[1]):
                u += self.__pesos[j + 1] * self.__X_treino[i][j]

            y = 1 if u >= 0 else -1
            
            EQM += (self.__y_treino[i] - y) ** 2 / self.__X_treino.shape[0]
            
        return EQM
    
    def treinar(self, X_treino, y_treino):
        self.__X_treino = X_treino
        self.__y_treino = y_treino
        self.__pesos = np.random.rand(self.__X_treino.shape[1] + 1)
        
        for i in range(self.__X_treino.shape[0]):
            n_epocas = 0
            
            while(n_epocas < self.__max_epocas):
                EQM_anterior = self.__EQM()
                u = self.__pesos[0]

                for j in range(self.__X_treino.shape[1]):
                    u += self.__pesos[j + 1] * self.__X_treino[i][j]

                y = 1 if u >= 0 else -1
                
                self.__pesos[0] += self.__taxa_de_apredizagem * (self.__y_treino[i] - y)
                
                for j in range(self.__X_treino.shape[1]):
                    self.__pesos[j + 1] += self.__taxa_de_apredizagem * (self.__y_treino[i] - y) * self.__X_treino[i][j]
                    
                if fabs(self.__EQM() - EQM_anterior) <= self.__precisao:
                    break
                
                n_epocas += 1                
                    
    def teste(self, X_teste, y_teste):
        self.__X_teste = X_teste
        self.__y_teste = y_teste        
        n_erros = 0

        for i in range(self.__X_teste.shape[0]):
            u = self.__pesos[0]

            for j in range(self.__X_teste.shape[1]):
                u += self.__pesos[j + 1] * self.__X_teste[i][j]

            y = 1 if u >= 0 else -1

            if self.__y_teste[i] - y != 0:
                n_erros += 1

        self.__percentual_de_acertos = 1 - n_erros / self.__X_teste.shape[0]
