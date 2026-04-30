"""
El problema del jugador pero como un problema de aprendizaje por refuerzo

"""

from RL import MDPsim, SARSA, Q_learning, PoliticaGreedy
from random import random, randint

class Jugador(MDPsim):
    """
    Clase que representa un MDP para el problema del jugador.
    
    El jugador tiene un capital inicial y el objetivo es llegar a un capital
    objetivo o quedarse sin dinero.
    
    """
    def __init__(self, meta, ph, gama):
        self.estados = tuple(range(meta + 1))
        self.meta = meta
        self.ph = ph
        self.gama = gama
        
    def estado_inicial(self):
        return randint(1, self.meta - 1)
    
    def acciones_legales(self, s):
        if s == 0 or s == self.meta:
            return []
        return [i for i in range(1, min(s, self.meta - s) + 1)]
    
    def recompensa(self, s, a, s_):
        return self.meta if s_ == self.meta else 0
    
    def transicion(self, s, a):
        return s + a if random() < self.ph else s - a
    
    def es_terminal(self, s):
        return s == 0 or s == self.meta
    
mdl = Jugador(meta=100, ph=0.40, gama=1)

Q_sarsa = SARSA( mdl, alfa=0.2, epsilon=0.02, n_ep=10_000, n_iter=100)
pi_s = PoliticaGreedy(Q_sarsa)

Q_ql = Q_learning( mdl, alfa=0.2, epsilon=0.02, n_ep=10_000, n_iter=100)
pi_q = PoliticaGreedy(Q_ql)

print("Estado".center(10) + '|' +  "SARSA".center(10) + '|' + "Q-learning".center(10))
print("-"*10 + '|' + "-"*10 + '|' + "-"*10)
for s in mdl.estados:
    if not mdl.es_terminal(s):
        print(str(s).center(10) + '|' 
              + str(pi_s(s)).center(10) + '|' 
              + str(pi_q(s)).center(10))
print("-"*10 + '|' + "-"*10 + '|' + "-"*10)

""" 
***************************************************************************************
Responde las siguientes preguntas:
***************************************************************************************
1. ¿Qué pasa si se modifica el valor de epsilón de la política epsilon-greedy?
2. ¿Para que sirve usar una politica epsilon-greedy?
3. ¿Qué pasa con la política óptima y porqué si p_h es mayor a 0.5?
4. ¿Y si es 0.5?
5. ¿Y si es menor a 0.5?
6. ¿Qué pasa si se modifica el valor de la tasa de aprendizaje?
7. ¿Qué pasa si se modifica el valor de gama?

***************************************************************************************

"""