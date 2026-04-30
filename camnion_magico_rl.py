"""
El camión mágico, pero ahora por simulación

"""

from RL import MDPsim, SARSA, Q_learning, PoliticaGreedy
from random import random, randint

class CamionMagico(MDPsim):
    """
    Clase que representa un MDP para el problema del camión mágico.
    
    Si caminas, avanzas 1 con coso 1
    Si usas el camion, con probabilidad rho avanzas el doble de donde estabas
    y con probabilidad 1-rho te quedas en el mismo lugar. Todo con costo 2.
    
    El objetivo es llegar a la meta en el menor costo posible
    
    """    
    
    def __init__(self, gama, rho, meta):
        self.gama = gama
        self.rho = rho
        self.meta = meta
        self.estados = tuple(range(1, meta + 1))
    
    def estado_inicial(self):
        #return randint(1, self.meta // 2 + 1)
        #return randint(1, self.meta - 1)
        return 1
    
    def acciones_legales(self, s):
        return (['caminar', 'usar_camion'] if s < self.meta // 2 else
                ['caminar'] if s < self.meta else 
                [])
    
    def recompensa(self, s, a, s_):
        return  -1  if a == 'caminar' else -2 
        
    def transicion(self, s, a):
        if a == 'caminar':
            return s + 1
        elif a == 'usar_camion':
            return 2*s if random() < self.rho else s
        
    def es_terminal(self, s):
        return s >= self.meta

mdl = CamionMagico(gama=0.999, rho=0.999, meta=145)
    
Q_sarsa = SARSA(mdl, epsilon=0.2, alfa=0.5,  n_ep=5_000, n_iter=150)
pi_s = PoliticaGreedy(Q_sarsa)

Q_ql = Q_learning(mdl, epsilon=0.2, alfa=0.5,  n_ep=5_000, n_iter=150)
pi_ql = PoliticaGreedy(Q_ql)

print(f"Los tramos donde se debe usar el camión segun SARSA son:")
print([s for s in mdl.estados if pi_s(s) == 'usar_camion'])
print("-"*50)
print(f"Los tramos donde se debe usar el camión segun Qlearning son:")
print([s for s in mdl.estados if pi_ql(s) == 'usar_camion'])
print("-"*50)


"""
**********************************************************************************
Ahora responde a las siguientes preguntas:
**********************************************************************************

- Prueba con diferentes valores de rho. ¿Qué observas? ¿Porqué crees que pase eso?
- Prueba con diferentes valores de gama. ¿Qué observas? ¿Porqué crees que pase eso?
- ¿Qué tan diferente es la política óptima de SARSA y Q-learning?
- ¿Cambia mucho el resultado cambiando los valores de recompensa?
- ¿Cuantas iteraciones se necesitan para que funcionen correctamente los algoritmos?
- ¿Qué pasaria si ahora el estado inicial es cualquier estado de la mitad para abajo?
**********************************************************************************

"""