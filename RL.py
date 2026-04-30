"""
Modulo com modelo de simulaciñon de MDPs y algoritmos de RL

"""

from abc import ABCMeta, abstractmethod
from random import choice, random

class MDPsim(metaclass=ABCMeta):
    def __init__(self, estados, gama):
        self.estados = estados
        self.gama = gama
        
    @abstractmethod
    def estado_inicial(self):
        """
        Devuelve el estado inicial.
        
        """
        raise NotImplementedError("Estado inicial no implementado")
    
    @abstractmethod
    def acciones_legales(self, s):
        """
        Devuelve una lista con las acciones legales en el estado s.
        
        """
        raise NotImplementedError("Acciones legales no implementada")
    
    @abstractmethod
    def recompensa(self, s, a, s_):
        """
        Devuelve la recompensa de la transición s, a, s'.
        
        """
        raise NotImplementedError("Recompensa no implementada")
    
    @abstractmethod
    def transicion(self, s, a):
        """
        Devuelve el estado s'
        
        """
        raise NotImplementedError("Transición no implementada")
    
    def es_terminal(self, s):
        """
        Devuelve True si el estado s es terminal.
        
        """
        return False

class Politica(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, s):
        """
        Devuelve la acción a tomar en el estado s.
        
        """
        raise NotImplementedError("Acción no implementada")
    
class PoliticaAleatoria(Politica):
    def __init__(self, mdp):
        self.mdp = mdp
        
    def __call__(self, s):
        return choice(self.mdp.acciones_legales(s))
    
class PoliticaEgreedy(Politica):
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
        
    def __call__(self, s):
        acciones = [a for (s_, a) in self.Q if s_ == s]
        if random() < self.epsilon:
            return choice(acciones)
        else:
            return max(acciones, key=lambda a: self.Q[(s, a)])
        
    def actualizar(self, Q):
        self.Q = Q.copy()
        
class PoliticaGreedy(Politica):
    def __init__(self, Q):
        self.pi = {s: max((a for (s_, a) in Q if s_ == s), key=lambda a: Q[(s, a)])
                   for s in set(s_ for (s_, _) in Q)}
    
    def __call__(self, s):
        return self.pi.get(s, None)

def TD0(mdp, politica, alfa, n_ep, n_iter):
    """
    Algoritmo de TD(0) para estimar la función de valor de un MDP.
    
    Parámetros:
        mdp: objeto de la clase MDP
        politica: objeto de la clase Politica
        alfa: tasa de aprendizaje
        n_ep: número máximo de episodios
        n_iter: número máximo de iteraciones por episodio
    
    """
    V = {s: 0 for s in mdp.estados}
    
    for _ in range(n_ep):
        s = mdp.estado_inicial()
        for _ in range(n_iter):
            a = politica(s)
            s_ = mdp.transicion(s, a)
            V[s] += alfa * (mdp.recompensa(s, a, s_) + mdp.gama * V[s_] - V[s])
            if mdp.es_terminal(s_):
                break
            s = s_  
    return V

def SARSA(mdp, epsilon, alfa, n_ep, n_iter):
    """
    Algoritmo SARSA para estimar la función de valor de un MDP.
    
    Parámetros:
        mdp: objeto de la clase MDP
        epsilon: probabilidad de exploración
        alfa: tasa de aprendizaje
        n_ep: número de episodios
        n_iter: número de iteraciones
    
    """
    Q = {(s, a): 0 
         for s in mdp.estados if not mdp.es_terminal(s) 
         for a in mdp.acciones_legales(s)}
    pi = PoliticaEgreedy(Q, epsilon)
     
    for _ in range(n_ep):
        s = mdp.estado_inicial()
        a = pi(s)
        for _ in range(n_iter):
            s_ = mdp.transicion(s, a)
            r = mdp.recompensa(s, a, s_)
            if mdp.es_terminal(s_):
                Q[(s, a)] += alfa * (r - Q[(s, a)])
                break
            a_ = pi(s_)
            Q[(s, a)] += alfa * (r + mdp.gama * Q[(s_, a_)] - Q[(s, a)])
            s, a = s_, a_
            pi.actualizar(Q)       
    return Q

def Q_learning(mdp, epsilon, alfa, n_ep, n_iter):
    """
    Algoritmo Q-learning para estimar la función de valor de un MDP.
    
    Parámetros:
        mdp: objeto de la clase MDP
        epsilon: probabilidad de exploración
        alfa: tasa de aprendizaje
        n_ep: número de episodios
        n_iter: número de iteraciones
    
    """
    Q = {(s, a): 0 
         for s in mdp.estados if not mdp.es_terminal(s)
         for a in mdp.acciones_legales(s)}
    pi = PoliticaEgreedy(Q, epsilon)
    
    for _ in range(n_ep):
        s = mdp.estado_inicial()
        for _ in range(n_iter):
            a = pi(s)
            s_ = mdp.transicion(s, a)
            r = mdp.recompensa(s, a, s_)
            if mdp.es_terminal(s_):
                Q[(s, a)] += alfa * (r - Q[(s, a)])
                break
            Q[(s, a)] += alfa * (
                r 
                + mdp.gama * max(Q[(s_, a_)] for a_ in mdp.acciones_legales(s_)) 
                - Q[(s, a)])
            s = s_
            pi.actualizar(Q)
    return Q