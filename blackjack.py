"""
Blackjack simplificado como problema de Aprendizaje por Refuerzo
"""

from RL import MDPsim, SARSA, Q_learning, PoliticaGreedy
from random import randint


class BlackJack(MDPsim):

    def __init__(self, gama=1):

        self.gama = gama

        # ========================================================
        # ESPACIO DE ESTADOS
        # Estado:
        # (suma_jugador, carta_visible_crupier, as_usable)
        # ========================================================

        self.estados = []

        for suma in range(12, 22):
            for carta_crupier in range(1, 11):
                for as_usable in [True, False]:
                    self.estados.append(
                        (suma, carta_crupier, as_usable)
                    )

        # estado terminal
        self.estados.append("TERMINAL")

    # ============================================================
    # REPARTIR CARTA
    # ============================================================

    def reparte_carta(self):
        """
        Devuelve una carta aleatoria.

        Probabilidades:
        - 1..9  -> 1/13
        - 10    -> 4/13
        """

        carta = randint(1, 13)

        if carta >= 10:
            return 10

        return carta

    # ============================================================
    # EVALUAR MANO
    # ============================================================

    def evaluar_mano(self, cartas):
        """
        Calcula:
        - suma total
        - si existe As usable

        Un As usable significa que puede contarse como 11
        sin pasarse de 21.
        """

        suma = sum(cartas)

        ases = cartas.count(1)

        as_usable = False

        # convertir UN As de 1 -> 11 si conviene
        if ases > 0 and suma + 10 <= 21:
            suma += 10
            as_usable = True

        return suma, as_usable

    # ============================================================
    # ESTADO INICIAL
    # ============================================================

    def estado_inicial(self):

        # --------------------------------------------------------
        # Repartir jugador
        # --------------------------------------------------------

        self.cartas_jugador = [
            self.reparte_carta(),
            self.reparte_carta()
        ]

        # --------------------------------------------------------
        # Repartir dealer
        # --------------------------------------------------------

        self.cartas_crupier = [
            self.reparte_carta(),
            self.reparte_carta()
        ]

        # --------------------------------------------------------
        # Evaluar mano jugador
        # --------------------------------------------------------

        suma_jugador, as_usable = self.evaluar_mano(
            self.cartas_jugador
        )

        # carta visible dealer
        carta_visible = self.cartas_crupier[0]

        # --------------------------------------------------------
        # Blackjack natural
        # --------------------------------------------------------

        self.blackjack_natural = (
            suma_jugador == 21 and
            len(self.cartas_jugador) == 2
        )

        # --------------------------------------------------------
        # Si la suma es menor a 12:
        # pedir automáticamente
        # --------------------------------------------------------

        while suma_jugador < 12:

            nueva = self.reparte_carta()

            self.cartas_jugador.append(nueva)

            suma_jugador, as_usable = self.evaluar_mano(
                self.cartas_jugador
            )

        return (
            suma_jugador,
            carta_visible,
            as_usable
        )

    # ============================================================
    # ACCIONES LEGALES
    # ============================================================

    def acciones_legales(self, s):

        if s == "TERMINAL":
            return []

        # 0 -> Stand
        # 1 -> Hit

        return [0, 1]

    # ============================================================
    # TRANSICIÓN
    # ============================================================

    def transicion(self, s, a):

        if s == "TERMINAL":
            return "TERMINAL"

        # ========================================================
        # HIT
        # ========================================================

        if a == 1:

            nueva = self.reparte_carta()

            self.cartas_jugador.append(nueva)

            suma, as_usable = self.evaluar_mano(
                self.cartas_jugador
            )

            # jugador se pasó
            if suma > 21:
                return "TERMINAL"

            return (
                suma,
                s[1],
                as_usable
            )

        # ========================================================
        # STAND
        # ========================================================

        else:

            suma_crupier, as_crupier = self.evaluar_mano(
                self.cartas_crupier
            )

            # política fija dealer:
            # pedir mientras tenga < 17

            while suma_crupier < 17:

                nueva = self.reparte_carta()

                self.cartas_crupier.append(nueva)

                suma_crupier, as_crupier = self.evaluar_mano(
                    self.cartas_crupier
                )

            # guardar suma final dealer
            self.suma_final_crupier = suma_crupier

            return "TERMINAL"

    # ============================================================
    # RECOMPENSA
    # ============================================================

    def recompensa(self, s, a, s_):

        # --------------------------------------------------------
        # estados no terminales
        # --------------------------------------------------------

        if s_ != "TERMINAL":
            return 0

        suma_jugador = s[0]

        # ========================================================
        # SI EL JUGADOR PIDIÓ Y TERMINÓ:
        # necesariamente fue bust
        # ========================================================

        if a == 1:
            return -1

        # ========================================================
        # BLACKJACK NATURAL
        # ========================================================

        if self.blackjack_natural:
            return 1.5

        # ========================================================
        # COMPARAR CON DEALER
        # ========================================================

        suma_crupier = self.suma_final_crupier

        # dealer se pasó
        if suma_crupier > 21:
            return 1

        # jugador gana
        if suma_jugador > suma_crupier:
            return 1

        # jugador pierde
        if suma_jugador < suma_crupier:
            return -1

        # empate
        return 0

    # ============================================================
    # ESTADO TERMINAL
    # ============================================================

    def es_terminal(self, s):

        return s == "TERMINAL"


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    blackjack = BlackJack(gama=1)

    # ============================================================
    # SARSA
    # ============================================================

    Q_sarsa = SARSA(
        blackjack,
        alfa=0.1,
        epsilon=0.1,
        n_ep=50000,
        n_iter=100
    )

    # ============================================================
    # Q-LEARNING
    # ============================================================

    Q_ql = Q_learning(
        blackjack,
        alfa=0.1,
        epsilon=0.1,
        n_ep=50000,
        n_iter=100
    )

    # ============================================================
    # POLÍTICAS ÓPTIMAS
    # ============================================================

    pi_s = PoliticaGreedy(Q_sarsa)

    pi_q = PoliticaGreedy(Q_ql)

    # ============================================================
    # MOSTRAR RESULTADOS
    # ============================================================

    print(
        "Estado".center(25) + "|" +
        "SARSA".center(15) + "|" +
        "Q-learning".center(15)
    )

    print(
        "-" * 25 + "|" +
        "-" * 15 + "|" +
        "-" * 15
    )

    for s in blackjack.estados:

        if not blackjack.es_terminal(s):

            accion_sarsa = pi_s(s)
            accion_q = pi_q(s)

            accion_sarsa = (
                "Stand" if accion_sarsa == 0 else "Hit"
            )

            accion_q = (
                "Stand" if accion_q == 0 else "Hit"
            )

            print(
                str(s).center(25) + "|" +
                accion_sarsa.center(15) + "|" +
                accion_q.center(15)
            )

    print(
        "-" * 25 + "|" +
        "-" * 15 + "|" +
        "-" * 15
    )


"""
===============================================================
RESPUESTAS TEÓRICAS
===============================================================

1. ¿Cuáles son los estados, acciones, recompensas y transiciones?

Estados:
Cada estado se representa como:

(suma_jugador, carta_visible_crupier, as_usable)

- suma_jugador:
  valor entre 12 y 21

- carta_visible_crupier:
  valor entre 1 y 10

- as_usable:
  True o False

La cardinalidad total es:

10 × 10 × 2 = 200 estados


Acciones:
0 -> Stand (plantarse)
1 -> Hit (pedir carta)


Recompensas:
+1   -> victoria
0    -> empate
-1   -> derrota o bust
+1.5 -> blackjack natural


Transiciones:
Dependen de:
- la acción del jugador
- la carta aleatoria repartida
- la política fija del dealer


===============================================================

2. ¿Cómo representar eficientemente los estados?

Se utiliza únicamente la información mínima necesaria
para tomar decisiones óptimas:

- suma del jugador
- carta visible del dealer
- existencia de As usable

No se guarda el historial completo de cartas debido
a la propiedad de Markov.


===============================================================

3. ¿Qué pasa si se modifica epsilon?

epsilon controla la exploración.

- epsilon alto:
  más acciones aleatorias
  más exploración

- epsilon bajo:
  más explotación de la política aprendida

Si epsilon = 0:
el agente deja de explorar y puede quedarse atrapado
en una política subóptima.


===============================================================

4. ¿Cómo afecta alfa?

alfa es la tasa de aprendizaje.

- alfa alto:
  aprendizaje rápido pero inestable

- alfa bajo:
  aprendizaje lento pero más estable

Valores intermedios suelen producir mejores resultados.


===============================================================

5. ¿Qué algoritmo es más adecuado?

Q-learning suele ser más adecuado porque aprende
directamente la política óptima futura.

SARSA es más conservador porque aprende considerando
la política exploratoria actual.


===============================================================

6. ¿Se puede explicar la política óptima?

Sí.

El agente aprende estrategias similares a las usadas
en casinos reales:

- pedir con valores bajos
- plantarse con valores altos
- arriesgar más si el dealer muestra cartas fuertes
- ser conservador si el dealer muestra cartas débiles

Por ejemplo:
- con 20 normalmente conviene plantarse
- con 13 frente a un dealer mostrando 10,
  suele convenir pedir
- con 16 frente a dealer mostrando 6,
  suele convenir plantarse

===============================================================
"""