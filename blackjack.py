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