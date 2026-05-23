from RL import MDPsim, SARSA, Q_learning, PoliticaGreedy
from random import randint


class BlackJack(MDPsim):

    def __init__(self, gama=1):

        self.gama = gama

        # Estados:
        # (suma_jugador, carta_visible_crupier, as_usable)

        self.estados = []

        for suma in range(12, 22):
            for carta_crupier in range(1, 11):
                for as_usable in [True, False]:

                    self.estados.append(
                        (suma, carta_crupier, as_usable)
                    )

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