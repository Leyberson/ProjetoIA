import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

class AntColonyOptimization:
    def __init__(self, matriz_completa, matriz_original, n_cidades, n_formigas=50, max_iter=500, alpha=1.0, beta=2.0, rho=0.1, tau_init=1e-6):
        self.matriz = matriz_completa  # Matriz completa para cálculos
        self.matriz_original = matriz_original  # Matriz original para visualização
        self.n_cidades = n_cidades
        self.n_formigas = n_formigas
        self.max_iter = max_iter
        self.alpha = alpha  # Influência do feromônio
        self.beta = beta    # Influência da heurística
        self.rho = rho      # Taxa de evaporação
        self.tau_init = tau_init  # Valor inicial de feromônio

        # Inicializar feromônios
        self.feromonios = np.full((n_cidades, n_cidades), self.tau_init)
        np.fill_diagonal(self.feromonios, 0)  # Sem feromônio em loops

        # Heurística (inverso da distância)
        self.eta = np.zeros((n_cidades, n_cidades))
        for i in range(n_cidades):
            for j in range(n_cidades):
                if i != j and self.matriz[i][j] != float('inf'):
                    self.eta[i][j] = 1.0 / self.matriz[i][j]
                else:
                    self.eta[i][j] = 0

    def calcular_distancia(self, rota):
        distancia = 0
        for i in range(len(rota) - 1):
            if self.matriz[rota[i]][rota[i + 1]] == float('inf'):
                return float('inf')
            distancia += self.matriz[rota[i]][rota[i + 1]]
        if self.matriz[rota[-1]][rota[0]] == float('inf'):
            return float('inf')
        distancia += self.matriz[rota[-1]][rota[0]]
        return distancia

    def construir_rota(self):
        # Começar no nó 0 (cidade 1)
        rota = [0]
        nao_visitados = set(range(1, self.n_cidades))

        # Construir a rota cidade por cidade
        while nao_visitados:
            cidade_atual = rota[-1]
            # Calcular probabilidades para as cidades não visitadas
            probs = []
            cidades_candidatas = list(nao_visitados)
            if not cidades_candidatas:
                break

            soma_prob = 0
            for prox_cidade in cidades_candidatas:
                if self.matriz[cidade_atual][prox_cidade] == float('inf'):
                    probs.append(0)
                else:
                    prob = (self.feromonios[cidade_atual][prox_cidade] ** self.alpha) * (self.eta[cidade_atual][prox_cidade] ** self.beta)
                    probs.append(prob)
                    soma_prob += prob

            # Evitar divisão por zero
            if soma_prob == 0:
                # Se todas as probabilidades são zero, escolher aleatoriamente
                prox_cidade = random.choice(cidades_candidatas)
            else:
                probs = [p / soma_prob for p in probs]
                prox_cidade = random.choices(cidades_candidatas, weights=probs, k=1)[0]

            rota.append(prox_cidade)
            nao_visitados.remove(prox_cidade)

        return rota

    def atualizar_feromonios(self, rotas, distancias):
        # Evaporação
        self.feromonios = (1 - self.rho) * self.feromonios

        # Reforço: cada formiga deposita feromônios
        for rota, distancia in zip(rotas, distancias):
            if distancia == float('inf'):
                continue
            delta_tau = 1.0 / distancia
            for i in range(len(rota) - 1):
                self.feromonios[rota[i]][rota[i + 1]] += delta_tau
            self.feromonios[rota[-1]][rota[0]] += delta_tau

        # Garantir que os feromônios não fiquem abaixo de um valor mínimo
        self.feromonios = np.maximum(self.feromonios, 1e-10)

    def run(self):
        start_time = time.time()
        melhor_distancia = float('inf')
        melhor_rota = None
        historico = []

        for _ in range(self.max_iter):
            # Construir rotas para todas as formigas
            rotas = []
            distancias = []
            for _ in range(self.n_formigas):
                rota = self.construir_rota()
                distancia = self.calcular_distancia(rota)
                rotas.append(rota)
                distancias.append(distancia)

            # Encontrar a melhor rota da iteração
            idx_melhor = np.argmin(distancias)
            melhor_distancia_iter = distancias[idx_melhor]
            melhor_rota_iter = rotas[idx_melhor]

            # Atualizar a melhor rota global
            if melhor_distancia_iter < melhor_distancia:
                melhor_distancia = melhor_distancia_iter
                melhor_rota = melhor_rota_iter.copy()

            # Atualizar feromônios
            self.atualizar_feromonios(rotas, distancias)

            # Armazenar a melhor distância da iteração
            historico.append(melhor_distancia)

        tempo_execucao = time.time() - start_time
        return melhor_rota, melhor_distancia, historico, tempo_execucao

    def plot_convergence(self, historico):
        plt.plot(historico)
        plt.xlabel('Iteração')
        plt.ylabel('Melhor Distância')
        plt.title('Convergência do Algoritmo de Colônia de Formigas')
        plt.show()

    def plot_route(self, melhor_rota):
        G = nx.Graph()
        for i in range(self.n_cidades):
            for j in range(i + 1, self.n_cidades):
                if self.matriz_original[i][j] != 0:
                    G.add_edge(i + 1, j + 1, weight=self.matriz_original[i][j])
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        edges = [(melhor_rota[i] + 1, melhor_rota[i + 1] + 1) for i in range(len(melhor_rota) - 1)] + [(melhor_rota[-1] + 1, melhor_rota[0] + 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', width=2)
        plt.title('Melhor Rota Encontrada - Colônia de Formigas')
        plt.show()