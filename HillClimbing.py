import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

class HillClimbing:
    def __init__(self, matriz_completa, matriz_original, n_cidades, max_iter=1000, tipo_permutacao=0):
        self.matriz = matriz_completa  # Matriz completa para cálculos
        self.matriz_original = matriz_original  # Matriz original para visualização
        self.n_cidades = n_cidades
        self.max_iter = max_iter
        self.tipo_permutacao = tipo_permutacao  # 0 para swap, 1 para inversão

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

    def gerar_rota_inicial(self):
        cidades = list(range(1, self.n_cidades))
        random.shuffle(cidades)
        return [0] + cidades

    def gerar_vizinhanca(self, rota):
        vizinhanca = []
        if self.tipo_permutacao == 0:  # Swap
            for i in range(1, len(rota)):
                for j in range(i + 1, len(rota)):
                    vizinho = rota.copy()
                    vizinho[i], vizinho[j] = vizinho[j], vizinho[i]
                    vizinhanca.append(vizinho)
        else:  # Inversão
            for i in range(1, len(rota)):
                for j in range(i + 1, len(rota)):
                    vizinho = rota.copy()
                    vizinho[i:j + 1] = vizinho[i:j + 1][::-1]
                    vizinhanca.append(vizinho)
        return vizinhanca

    def run(self):
        start_time = time.time()
        rota_atual = self.gerar_rota_inicial()
        distancia_atual = self.calcular_distancia(rota_atual)
        melhor_rota = rota_atual.copy()
        melhor_distancia = distancia_atual
        historico = [melhor_distancia]

        for _ in range(self.max_iter):
            vizinhanca = self.gerar_vizinhanca(rota_atual)
            melhor_vizinho = rota_atual
            melhor_distancia_vizinho = distancia_atual

            # Avaliar vizinhança
            for vizinho in vizinhanca:
                dist = self.calcular_distancia(vizinho)
                if dist < melhor_distancia_vizinho:
                    melhor_vizinho = vizinho
                    melhor_distancia_vizinho = dist

            # Atualizar solução atual se houver melhora
            if melhor_distancia_vizinho < distancia_atual:
                rota_atual = melhor_vizinho
                distancia_atual = melhor_distancia_vizinho
                if distancia_atual < melhor_distancia:
                    melhor_rota = rota_atual.copy()
                    melhor_distancia = distancia_atual
            else:
                # Parar se não houver melhora (máximo local atingido)
                break

            historico.append(melhor_distancia)

        tempo_execucao = time.time() - start_time
        return melhor_rota, melhor_distancia, historico, tempo_execucao

    def plot_convergence(self, historico):
        plt.plot(historico)
        plt.xlabel('Iteração')
        plt.ylabel('Melhor Distância')
        plt.title('Convergência do Hill Climbing')
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
        plt.title('Melhor Rota Encontrada - Hill Climbing')
        plt.show()