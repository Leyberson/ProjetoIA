import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class GeneticAlgorithm:
    def __init__(self, matriz_completa, matriz_original, n_cidades, tam_pop=200, geracoes=500, prob_mutacao=0.25, tipo_mutacao=0):
        self.matriz = matriz_completa  # Matriz completa para cálculos
        self.matriz_original = matriz_original  # Matriz original para visualização
        self.n_cidades = n_cidades
        self.tam_pop = tam_pop
        self.geracoes = geracoes
        self.prob_mutacao = prob_mutacao
        self.tipo_mutacao = tipo_mutacao  # 0 para swap, 1 para inversão

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

    def fitness(self, rota):
        distancia = self.calcular_distancia(rota)
        return 1 / distancia if distancia != float('inf') else 0

    def selecao_torneio(self, populacao, k=3):
        selecionados = random.sample(populacao, k)
        return max(selecionados, key=lambda rota: self.fitness(rota))

    def crossover(self, pai1, pai2):
        tamanho = len(pai1)
        filho = [None] * tamanho
        filho[0] = 0
        
        start, end = sorted(random.sample(range(1, tamanho), 2))
        filho[start:end] = pai1[start:end]
        
        cidades_restantes = [c for c in pai2[1:] if c not in filho[start:end]]
        pos = 1
        for cidade in cidades_restantes:
            while pos < tamanho and filho[pos] is not None:
                pos += 1
            if pos < tamanho:
                filho[pos] = cidade
        
        if None in filho:
            return self.gerar_rota_inicial()
        return filho

    def mutacao_swap(self, rota):
        if random.random() < self.prob_mutacao:
            i, j = random.sample(range(1, len(rota)), 2)
            rota[i], rota[j] = rota[j], rota[i]
        return rota

    def mutacao_inversao(self, rota):
        if random.random() < self.prob_mutacao:
            tamanho = len(rota)
            start, end = sorted(random.sample(range(1, tamanho), 2))
            rota[start:end + 1] = rota[start:end + 1][::-1]
        return rota

    def run(self):
        populacao = [self.gerar_rota_inicial() for _ in range(self.tam_pop)]
        melhor_distancia = float('inf')
        melhor_rota = None
        historico = []

        for geracao in range(self.geracoes):
            nova_populacao = []
            # Elitismo: preservar a melhor rota
            if melhor_rota is not None:
                nova_populacao.append(melhor_rota.copy())

            # Gerar novos indivíduos até completar a população
            while len(nova_populacao) < self.tam_pop:
                pai1 = self.selecao_torneio(populacao)
                pai2 = self.selecao_torneio(populacao)
                filho = self.crossover(pai1, pai2)
                if self.tipo_mutacao == 0:
                    filho = self.mutacao_swap(filho)
                else:
                    filho = self.mutacao_inversao(filho)
                nova_populacao.append(filho)

            populacao = nova_populacao
            melhor_distancia_geracao = float('inf')
            melhor_rota_geracao = None

            # Encontrar a melhor rota da geração
            for rota in populacao:
                dist = self.calcular_distancia(rota)
                if dist < melhor_distancia_geracao:
                    melhor_distancia_geracao = dist
                    melhor_rota_geracao = rota.copy()

            # Atualizar a melhor rota global
            if melhor_distancia_geracao < melhor_distancia:
                melhor_distancia = melhor_distancia_geracao
                melhor_rota = melhor_rota_geracao

            historico.append(melhor_distancia)

        return melhor_rota, melhor_distancia, historico

    def plot_convergence(self, historico):
        plt.plot(historico)
        plt.xlabel('Geração')
        plt.ylabel('Melhor Distância')
        plt.title('Convergência do Algoritmo Genético')
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
        plt.title('Melhor Rota Encontrada')
        plt.show()