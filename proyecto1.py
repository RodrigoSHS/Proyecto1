import numpy as np
from typing import Dict, List, Set, Optional
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

class Graph:
    """Clase base para representar un grafo con operaciones basicas."""
    
    def __init__(self):
        """Inicializa un grafo vacío con nodos y aristas."""
        self.nodes: Set[int] = set()
        self.edges: Dict[int, Set[int]] = {}
        self.weights: Dict[tuple, float] = {}
    
    def add_node(self, node: int) -> None:
        """
        Añade un nodo al grafo.
        """
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = set()
    
    def add_edge(self, source: int, target: int, weight: float = 1.0) -> None:
        """
        Añade una arista dirigida entre dos nodos del grafo.
        """
        self.add_node(source)
        self.add_node(target)
        self.edges[source].add(target)
        self.weights[(source, target)] = weight
    
    def get_neighbors(self, node: int) -> Set[int]:
        """
        Devuelve los nodos vecinos de un nodo dado.
        
        Args:
            node (int): Identificador del nodo
            
        Returns:
            Set[int]: Conjunto de nodos vecinos
        """
        return self.edges.get(node, set())

class Network(Graph):
    """Clase que extiende Graph para realizar análisis de redes."""
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Crea la matriz de adyacencia de la red.
        
        np.ndarray: Matriz de adyacencia que muestra las conexiones entre nodos
        """
        n = len(self.nodes)
        node_to_index = {node: i for i, node in enumerate(sorted(self.nodes))}
        matrix = np.zeros((n, n))
        
        for source in self.edges:
            for target in self.edges[source]:
                i, j = node_to_index[source], node_to_index[target]
                matrix[i, j] = self.weights.get((source, target), 1.0)
        
        return matrix
    
    def visualize(self) -> None:
        """Visualiza la red usando networkx y matplotlib para mostrar los nodos y sus conexiones."""
        G = nx.DiGraph()
        for source in self.edges:
            for target in self.edges[source]:
                G.add_edge(source, target, weight=self.weights.get((source, target), 1.0))
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, arrowsize=20)
        plt.title("Visualización de la Red")
        plt.show()

class PageRank:
    """Implementación del algoritmo PageRank para calcular la importancia de cada nodo en la red."""
    
    def __init__(self, network: Network):
        """
        Inicializa el calculador de PageRank con la red dada.
        
        Args:
            network (Network): La red sobre la cual se realizará el análisis
        """
        self.network = network
        self.ranks: Optional[Dict[int, float]] = None
    
    def calculate(self, damping_factor: float = 0.85, max_iterations: int = 100, tolerance: float = 1e-6) -> Dict[int, float]:
        """
        Calcula los valores de PageRank para todos los nodos en la red.
        
        Args:
            damping_factor (float): Factor de amortiguación para evitar ciclos infinitos (por defecto es 0.85)
            max_iterations (int): Número máximo de iteraciones para el cálculo
            tolerance (float): Tolerancia para determinar cuándo el cálculo ha convergido
            
        Regresa:
            Diccionario con cada nodo y su valor de PageRank
        """
        n = len(self.network.nodes)
        nodes = sorted(self.network.nodes)
        
        # Inicializar los valores de PageRank con una distribución uniforme
        ranks = {node: 1/n for node in nodes}
        
        for _ in range(max_iterations):
            new_ranks = {}
            max_diff = 0
            
            # Calcular el nuevo valor de PageRank para cada nodo
            for node in nodes:
                rank_sum = 0
                # Buscar los nodos que apuntan al nodo actual
                for source in nodes:
                    if node in self.network.get_neighbors(source):
                        out_degree = len(self.network.get_neighbors(source))
                        if out_degree > 0:
                            rank_sum += ranks[source] / out_degree
                
                new_rank = (1 - damping_factor) / n + damping_factor * rank_sum
                new_ranks[node] = new_rank
                max_diff = max(max_diff, abs(new_rank - ranks[node]))
            
            # Actualizar los valores de PageRank
            ranks = new_ranks
            
            # Verificar si los valores han convergido
            if max_diff < tolerance:
                break
        
        self.ranks = ranks
        return ranks
    
    def get_top_nodes(self, n: int = 5) -> List[tuple]:
        """
        Devuelve los n nodos más importantes según su valor de PageRank.
        
        Args:
            n (int): Número de nodos principales a retornar
            
        Returns:
            List[tuple]: Lista con los nodos y sus valores de PageRank, ordenados de mayor a menor
        """
        if self.ranks is None:
            self.calculate()
        
        sorted_nodes = sorted(self.ranks.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:n]

def main():
    # Cargar el archivo de Excel con la información de las páginas web
    file_path = "Web.xlsx"
    excel_data = pd.ExcelFile(file_path)
    df = excel_data.parse("Hoja1")

    # Crear un diccionario para mapear los indices a los nombres de las páginas web
    index_to_website = dict(zip(df["Index"], df["Website"]))

    # Crear el grafo a partir de los datos del archivo Excel usando los nombres de las páginas web
    network = Network()

    # Añadir las conexiones entre las paginas según los datos del archivo Excel
    for _, row in df.iterrows():
        source = index_to_website[row["Index"]]
        cited_by = row["Cited by"]
        
        # Separar los indices citados (pueden estar separados por comas)
        cited_indices = [int(idx.strip()) for idx in cited_by.split(',')]
        
        # Añadir aristas desde las páginas citadas hacia la página actual
        for idx in cited_indices:
            target = index_to_website[idx]
            network.add_edge(target, source)

    # Visualizar el grafo con los nombres de las páginas web
    network.visualize()

    # Calcular el PageRank de los nodos del grafo
    pagerank = PageRank(network)
    ranks = pagerank.calculate()

    # Mostrar los valores de PageRank calculados para cada página web
    print("\nValores de PageRank:")
    for node, rank in ranks.items():
        print(f"Página {node}: {rank:.4f}")

    # Obtener las 3 páginas principales según su valor de PageRank
    top_nodes = pagerank.get_top_nodes(3)
    print("\nTop 3 páginas por PageRank:")
    for node, rank in top_nodes:
        print(f"Página {node}: {rank:.4f}")

if __name__ == "__main__":
    main()