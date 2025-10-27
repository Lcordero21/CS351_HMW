from typing import Any

class Graph:
    VERTEX_COUNT = 10

    def __init__(self):
        adj_list:dict[str] = {}
        adj_matrix = [graph.VERTEX_COUNT *[graph.VERTEX_COUNT * None]]

    def add_vertex (self, vertex: Any):
        """
        Input: Vertex(Any)
        Output: None
        """
        self.adj_list[vertex] = []

        self.adj_list.append(vertex)

    def add_edge(self, vertex_from: Any, vertex_to: Any):
        #add vertex_to to the vertex_from

        self.adj_list[vertex_from].append(vertex_to)

        self.adj_matrix[vertex_from][vertex_to] = 1
        pass