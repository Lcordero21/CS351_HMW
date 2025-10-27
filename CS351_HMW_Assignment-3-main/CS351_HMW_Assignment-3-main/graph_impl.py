from graph_interfaces import IEdge, IGraph, IVertex
from typing import Dict, List, Protocol, TypeVar, Tuple, Optional

# Implementation definitions
# You should implement the bodies of the methods required by the interface protocols.

T= TypeVar('T')

class Graph[T](IGraph[T]):

    def __init__(self)-> None:
        """
        Purpose: 
            To initialize the graph with an adjacency list (which I commented out) and a list for vertices
        """
        #self._adj_list: Dict[str, List[str]] = {}
        self._vertices: List[IVertex] = []

    def get_vertices(self) -> List[IVertex]: 
        """
        Purpose: 
            To retrieve stored vertices.
        """
        return self._vertices

    def get_edges(self, vertex: IVertex) -> List[IEdge]: 
        """
        Purpose: 
            To retrieve edges of a specified vertex
        """
        return vertex.get_edges()

    def add_vertex(self, vertex_name: str) -> None: 
        """
        Purpose:
            To add a new vertex to the graph.
        """
        for vertex in self._vertices:
            if vertex.get_name() == vertex_name:
                return None
        self._vertices.append(Vertex(vertex_name))

    def remove_vertex(self, vertex_name: str) -> None: 
        """
        Purpose:
            To remove a vertex from the self._vertices list.
        """
        for vertex in self._vertices:
            if vertex.get_name() == vertex_name:
                self._vertices.remove(self._vertices[vertex])
        

    def add_edge(self, edge: IEdge, from_vertex_name: str, to_vertex_name: str, weight: float) -> None: 
        """
        Purpose:
            To add a new edge (by adding a connection between the two vertices)
        """
        from_vertex = None
        to_vertex = None

        for vertex in self._vertices:
            if vertex.get_name() == from_vertex_name:
                from_vertex = vertex

            if vertex.get_name() == to_vertex_name:
                to_vertex = vertex
        
        if from_vertex is None or to_vertex is None:
            raise Exception("One or more of the vertexes do not exist")

        the_edge = Edge(edge, to_vertex, weight)
        from_vertex.add_edge(the_edge)

        second_edge = Edge(edge, from_vertex, weight)
        to_vertex.add_edge(second_edge)



class Vertex(IVertex):
    def __init__(self, name):
        """
        Purpose:
            To initialize the vertex with _visited (which stores a boolean), 
            _name (the vertex name), and _edges(which contains all the edges 
            connected to the vertex)
        """
        self._visited: bool = False
        self._name: str = name
        self._edges: List[IEdge] = []

    def get_name(self) -> str: 
        """
        Purpose:
            Returns the name of the vertex.
        """
        return self._name
    
    def set_name(self, name: str) -> None:
        """
        Purpose:
            Sets the name of the vertex (in case it needs to be changed)
        """
        self._name = name

    def add_edge(self, edge) -> None: 
        """
        Purpose:
            Adds a new edge to the vertex.
        """
        self._edges.append(edge)

    def remove_edge(self, edge_name) -> None:
        """
        Purpose: 
            Removes an edge from the vertex (it may not be correct though...).
        """
        for edge in self._edges:
            if edge.get_name() == edge_name:
                self._edges.remove(edge)
                return
            
    def get_edges(self) -> List[IEdge]: 
        """
        Purpose:
            Returns the object addresses of all the edges stored in the _edges list.
        """
        return self._edges

    def set_visited(self, visited: bool) -> None: 
        """
        Purpose:
            Sets the vertex's visited status to a new boolean (or the same. It doesn't change much.)
        """
        self._visited = visited

    def is_visited(self) -> bool: 
        """
        Purpose:
            Checks to see the vertex's visited status (a boolean).
        """
        return self._visited

class Edge(IEdge):
    def __init__(self, name: str, destination: IVertex, weight: float) -> None:
        """
        Purpose:
            Initializes a new vertex with it's name, it's destination, and it's weight.
        """
        self._name: str = name
        self._destination: IVertex = destination
        self._weight: float = weight

    def get_name(self) -> str: 
        """
        Purpose:
            Returns the name of the edge.
        """
        return self._name
    
    def set_name(self, name: str) -> None: 
        """
        Purpose:
            Sets the name of the Edge to something new.
        """
        self._name = name

    def get_destination(self) -> IVertex: 
        """
        Purpose:
            Gets the destination of the edge.
        """
        return self._destination
    
    def get_weight(self) -> float: 
        """
        Purpose:
            Gets the weight of the edge.
        """
        return self._weight
    
    def set_weight(self, weight: float) -> None:
        """
        Purpose:
            Sets the weight of the edge.
        """
        self._weight = weight
