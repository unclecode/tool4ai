# File: tool4ai/utils/dependency_graph.py

from typing import Dict, List, Set, Tuple
from tool4ai.core.tool import Tool
import graphviz

class CyclicDependencyError(Exception):
    """Exception raised when a cyclic dependency is detected."""
    pass

class DependencyGraph:
    def __init__(self):
        self.nodes: Dict[str, Tool] = {}
        self.edges: Dict[str, Set[Tuple[str, str]]] = {}

    def add_node(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("Node must be an instance of Tool")
        if tool.name in self.nodes:
            raise ValueError(f"Tool with id '{tool.name}' already exists in the graph")
        self.nodes[tool.name] = tool
        if tool.name not in self.edges:
            self.edges[tool.name] = set()

    def add_dependency(self, source_name: str, target_name: str, source_attr: str) -> None:
        if source_name not in self.nodes:
            raise ValueError(f"Source tool '{source_name}' not found in the graph")
        if target_name not in self.nodes:
            raise ValueError(f"Target tool '{target_name}' not found in the graph")
        if source_name == target_name:
            raise ValueError("A tool cannot depend on itself")
        
        self.edges[source_name].add((target_name, source_attr))
        
        if self._has_cycle():
            self.edges[source_name].remove((target_name, source_attr))
            raise CyclicDependencyError("Adding this dependency would create a cycle")

    def _has_cycle(self) -> bool:
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor, _ in self.edges[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return True

        return False

    def visualize(self, filename: str = "dependency_graph") -> None:
        dot = graphviz.Digraph(comment='Dependency Graph')
        dot.attr(rankdir='LR', size='8,5')

        for node_name, tool in self.nodes.items():
            dot.node(node_name, f"{node_name}\n{tool.__class__.__name__}", shape='box')

        for source, dependencies in self.edges.items():
            for target, attr in dependencies:
                dot.edge(source, target, label=attr)

        connected_nodes = set()
        for dependencies in self.edges.values():
            for target, _ in dependencies:
                connected_nodes.add(target)
        standalone_nodes = set(self.nodes.keys()) - connected_nodes

        with dot.subgraph(name='cluster_standalone') as c:
            c.attr(label='Standalone Tools')
            for node in standalone_nodes:
                c.node(node, f"{node}\n{self.nodes[node].__class__.__name__}", shape='box')

        dot.render(filename, view=True, format='png', cleanup=True)
        
    def __repr__(self) -> str:
        return f"DependencyGraph(nodes={list(self.nodes.keys())}, edges={self.edges})"