# visualization.py
import json
import graphviz
import importlib.resources
from typing import Dict, Any

class GraphVisualizer:
    def visualize(self, graph, output_file: str = "tool_dependency_graph"):
        dot = graphviz.Digraph(comment="Tool Dependency Graph")
        dot.attr(rankdir="LR")  # Left to right layout

        # Create nodes (excluding non-actionable sub-queries)
        for level, indices in enumerate(graph.get_execution_order()):
            with dot.subgraph() as s:
                s.attr(rank="same")
                for index in indices:
                    sub_query = graph.sub_queries[index]
                    s.node(
                        str(index),
                        f"{sub_query.tool}\n(Task {index})",
                        shape="circle",
                        style="filled",
                        fillcolor="#8B0000",
                        fontcolor="white",
                    )

        # Create edges with labels
        for dependent_index, dependencies in graph.dependency_map.items():
            for dependency_index in dependencies:
                dependent_sub_query = graph.sub_queries[dependent_index]
                label = dependent_sub_query.dependency_attr
                dot.edge(str(dependency_index), str(dependent_index), label=label)

        # Save the graph
        dot.render(output_file, format="png", cleanup=True)
        print(f"Graph visualization saved as {output_file}.png")

    def to_cytoscape_json(self, graph) -> str:
        nodes = []
        edges = []

        for index, sub_query in graph.sub_queries.items():
            label = f"{sub_query.tool}\n(Task {index})" if sub_query.tool else f"Expression\n(Task {index})"
            nodes.append({
                "data": {
                    "id": str(index),
                    "label": label,
                    "task": sub_query.task,
                    "tool": sub_query.tool,
                    "actionable": bool(sub_query.tool),
                    "arguments": json.dumps(sub_query.arguments) or "{}",
                }
            })

            if index in graph.dependency_map:
                for dep_index in graph.dependency_map[index]:
                    edges.append({
                        "data": {
                            "source": str(dep_index),
                            "target": str(index),
                            "label": graph.sub_queries[index].dependency_attr,
                        }
                    })

        return json.dumps({"nodes": nodes, "edges": edges})

    def generate_interactive_html(self, graph, output_file: str = "interactive_graph.html"):
        cytoscape_data = self.to_cytoscape_json(graph)

        try:
            template_content = importlib.resources.read_text(__name__, "graph_template.html")
            # template_content = resource_string(__name__, "resources/graph_template.html").decode("utf-8")
        except:
            template_content = importlib.resources.read_text("tool4ai.resources", "graph_template.html")
            # template_content = resource_string("tool4ai", "resources/graph_template.html").decode("utf-8")

        html_content = template_content.replace("{CYTOSCAPE_DATA_PLACEHOLDER}", cytoscape_data)

        with open(output_file, "w") as f:
            f.write(html_content)

        print(f"Interactive graph saved as {output_file}")