
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HarnessAnalyzer:
    def __init__(self):
        self.harness_graph = nx.Graph()
        self.node_positions = {}
        
    def load_sample_data(self):
        """Load sample wire harness data"""
        # Create nodes (connectors/components)
        nodes = [
            ("ECU", {"type": "controller", "position": (0, 0)}),
            ("Sensor1", {"type": "sensor", "position": (2, 1)}),
            ("Sensor2", {"type": "sensor", "position": (3, -1)}),
            ("Actuator1", {"type": "actuator", "position": (4, 2)}),
            ("Actuator2", {"type": "actuator", "position": (5, 0)}),
            ("Junction1", {"type": "junction", "position": (1, 0)}),
            ("Junction2", {"type": "junction", "position": (3, 0)})
        ]
        
        # Create edges (wires with properties)
        edges = [
            ("ECU", "Junction1", {"wire_type": "power", "gauge": 18, "length": 0.5, "signals": ["CAN_H", "CAN_L"]}),
            ("Junction1", "Sensor1", {"wire_type": "signal", "gauge": 22, "length": 1.2, "signals": ["Analog"]}),
            ("Junction1", "Junction2", {"wire_type": "power", "gauge": 20, "length": 2.0, "signals": ["CAN_H", "CAN_L"]}),
            ("Junction2", "Sensor2", {"wire_type": "signal", "gauge": 22, "length": 1.0, "signals": ["Digital"]}),
            ("Junction2", "Actuator1", {"wire_type": "power", "gauge": 16, "length": 1.5, "signals": ["PWM"]}),
            ("Junction2", "Actuator2", {"wire_type": "power", "gauge": 16, "length": 2.1, "signals": ["PWM"]})
        ]
        
        # Create graph
        self.harness_graph.add_nodes_from(nodes)
        self.harness_graph.add_edges_from(edges)
        
        # Store node positions for visualization
        self.node_positions = {node: data["position"] for node, data in self.harness_graph.nodes(data=True)}
        
        return self.harness_graph
    
    def calculate_total_length(self):
        """Calculate the total wire length in the harness"""
        total_length = sum(data["length"] for _, _, data in self.harness_graph.edges(data=True))
        return total_length
    
    def estimate_bundle_diameter(self, path=None):
        """Estimate wire bundle diameter based on wire gauges
        
        Args:
            path (list): Optional path of nodes to calculate diameter for a specific segment
                         If None, calculates for each segment
        """
        # Wire gauge to diameter mapping (AWG to mm)
        gauge_to_diameter = {
            16: 1.29,
            18: 1.02,
            20: 0.81,
            22: 0.64,
            24: 0.51
        }
        
        if path:
            # Calculate for specific path
            edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            diameters = []
            
            for u, v in edges:
                if self.harness_graph.has_edge(u, v):
                    gauge = self.harness_graph[u][v]["gauge"]
                    diameters.append(gauge_to_diameter.get(gauge, 0.8))  # Default if gauge not found
            
            # Simple estimation: sum of wire cross-sectional areas
            total_area = sum([(d/2)**2 * np.pi for d in diameters])
            bundle_diameter = 2 * np.sqrt(total_area / np.pi)
            
            return bundle_diameter
        else:
            # Calculate for each segment
            segment_diameters = {}
            
            for u, v, data in self.harness_graph.edges(data=True):
                gauge = data.get("gauge", 20)  # Default to 20 AWG if not specified
                wire_diameter = gauge_to_diameter.get(gauge, 0.8)  # Default if gauge not found
                num_signals = len(data.get("signals", [1]))  # Count number of signals
                
                # Simple bundle estimation: assuming circular packing of wires
                bundle_area = num_signals * (wire_diameter/2)**2 * np.pi
                bundle_diameter = 2 * np.sqrt(bundle_area / np.pi)
                
                segment_diameters[(u, v)] = bundle_diameter
            
            return segment_diameters
    
    def find_optimal_path(self, source, target, weight='length'):
        """Find optimal path between two nodes using Dijkstra's algorithm
        
        Args:
            source (str): Source node name
            target (str): Target node name
            weight (str): Edge attribute to use as weight (default: length)
        """
        try:
            path = nx.dijkstra_path(self.harness_graph, source, target, weight=weight)
            length = nx.dijkstra_path_length(self.harness_graph, source, target, weight=weight)
            return path, length
        except nx.NetworkXNoPath:
            return None, float('inf')
    
    def estimate_installation_complexity(self, path=None):
        """Estimate installation complexity based on bundle diameter and path length
        
        Returns a score from 1-10 where higher means more complex
        """
        if not path:
            # For entire harness
            total_length = self.calculate_total_length()
            avg_diameter = np.mean(list(self.estimate_bundle_diameter().values()))
            num_junctions = sum(1 for node, data in self.harness_graph.nodes(data=True) 
                               if data.get("type") == "junction")
            
            # Simple complexity formula
            complexity = (total_length * avg_diameter * 0.5) + (num_junctions * 1.5)
            
            # Scale to 1-10
            return min(10, max(1, complexity))
        else:
            # For specific path
            path_length = sum(self.harness_graph[path[i]][path[i+1]]["length"] 
                             for i in range(len(path)-1))
            bundle_diameter = self.estimate_bundle_diameter(path)
            
            # Simple complexity formula for path
            complexity = path_length * bundle_diameter * 0.8
            
            # Scale to 1-10
            return min(10, max(1, complexity))
    
    def visualize_harness(self, ax=None, highlight_path=None):
        """Visualize the wire harness
        
        Args:
            ax (matplotlib.axes): Optional matplotlib axes to plot on
            highlight_path (list): Optional list of nodes to highlight
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create node colors based on type
        node_colors = []
        for node in self.harness_graph.nodes():
            node_type = self.harness_graph.nodes[node].get("type", "")
            if node_type == "controller":
                node_colors.append('red')
            elif node_type == "sensor":
                node_colors.append('green')
            elif node_type == "actuator":
                node_colors.append('blue')
            elif node_type == "junction":
                node_colors.append('orange')
            else:
                node_colors.append('gray')
        
        # Draw the network
        nx.draw_networkx_nodes(self.harness_graph, self.node_positions, 
                              node_color=node_colors, node_size=500, ax=ax)
        
        # Draw edges with colors based on wire type
        edge_colors = []
        for u, v, data in self.harness_graph.edges(data=True):
            wire_type = data.get("wire_type", "")
            if wire_type == "power":
                edge_colors.append('red')
            elif wire_type == "signal":
                edge_colors.append('blue')
            else:
                edge_colors.append('black')
        
        nx.draw_networkx_edges(self.harness_graph, self.node_positions, 
                              edge_color=edge_colors, width=2, ax=ax)
        
        # Draw edge labels with gauge
        edge_labels = {(u, v): f"{data.get('gauge', '')}AWG" 
                      for u, v, data in self.harness_graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.harness_graph, self.node_positions, 
                                    edge_labels=edge_labels, ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(self.harness_graph, self.node_positions, font_weight='bold', ax=ax)
        
        # Highlight path if provided
        if highlight_path and len(highlight_path) > 1:
            path_edges = [(highlight_path[i], highlight_path[i+1]) for i in range(len(highlight_path)-1)]
            nx.draw_networkx_edges(self.harness_graph, self.node_positions, 
                                  edgelist=path_edges, edge_color='yellow', 
                                  width=4, ax=ax)
        
        ax.set_title("Wire Harness Layout")
        ax.set_axis_off()
        
        return ax
    
    def generate_report(self):
        """Generate a summary report of the harness analysis"""
        total_length = self.calculate_total_length()
        bundle_diameters = self.estimate_bundle_diameter()
        avg_diameter = np.mean(list(bundle_diameters.values()))
        max_diameter = max(bundle_diameters.values())
        installation_complexity = self.estimate_installation_complexity()
        
        # Calculate network properties
        num_components = self.harness_graph.number_of_nodes()
        num_connections = self.harness_graph.number_of_edges()
        
        report = {
            "Total Components": num_components,
            "Total Connections": num_connections,
            "Total Wire Length (m)": round(total_length, 2),
            "Average Bundle Diameter (mm)": round(avg_diameter, 2),
            "Maximum Bundle Diameter (mm)": round(max_diameter, 2),
            "Installation Complexity (1-10)": round(installation_complexity, 1)
        }
        
        return report

# Example usage
if __name__ == "__main__":
    analyzer = HarnessAnalyzer()
    analyzer.load_sample_data()
    
    # Test some functions
    print(f"Total harness length: {analyzer.calculate_total_length()} meters")
    print(f"Installation complexity: {analyzer.estimate_installation_complexity()}")
    
    # Visualize
    plt.figure(figsize=(10, 8))
    analyzer.visualize_harness()
    plt.show()