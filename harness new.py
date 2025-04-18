import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HarnessAnalyzer:
    def __init__(self):
        self.harness_graph = nx.Graph()
        self.node_positions = {}
        
    def load_sample_data(self, include_sub_junctions=True):
        """Load sample wire harness data with optional sub-junction structure
        
        Args:
            include_sub_junctions (bool): Whether to include sub-junction structures
        """
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
        
        # Add sub-junctions if requested
        if include_sub_junctions:
            # Add sub-junction nodes
            sub_nodes = [
                ("SubJunction1", {"type": "sub_junction", "position": (1.5, 0.5), "parent": "Junction1"}),
                ("SubJunction2", {"type": "sub_junction", "position": (3.5, -0.5), "parent": "Junction2"}),
                ("Sensor3", {"type": "sensor", "position": (2, 0.5)}),
                ("Sensor4", {"type": "sensor", "position": (4, -1)})
            ]
            nodes.extend(sub_nodes)
            
            # Add sub-junction edges
            sub_edges = [
                ("Junction1", "SubJunction1", {"wire_type": "signal", "gauge": 20, "length": 0.8, "signals": ["I2C_SDA", "I2C_SCL"]}),
                ("SubJunction1", "Sensor3", {"wire_type": "signal", "gauge": 24, "length": 0.6, "signals": ["I2C_SDA", "I2C_SCL"]}),
                ("Junction2", "SubJunction2", {"wire_type": "signal", "gauge": 20, "length": 0.7, "signals": ["Serial_TX", "Serial_RX"]}),
                ("SubJunction2", "Sensor4", {"wire_type": "signal", "gauge": 24, "length": 0.5, "signals": ["Serial_TX", "Serial_RX"]})
            ]
            edges.extend(sub_edges)
        
        # Create graph
        self.harness_graph.add_nodes_from(nodes)
        self.harness_graph.add_edges_from(edges)
        
        # Store node positions for visualization
        self.node_positions = {node: data["position"] for node, data in self.harness_graph.nodes(data=True)}
        
        # Add junction hierarchy information to graph
        self._compute_junction_hierarchy()
        
        return self.harness_graph
    
    def _compute_junction_hierarchy(self):
        """Compute the junction hierarchy to track which junctions feed into others"""
        junction_nodes = [node for node, data in self.harness_graph.nodes(data=True) 
                         if data.get("type") in ["junction", "sub_junction"]]
        
        # For each junction, identify its downstream junctions
        for junction in junction_nodes:
            # Get all neighbors
            neighbors = list(self.harness_graph.neighbors(junction))
            
            # Find sub-junctions among neighbors
            sub_junctions = [n for n in neighbors if 
                            self.harness_graph.nodes[n].get("type") in ["junction", "sub_junction"]]
            
            # Store this information in the junction node attributes
            self.harness_graph.nodes[junction]["downstream_junctions"] = sub_junctions
    
    def calculate_total_length(self):
        """Calculate the total wire length in the harness"""
        total_length = sum(data["length"] for _, _, data in self.harness_graph.edges(data=True))
        return total_length
    
    def get_downstream_edges(self, node):
        """Get all edges in branches downstream from the given node
        
        Args:
            node (str): Starting node name
            
        Returns:
            list: List of edge tuples (u, v) that are downstream from the node
        """
        visited = set()
        downstream_edges = []
        
        def dfs(current_node, parent=None):
            visited.add(current_node)
            
            for neighbor in self.harness_graph.neighbors(current_node):
                if neighbor != parent:
                    if neighbor not in visited:
                        downstream_edges.append((current_node, neighbor))
                        dfs(neighbor, current_node)
        
        dfs(node)
        return downstream_edges
    
    def estimate_bundle_diameter(self, path=None, consider_hierarchy=True):
        """Estimate wire bundle diameter based on wire gauges and branch hierarchy
        
        Args:
            path (list): Optional path of nodes to calculate diameter for a specific segment
                         If None, calculates for each segment
            consider_hierarchy (bool): Whether to consider junction hierarchy when calculating
                                      bundle diameters
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
        
        elif consider_hierarchy:
            # Calculate bundle diameters considering hierarchy
            segment_diameters = {}
            space_utilization = {}
            
            # Calculate base diameters for each segment
            for u, v, data in self.harness_graph.edges(data=True):
                gauge = data.get("gauge", 20)  # Default to 20 AWG if not specified
                wire_diameter = gauge_to_diameter.get(gauge, 0.8)  # Default if gauge not found
                num_signals = len(data.get("signals", [1]))  # Count number of signals
                
                # Simple diameter for this segment alone
                segment_area = num_signals * (wire_diameter/2)**2 * np.pi
                segment_diameters[(u, v)] = 2 * np.sqrt(segment_area / np.pi)
            
            # Calculate bundle diameters at junctions considering downstream branches
            junction_nodes = [node for node, data in self.harness_graph.nodes(data=True) 
                             if data.get("type") in ["junction", "sub_junction"]]
            
            # Process junctions from leaf to root
            for junction in sorted(junction_nodes, 
                                   key=lambda j: len(self.get_downstream_edges(j)), 
                                   reverse=False):
                # Get all downstream edges
                downstream_edges = self.get_downstream_edges(junction)
                
                # Calculate total area of downstream wires
                total_downstream_area = 0
                for u, v in downstream_edges:
                    if (u, v) in segment_diameters:
                        diameter = segment_diameters[(u, v)]
                        area = (diameter/2)**2 * np.pi
                        total_downstream_area += area
                
                # Get edges connecting to this junction
                incoming_edges = []
                for u, v in self.harness_graph.edges():
                    if u == junction or v == junction:
                        # Make sure it's properly oriented (other_node, junction)
                        other_node = v if u == junction else u
                        edge = (other_node, junction) if u == junction else (junction, other_node)
                        incoming_edges.append(edge)
                
                # Update diameter for all incoming edges to account for downstream wires
                for edge in incoming_edges:
                    u, v = edge
                    
                    # Skip if this is a downstream edge
                    is_downstream = False
                    for du, dv in downstream_edges:
                        if (u == du and v == dv) or (u == dv and v == du):
                            is_downstream = True
                            break
                    
                    if is_downstream:
                        continue
                    
                    # Calculate base area for this segment
                    base_diameter = segment_diameters.get((u, v), segment_diameters.get((v, u), 0))
                    base_area = (base_diameter/2)**2 * np.pi
                    
                    # Combine with downstream area
                    total_area = base_area + total_downstream_area
                    if total_area > 0:  # Ensure we don't have zero area
                        new_diameter = 2 * np.sqrt(total_area / np.pi)
                        
                        # Update diameter
                        segment_diameters[(u, v)] = new_diameter
                        segment_diameters[(v, u)] = new_diameter  # Update both directions
                        
                        # Calculate space utilization
                        space_utilization[(u, v)] = (base_area / total_area) * 100
                        space_utilization[(v, u)] = space_utilization[(u, v)]
            
            # Ensure we have at least one value in space_utilization
            if not space_utilization:
                # Add default utilization values if none were calculated
                for u, v in segment_diameters.keys():
                    space_utilization[(u, v)] = 100.0  # If there's no bundling, utilization is 100%
            
            return segment_diameters, space_utilization
            
        else:
            # Original calculation for each segment independently
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
            diameters, _ = self.estimate_bundle_diameter(consider_hierarchy=True)
            avg_diameter = np.mean(list(diameters.values())) if diameters else 0
            num_junctions = sum(1 for node, data in self.harness_graph.nodes(data=True) 
                               if data.get("type") in ["junction", "sub_junction"])
            
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
    
    def visualize_harness(self, ax=None, highlight_path=None, show_diameters=False, show_utilization=False):
        """Visualize the wire harness
        
        Args:
            ax (matplotlib.axes): Optional matplotlib axes to plot on
            highlight_path (list): Optional list of nodes to highlight
            show_diameters (bool): Whether to display bundle diameters on edges
            show_utilization (bool): Whether to display space utilization percentages
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate bundle diameters and space utilization if needed
        if show_diameters or show_utilization:
            diameters, utilization = self.estimate_bundle_diameter(consider_hierarchy=True)
        
        # Create node colors based on type
        node_colors = []
        node_sizes = []
        for node in self.harness_graph.nodes():
            node_type = self.harness_graph.nodes[node].get("type", "")
            if node_type == "controller":
                node_colors.append('red')
                node_sizes.append(600)
            elif node_type == "sensor":
                node_colors.append('green')
                node_sizes.append(500)
            elif node_type == "actuator":
                node_colors.append('blue')
                node_sizes.append(500)
            elif node_type == "junction":
                node_colors.append('orange')
                node_sizes.append(450)
            elif node_type == "sub_junction":
                node_colors.append('yellow')
                node_sizes.append(350)
            else:
                node_colors.append('gray')
                node_sizes.append(400)
        
        # Draw the network
        nx.draw_networkx_nodes(self.harness_graph, self.node_positions, 
                              node_color=node_colors, node_size=node_sizes, ax=ax)
        
        # Draw edges with colors based on wire type
        edge_colors = []
        edge_widths = []
        for u, v, data in self.harness_graph.edges(data=True):
            wire_type = data.get("wire_type", "")
            if wire_type == "power":
                edge_colors.append('red')
            elif wire_type == "signal":
                edge_colors.append('blue')
            else:
                edge_colors.append('black')
                
            # Adjust edge width based on gauge
            gauge = data.get("gauge", 20)
            edge_widths.append(1 + (24 - gauge) / 3)  # Thicker lines for lower gauges
        
        nx.draw_networkx_edges(self.harness_graph, self.node_positions, 
                              edge_color=edge_colors, width=edge_widths, ax=ax)
        
        # Draw edge labels
        edge_labels = {}
        for u, v, data in self.harness_graph.edges(data=True):
            label = f"{data.get('gauge', '')}AWG"
            
            if show_diameters and (u, v) in diameters:
                diameter = round(diameters.get((u, v), 0), 2)
                label += f"\n{diameter}mm"
                
            if show_utilization and (u, v) in utilization:
                util = round(utilization.get((u, v), 0), 1)
                label += f"\n{util}%"
                
            edge_labels[(u, v)] = label
        
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
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Controller'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Sensor'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Actuator'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Junction'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Sub-Junction')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title("Wire Harness Layout")
        ax.set_axis_off()
        
        return ax
    
    def generate_report(self, include_utilization=True):
        """Generate a summary report of the harness analysis
        
        Args:
            include_utilization (bool): Whether to include space utilization stats
        """
        total_length = self.calculate_total_length()
        
        if include_utilization:
            diameters, utilization = self.estimate_bundle_diameter(consider_hierarchy=True)
            
            # Check if we have valid diameter values
            if diameters:
                avg_diameter = np.mean(list(diameters.values()))
                max_diameter = max(diameters.values())
            else:
                avg_diameter = 0
                max_diameter = 0
                
            # Check if we have valid utilization values
            if utilization:
                avg_utilization = np.mean(list(utilization.values()))
                min_utilization = min(utilization.values())
            else:
                avg_utilization = 0
                min_utilization = 0
        else:
            diameters = self.estimate_bundle_diameter()
            if diameters:
                avg_diameter = np.mean(list(diameters.values()))
                max_diameter = max(diameters.values())
            else:
                avg_diameter = 0
                max_diameter = 0
        
        installation_complexity = self.estimate_installation_complexity()
        
        # Calculate network properties
        num_components = self.harness_graph.number_of_nodes()
        num_connections = self.harness_graph.number_of_edges()
        num_junctions = sum(1 for node, data in self.harness_graph.nodes(data=True) 
                           if data.get("type") in ["junction", "sub_junction"])
        
        report = {
            "Total Components": num_components,
            "Total Connections": num_connections,
            "Number of Junctions": num_junctions,
            "Total Wire Length (m)": round(total_length, 2),
            "Average Bundle Diameter (mm)": round(avg_diameter, 2),
            "Maximum Bundle Diameter (mm)": round(max_diameter, 2),
            "Installation Complexity (1-10)": round(installation_complexity, 1)
        }
        
        if include_utilization:
            report.update({
                "Average Space Utilization (%)": round(avg_utilization, 1),
                "Minimum Space Utilization (%)": round(min_utilization, 1),
            })
        
        return report

# Example usage
if __name__ == "__main__":
    analyzer = HarnessAnalyzer()
    analyzer.load_sample_data(include_sub_junctions=True)
    
    # Test some functions
    print(f"Total harness length: {analyzer.calculate_total_length()} meters")
    print(f"Installation complexity: {analyzer.estimate_installation_complexity()}")
    
    # Generate and print report
    try:
        report = analyzer.generate_report(include_utilization=True)
        print("\nHarness Analysis Report:")
        for key, value in report.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error generating report: {e}")
        
    try:
        # Visualize with diameters and utilization
        plt.figure(figsize=(12, 10))
        analyzer.visualize_harness(show_diameters=True, show_utilization=True)
        plt.tight_layout()
        plt.savefig("harness_diagram.png")
        print("\nHarness diagram saved as 'harness_diagram.png'")
        plt.show()
    except Exception as e:
        print(f"Error visualizing harness: {e}")