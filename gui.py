
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import pandas as pd
import numpy as np
import os
import sys

# Import harness analyzer
from harness_analyzer import HarnessAnalyzer

class WireHarnessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wire Harness Analysis Tool")
        self.root.geometry("1200x800")
        
        # Initialize analyzer
        self.analyzer = HarnessAnalyzer()
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the GUI components"""  
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for controls
        left_panel = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Load sample data button
        ttk.Button(left_panel, text="Load Sample Data", 
                   command=self.load_sample_data).pack(fill=tk.X, pady=5)
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(left_panel, text="Analysis", padding="10")
        analysis_frame.pack(fill=tk.X, pady=10, padx=5)
        
        ttk.Button(analysis_frame, text="Calculate Total Length", 
                   command=self.show_total_length).pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_frame, text="Estimate Bundle Diameters", 
                   command=self.show_bundle_diameters).pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_frame, text="Generate Report", 
                   command=self.show_report).pack(fill=tk.X, pady=5)
        
        # Path optimization section
        path_frame = ttk.LabelFrame(left_panel, text="Path Optimization", padding="10")
        path_frame.pack(fill=tk.X, pady=10, padx=5)
        
        ttk.Label(path_frame, text="Source:").pack(anchor=tk.W)
        self.source_var = tk.StringVar()
        self.source_combo = ttk.Combobox(path_frame, textvariable=self.source_var)
        self.source_combo.pack(fill=tk.X, pady=5)
        
        ttk.Label(path_frame, text="Target:").pack(anchor=tk.W)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(path_frame, textvariable=self.target_var)
        self.target_combo.pack(fill=tk.X, pady=5)
        
        ttk.Button(path_frame, text="Find Optimal Path", 
                   command=self.find_optimal_path).pack(fill=tk.X, pady=5)
        
        # Create right panel for visualization
        right_panel = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure for visualization
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create results panel
        self.results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        self.results_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.results_text = tk.Text(self.results_frame, height=10, width=50)
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def load_sample_data(self):
        """Load sample harness data and visualize it"""
        self.analyzer.load_sample_data()
        self.status_var.set("Sample data loaded")
        
        # Update comboboxes
        nodes = list(self.analyzer.harness_graph.nodes())
        self.source_combo['values'] = nodes
        self.target_combo['values'] = nodes
        
        if nodes:
            self.source_combo.current(0)
            self.target_combo.current(len(nodes) - 1)
        
        # Visualize
        self.ax.clear()
        self.analyzer.visualize_harness(ax=self.ax)
        self.canvas.draw()
        
        # Show basic info
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Wire Harness Graph Loaded\n")
        self.results_text.insert(tk.END, f"Number of nodes: {self.analyzer.harness_graph.number_of_nodes()}\n")
        self.results_text.insert(tk.END, f"Number of edges: {self.analyzer.harness_graph.number_of_edges()}\n")
    
    def show_total_length(self):
        """Calculate and display total wire length"""
        if not hasattr(self.analyzer, 'harness_graph') or self.analyzer.harness_graph.number_of_nodes() == 0:
            messagebox.showinfo("Error", "Please load a wire harness first")
            return
        
        total_length = self.analyzer.calculate_total_length()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Total Wire Length: {total_length:.2f} meters\n")
        self.status_var.set(f"Total length calculated: {total_length:.2f} m")
    
    def show_bundle_diameters(self):
        """Calculate and display bundle diameters"""
        if not hasattr(self.analyzer, 'harness_graph') or self.analyzer.harness_graph.number_of_nodes() == 0:
            messagebox.showinfo("Error", "Please load a wire harness first")
            return
        
        bundle_diameters = self.analyzer.estimate_bundle_diameter()
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Bundle Diameters (mm):\n")
        
        for (u, v), diameter in bundle_diameters.items():
            self.results_text.insert(tk.END, f"{u} to {v}: {diameter:.2f} mm\n")
        
        avg_diameter = np.mean(list(bundle_diameters.values()))
        max_diameter = max(bundle_diameters.values())
        
        self.results_text.insert(tk.END, f"\nAverage Diameter: {avg_diameter:.2f} mm\n")
        self.results_text.insert(tk.END, f"Maximum Diameter: {max_diameter:.2f} mm\n")
        
        self.status_var.set(f"Bundle diameters calculated")
    
    def find_optimal_path(self):
        """Find and visualize optimal path"""
        if not hasattr(self.analyzer, 'harness_graph') or self.analyzer.harness_graph.number_of_nodes() == 0:
            messagebox.showinfo("Error", "Please load a wire harness first")
            return
        
        source = self.source_var.get()
        target = self.target_var.get()
        
        if not source or not target:
            messagebox.showinfo("Error", "Please select source and target nodes")
            return
        
        path, length = self.analyzer.find_optimal_path(source, target)
        
        if path is None:
            messagebox.showinfo("No Path", f"No path found between {source} and {target}")
            return
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Optimal path from {source} to {target}:\n")
        self.results_text.insert(tk.END, " -> ".join(path) + "\n")
        self.results_text.insert(tk.END, f"Total path length: {length:.2f} meters\n")
        
        # Calculate installation complexity for this path
        complexity = self.analyzer.estimate_installation_complexity(path)
        self.results_text.insert(tk.END, f"Installation complexity: {complexity:.1f}/10\n")
        
        # Calculate bundle diameter for this path
        diameter = self.analyzer.estimate_bundle_diameter(path)
        self.results_text.insert(tk.END, f"Bundle diameter: {diameter:.2f} mm\n")
        
        # Visualize
        self.ax.clear()
        self.analyzer.visualize_harness(ax=self.ax, highlight_path=path)
        self.canvas.draw()
        
        self.status_var.set(f"Optimal path found: {length:.2f} m")
    
    def show_report(self):
        """Generate and display comprehensive report"""
        if not hasattr(self.analyzer, 'harness_graph') or self.analyzer.harness_graph.number_of_nodes() == 0:
            messagebox.showinfo("Error", "Please load a wire harness first")
            return
        
        report = self.analyzer.generate_report()
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Wire Harness Analysis Report\n")
        self.results_text.insert(tk.END, "=" * 40 + "\n\n")
        
        for key, value in report.items():
            self.results_text.insert(tk.END, f"{key}: {value}\n")
        
        self.status_var.set("Report generated")

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    app = WireHarnessApp(root)
    root.mainloop()