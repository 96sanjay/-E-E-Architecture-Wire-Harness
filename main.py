
import tkinter as tk
from gui import WireHarnessApp

def main():
    """Main entry point for the Wire Harness Analysis Tool"""
    root = tk.Tk()
    app = WireHarnessApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()