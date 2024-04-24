import tkinter as tk
from LyingDetectionInterface import LyingDetectionInterface

if __name__ == '__main__':
    root = tk.Tk()
    lying_interface = LyingDetectionInterface(root)
    root.mainloop()