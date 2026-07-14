import tkinter as tk
import pandas as pd
from tkinter import filedialog, messagebox

from analysis_pipeline import train_pipeline, predict_pipeline, save_results
from multiprocessing import freeze_support, set_start_method
from pathlib import Path
from tkinter import ttk
import config

class RamanGUI:
    def __init__(self, root):
        self.root = root
        root.title("MoS2 Raman Analyzer")
        root.geometry("500x500")
        self.file_path = None

        errors = startup_check()
        if errors:
            print(errors)
            print(type(errors))
            message = "\n\n".join(errors)
            tk.messagebox.showerror("Startup Error", message)
            raise SystemExit

        self.label = tk.Label(root, text="No LAS selected")
        self.label.pack(pady=20)

        self.select_button = tk.Button(
            root, text="Select LAS File", command=self.select_file
        )
        self.select_button.pack()

        scan_frame = tk.Frame(root)
        scan_frame.pack(pady=10)

        tk.Label(scan_frame, text="Scan Width:").grid(row=0, column=0)
        self.width_entry = tk.Entry(scan_frame, width=6)
        self.width_entry.insert(0, "300")
        self.width_entry.grid(row=0, column=1, padx=5)

        tk.Label(scan_frame,text="Scan Height:").grid(row=0, column=2)
        self.height_entry = tk.Entry(scan_frame,width=6)
        self.height_entry.insert(0, "300")
        self.height_entry.grid(row=0, column=3, padx=5)

        self.run_button = tk.Button(
            root, text="Analyze", command=self.run_analysis
        )
        self.run_button.pack(pady=20)

        self.status = tk.Label(root, text="Waiting...")
        self.status.pack()

        self.progress = ttk.Progressbar(root, length=350, mode="determinate")
        self.progress.pack(pady=10)

        self.table = ttk.Treeview(root)
        self.table.pack(expand=True, fill="both", padx=10, pady=10)
            
    def update_status(self, message):
        self.status.config(text=message)
        self.root.update()

    def update_progress(self, current, total):
        percent = (current / total) * 100
        self.progress["value"] = percent
        self.status.config(text=f"Fitting spectra: {current}/{total}")
        self.root.update()

    def display_results(self, dataframe):
        for row in self.table.get_children():
            self.table.delete(row)

        self.table["columns"] = list(dataframe.columns)
        self.table["show"] = "headings"

        for column in dataframe.columns:
            self.table.heading(column, text=column)
            self.table.column(column, width=100)

        for _, row in dataframe.head(100).iterrows():
            self.table.insert("", "end", values=list(row))

    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[
                ("Text files", "*.txt")
            ]
        )

        if self.file_path:
            self.label.config(text=self.file_path)

    def run_analysis(self):
        if self.file_path is None:
            messagebox.showerror(
                "Error",
                "Select a LAS file first"
            )
            return

        self.status.config(text="Running...")
        self.root.update()

        results = predict_pipeline(
            self.file_path,
            workers=config.PREDICTION_WORKERS,
            status_callback=self.update_status,
            progress_callback=self.update_progress
        )

        save_results(results, print)
        self.display_results(results)
        self.status.config(text="Finished!")
        messagebox.showinfo("Done", "Analysis complete")


def startup_check():
    """
    Verify that all required application files exist.
    """

    errors = []

    # Required folders
    required_dirs = [
        config.MODEL_DIR,
        config.TRAINING_DIR,
        config.RESULTS_DIR
        ]

    for directory in required_dirs:
        if not Path(directory).exists():
            errors.append(
                f"Missing folder:\n{directory}"
            )

    # Required models
    required_models = [
        "lda.pkl",
        "linear.pkl",
        "random_forest.pkl",
        "ridge.pkl"
    ]

    for model in required_models:
        path = Path(config.MODEL_DIR) / model
        if not path.exists():
            errors.append(
                f"Missing model:\n{path}"
            )
    return errors

if __name__ == "__main__":
    freeze_support()

    try:
        set_start_method("spawn")
    except:
        pass

    root = tk.Tk()
    app = RamanGUI(root)
    root.mainloop()
