import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import yaml
from pathlib import Path
import sys

class ConfigEditorApp:
    def __init__(self, root, config_path):
        self.root = root
        self.root.title("Config Editor")
        self.root.geometry("1000x700")
        self.root.resizable(0, 0)
        self.root.attributes('-topmost', True)
        
        self.yaml_data = {}
        self.file_path = None

        self._build_ui()
        #self._create_menu()
        self.load_config(config_path)

    def _build_ui(self):
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        #ttk.Button(top_frame, text="Open YAML", command=self.open_file).pack(side=tk.LEFT, padx=5)
        #self.file_label = ttk.Label(top_frame, text="No file loaded")
        #self.file_label.pack(side=tk.LEFT, padx=15)

        ttk.Button(top_frame, text="Save", command=self.save_file).pack(side=tk.RIGHT, padx=5)

        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.main_frame = ttk.Frame(self.canvas)

        self.main_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.status = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        #self.status.pack(side=tk.BOTTOM, fill=tk.X)
        #self.status.pack(side=tk.BOTTOM, fill=tk.X, expand=False, ipady=2)

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.open_file)
        filemenu.add_command(label="Save", command=self.save_file)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml *.yml")])
        if not path: return
        try:
            with open(path, encoding="utf-8") as f:
                self.yaml_data = yaml.safe_load(f) or {}
            self.file_path = Path(path)
            self.file_label.config(text=f"Editing: {self.file_path.name}")
            self._refresh_form()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_config(self, path):
        #path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml *.yml")])
        if not path: return
        try:
            with open(path, encoding="utf-8") as f:
                self.yaml_data = yaml.safe_load(f) or {}
            self.file_path = Path(path)
            #self.file_label.config(text=f"Editing: {self.file_path.name}")
            self._refresh_form()

        except Exception as e:
            messagebox.showerror("Error", str(e))


    def _refresh_form(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Layout Mapping: 
        # Column 0: preProcessing, Segmentation, splitSegments
        # Column 1: beamModeling, beamExporter
        col_map = {
            "preProcessing": 0,
            "Segmentation": 0,
            "splitSegments": 0,
            "beamModeling": 1,
            "beamExporter": 1
        }

        # Track row counters for each column
        self.col_row_counters = {0: 0, 1: 0}

        for key, value in self.yaml_data.items():
            target_col = col_map.get(key, 0)
            self._draw_section(key, value, target_col, (key,))
            # Add this inside _refresh_form() after drawing sections

    def _draw_section(self, key, value, col, path):
        start_row = self.col_row_counters[col]
        
        # Section Header
        lbl = ttk.Label(self.main_frame, text=f"[{key.upper()}]", 
                        font=("Segoe UI", 10, "bold"), foreground="#005fb8")
        lbl.grid(row=start_row, column=col, sticky="w", padx=20, pady=(20, 5))
        
        self.col_row_counters[col] += 1
        
        if isinstance(value, dict):
            self._draw_nested_elements(value, col, path)

    def _draw_nested_elements(self, data, col, path):
        for key, value in data.items():
            current_path = path + (key,)
            current_row = self.col_row_counters[col]

            # Use a frame per row for cleaner alignment within the column
            row_frame = ttk.Frame(self.main_frame)
            row_frame.grid(row=current_row, column=col, sticky="ew", padx=30, pady=2)

            ttk.Label(row_frame, text=f"{key}:", font=("Consolas", 10), width=25, anchor="e").pack(side=tk.LEFT)

            if isinstance(value, list):
                val_str = ", ".join(map(str, value))
            else:
                val_str = str(value)

            var = tk.StringVar(value=val_str)
            
            if isinstance(value, bool):
                var_bool = tk.BooleanVar(value=value)
                ent = ttk.Checkbutton(row_frame, variable=var_bool, 
                                     command=lambda p=current_path, v=var_bool: self._update_nested_value(p, v.get()))
            else:
                ent = ttk.Entry(row_frame, textvariable=var, font=("Consolas", 10), width=35)
                ent.bind("<KeyRelease>", lambda e, p=current_path, v=var: self._update_nested_value(p, v.get()))
            
            ent.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            self.col_row_counters[col] += 1

    def _update_nested_value(self, path, new_value_str):
        target = self.yaml_data
        for node in path[:-1]:
            target = target[node]
        
        key = path[-1]
        original_val = target[key]

        try:
            if isinstance(original_val, bool):
                target[key] = bool(new_value_str)
            elif isinstance(original_val, int):
                target[key] = int(new_value_str) if new_value_str != "" else 0
            elif isinstance(original_val, float):
                target[key] = float(new_value_str) if new_value_str != "" else 0.0
            elif isinstance(original_val, list):
                target[key] = [i.strip() for i in new_value_str.split(",") if i.strip()]
            else:
                target[key] = new_value_str
            #self.status.config(text=f"Updated {' > '.join(path)}")
        except ValueError:
            pass 

    def save_file(self):
        if not self.file_path: return
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.yaml_data, f, sort_keys=False, allow_unicode=True)
            messagebox.showinfo("Success", "Configuration saved!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":

    config_path = sys.argv[1]
    root = tk.Tk()
    app = ConfigEditorApp(root, config_path)
    root.mainloop()