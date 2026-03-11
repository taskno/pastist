import tkinter as tk
from tkinter import filedialog, messagebox
import yaml
import os
import shutil

CONFIG_PATH = None

class ProjectManager:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PASTiSt Project Manager")
        self.root.geometry("420x780")
        self.root.resizable(0, 0)
        self.root.attributes('-topmost', True)

        
        # Get directory where this script is located
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.template_config = os.path.join(self.script_dir, "config.yml")
        
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")
        
        tk.Button(main_frame, text="Create New Project", command=self.new_project,
                 width=20, height=2, font=("Helvetica", 11)).pack(pady=15)
        
        tk.Button(main_frame, text="Load Project", command=self.load_project,
                 width=20, height=2, font=("Helvetica", 11)).pack(pady=15)
        
        self.status_label = tk.Label(main_frame, text="Ready", wraplength=380,
                                    font=("Helvetica", 10), justify="left")
        self.status_label.pack(pady=20)
    
    def new_project(self):
        global CONFIG_PATH
        # 1. Choose / create project folder
        project_dir = filedialog.askdirectory(title="Select or create an EMPTY project folder")
        if not project_dir:
            return
        
        # Optional: warn if not empty
        if os.listdir(project_dir):
            if not messagebox.askyesno("Folder not empty", 
                                      "Selected folder is not empty.\nContinue anyway?"):
                return
        
        # 2. Select the file to process
        file_path = filedialog.askopenfilename(
            title="Select point cloud file to process",
            filetypes=[("Point cloud files", " *.ply *.las *.laz"), 
                      ("All files", "*.*")]
        )
        if not file_path:
            return
        
        abs_file_path = os.path.abspath(file_path)
        
        # 3. Prepare config.yml
        CONFIG_PATH = os.path.join(project_dir, "config.yml")
        
        # Try to copy template if it exists
        config_data = {}
        if os.path.isfile(self.template_config):
            try:
                with open(self.template_config, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
            except Exception as e:
                messagebox.showwarning("Template warning", 
                                      f"Could not read template config.yml\n{e}\nUsing default structure.")
        
        # Set / overwrite the file path in the expected location
        if 'preProcessing' not in config_data:
            config_data['preProcessing'] = {}
        
        config_data['preProcessing']['point_cloud'] = abs_file_path
        
        # 4. Write config.yml
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f, sort_keys=False, allow_unicode=True)
            
            self.status_label.config(
                text=f"Project created successfully!\n\nLocation:\n{project_dir}\n\nFile:\n{abs_file_path}"
            )
            messagebox.showinfo("Success", 
                               "Project folder prepared.\nconfig.yml created with selected file path.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create config.yml\n{str(e)}")
    
    def load_project(self):
        global CONFIG_PATH
        project_dir = filedialog.askdirectory(title="Select existing project folder")
        if not project_dir:
            return
        
        CONFIG_PATH = os.path.join(project_dir, "config.yml")
        if not os.path.isfile(CONFIG_PATH):
            messagebox.showerror("Not found", "config.yml not found in selected folder.")
            return
        
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            pcd_path = config['preProcessing']['point_cloud']
            
            if pcd_path and os.path.isfile(pcd_path):
                status_text = f"Project loaded successfully!\n\nFolder:\n{project_dir}\n\nFile:\n{pcd_path}"
                msg = f"Project loaded.\nPoint cloud file:\n{pcd_path}"
            else:
                status_text = f"Config loaded, but file not found:\n{pcd_path}"
                msg = "Config loaded, but referenced file does not exist."
            
            self.status_label.config(text=status_text)
            messagebox.showinfo("Loaded", msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read config.yml\n{str(e)}")

    def run(self):
        """Start the dialog and return config_path after user finishes"""
        self.root.mainloop()
        return CONFIG_PATH


if __name__ == "__main__":
    #CONFIG_PATH = None
    manager = ProjectManager()
    config_file = manager.run()
    
    if CONFIG_PATH:
        print("User selected project:")
        print(f"  CONFIG_PATH : {CONFIG_PATH}")
        
    else:
        print("No project was selected / window was closed.")