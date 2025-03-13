import os
import sys
import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec

# Import data loading module
sys.path.append('.')
from data_loader import FacialExpressionDataset, TAGS_MAPPING, decode_label

class DatasetViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Expression Dataset Visualization Tool")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)
        
        # Dataset attributes
        self.dataset = None
        self.dataset_root = tk.StringVar(value="./datasets")
        self.current_index = 0
        self.filter_tag = tk.StringVar(value="All")
        self.threshold = tk.DoubleVar(value=0.3)
        self.split = tk.StringVar(value="train")
        
        # Filtered index list
        self.filtered_indices = []
        
        # Build UI interface
        self._build_ui()
        
        # Loading status
        self.is_loaded = False
        
    def _build_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top control area
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # First row controls
        row1 = ttk.Frame(control_frame)
        row1.pack(fill=tk.X, pady=5)
        
        ttk.Label(row1, text="Dataset Path:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(row1, textvariable=self.dataset_root, width=50).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row1, text="Browse...", command=self._browse_dataset).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row1, text="Load Dataset", command=self._load_dataset).pack(side=tk.LEFT, padx=(0, 5))
        
        # Second row controls
        row2 = ttk.Frame(control_frame)
        row2.pack(fill=tk.X, pady=5)
        
        ttk.Label(row2, text="Dataset Split:").pack(side=tk.LEFT, padx=(0, 5))
        split_combo = ttk.Combobox(row2, textvariable=self.split, values=["train", "val", "test"], width=10, state="readonly")
        split_combo.pack(side=tk.LEFT, padx=(0, 5))
        split_combo.bind("<<ComboboxSelected>>", self._reload_dataset)
        
        ttk.Label(row2, text="Label Threshold:").pack(side=tk.LEFT, padx=(0, 5))
        threshold_scale = ttk.Scale(row2, from_=0.0, to=1.0, variable=self.threshold, orient=tk.HORIZONTAL, length=100)
        threshold_scale.pack(side=tk.LEFT, padx=(0, 5))
        threshold_scale.bind("<ButtonRelease-1>", self._update_view)
        
        ttk.Label(row2, text="Filter Expression:").pack(side=tk.LEFT, padx=(10, 5))
        tags_values = ["All"] + list(TAGS_MAPPING.values())
        filter_combo = ttk.Combobox(row2, textvariable=self.filter_tag, values=tags_values, width=25, state="readonly")
        filter_combo.pack(side=tk.LEFT, padx=(0, 5))
        filter_combo.bind("<<ComboboxSelected>>", self._apply_filter)
        
        ttk.Button(row2, text="Previous", command=self._prev_image).pack(side=tk.RIGHT, padx=5)
        ttk.Button(row2, text="Next", command=self._next_image).pack(side=tk.RIGHT, padx=5)
        self.index_label = ttk.Label(row2, text="0/0")
        self.index_label.pack(side=tk.RIGHT, padx=10)
        
        # Image and label display area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image display area
        self.image_frame = ttk.LabelFrame(content_frame, text="Image Preview")
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label information area
        info_frame = ttk.LabelFrame(content_frame, text="Label Information", width=400)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Use Matplotlib to display bar chart on the right
        self.fig = plt.Figure(figsize=(5, 8), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=info_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready. Please load a dataset...")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(fill=tk.X)
        
        # Bind keyboard shortcuts
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Right>", lambda e: self._next_image())
        
    def _browse_dataset(self):
        """Browse to select dataset directory"""
        directory = filedialog.askdirectory(initialdir=self.dataset_root.get())
        if directory:
            self.dataset_root.set(directory)
    
    def _load_dataset(self):
        """Load dataset"""
        try:
            self.status_var.set(f"Loading {self.split.get()} dataset...")
            self.root.update_idletasks()
            
            # Create dataset instance
            self.dataset = FacialExpressionDataset(
                dataset_root=self.dataset_root.get(),
                split=self.split.get(),
                transform=None
            )
            
            # Set initial index
            self.current_index = 0
            self._apply_filter()
            
            if len(self.filtered_indices) > 0:
                self.is_loaded = True
                self._update_view()
                self.status_var.set(f"Loaded {self.split.get()} dataset, {len(self.filtered_indices)} samples (filtered) / {len(self.dataset)} samples (total)")
            else:
                self.status_var.set(f"No samples matching the filter criteria")
        except Exception as e:
            self.is_loaded = False
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.status_var.set(f"Failed to load dataset: {str(e)}")
    
    def _reload_dataset(self, event=None):
        """Reload dataset (when switching splits)"""
        if hasattr(self, 'dataset') and self.dataset is not None:
            self._load_dataset()
    
    def _apply_filter(self, event=None):
        """Apply label filtering"""
        if not hasattr(self, 'dataset') or self.dataset is None:
            return
        
        filter_tag = self.filter_tag.get()
        threshold = self.threshold.get()
        
        # Reset filtered indices
        self.filtered_indices = []
        
        # Iterate through all samples
        for idx in range(len(self.dataset)):
            _, label, _ = self.dataset[idx]
            
            if filter_tag == "All":
                # Don't filter specific labels, but at least one label should exceed threshold
                if torch.any(label > threshold):
                    self.filtered_indices.append(idx)
            else:
                # Filter specific label
                tag_idx = -1
                for k, v in TAGS_MAPPING.items():
                    if v == filter_tag:
                        tag_idx = k - 1  # Label index starts from 0
                        break
                
                if tag_idx >= 0 and tag_idx < len(label) and label[tag_idx] > threshold:
                    self.filtered_indices.append(idx)
        
        # Reset current index
        self.current_index = 0 if self.filtered_indices else -1
        
        # Update view
        self._update_view()
        self.status_var.set(f"Filtering complete, found {len(self.filtered_indices)} samples")
    
    def _update_view(self, event=None):
        """Update current view display"""
        if not self.is_loaded or not self.filtered_indices:
            # Clear display
            self.image_label.configure(image='')
            self.fig.clear()
            self.canvas.draw()
            self.index_label.configure(text="0/0")
            return
        
        # Get current sample
        dataset_idx = self.filtered_indices[self.current_index]
        image, label, img_path = self.dataset[dataset_idx]
        
        # Update image display
        if isinstance(image, torch.Tensor):
            # If tensor, conversion needed
            if image.shape[0] == 3:  # Check if RGB format
                # Destandardize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_display = image * std + mean
                
                # Convert to PIL image
                img_display = img_display.permute(1, 2, 0).numpy()
                img_display = np.clip(img_display * 255, 0, 255).astype(np.uint8)
            else:
                img_display = image.numpy().astype(np.uint8)
        else:
            # If already numpy array
            img_display = image
        
        # Convert to PIL image and resize to fit display area
        pil_img = Image.fromarray(img_display)
        
        # Calculate resize, maintaining aspect ratio
        img_width, img_height = pil_img.size
        frame_width = self.image_frame.winfo_width() - 30  # Subtract padding
        frame_height = self.image_frame.winfo_height() - 30
        
        if frame_width > 100 and frame_height > 100:  # Ensure window has reasonable size
            scale = min(frame_width / img_width, frame_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to Tkinter usable image
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img  # Keep reference to prevent garbage collection
        
        # Update label information display
        self._update_label_display(label, os.path.basename(img_path))
        
        # Update index display
        self.index_label.configure(text=f"{self.current_index + 1}/{len(self.filtered_indices)}")
    
    def _update_label_display(self, label, img_name):
        """Update label information display"""
        threshold = self.threshold.get()
        
        # Decode label
        active_labels = decode_label(label, threshold)
        
        # Clear graph
        self.fig.clear()
        
        # Create subplot layout
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3]) 
        
        # Text information display
        ax_info = self.fig.add_subplot(gs[0])
        ax_info.axis('off')
        
        # Image name and active label count
        info_text = f"Image: {img_name}\nLabel Count: {len(active_labels)}"
        ax_info.text(0.05, 0.9, info_text, transform=ax_info.transAxes, fontsize=9, 
                   verticalalignment='top', wrap=True)
        
        # Bar chart display
        ax_bar = self.fig.add_subplot(gs[1])
        
        if active_labels:
            # Get label names and confidence
            sorted_labels = sorted(active_labels, key=lambda x: x[1], reverse=True)
            names = [item[0] for item in sorted_labels]
            values = [item[1] for item in sorted_labels]
            
            # Draw horizontal bar chart
            y_pos = range(len(names))
            ax_bar.barh(y_pos, values, align='center', color='skyblue')
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(names)
            ax_bar.invert_yaxis()  # Display labels from top to bottom
            ax_bar.set_xlabel('Confidence')
            ax_bar.set_title('Active Expression Labels')
            
            # Add value labels
            for i, v in enumerate(values):
                ax_bar.text(v + 0.01, i, f"{v:.2f}", va='center')
        else:
            ax_bar.text(0.5, 0.5, "No labels exceed threshold", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax_bar.transAxes)
        
        # Draw
        self.canvas.draw()
    
    def _next_image(self):
        """Show next image"""
        if not self.is_loaded or not self.filtered_indices:
            return
        
        self.current_index = (self.current_index + 1) % len(self.filtered_indices)
        self._update_view()
    
    def _prev_image(self):
        """Show previous image"""
        if not self.is_loaded or not self.filtered_indices:
            return
        
        self.current_index = (self.current_index - 1) % len(self.filtered_indices)
        self._update_view()

def main():
    root = tk.Tk()
    app = DatasetViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()