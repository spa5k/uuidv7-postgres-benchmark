#!/usr/bin/env python3
"""
Create visual diagrams for UUIDv7 structure and implementation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_uuid_structure_diagram():
    """Create a visual representation of UUIDv7 structure"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # UUID sections with bit positions
    sections = [
        {"name": "unix_ts_ms\n(48 bits)", "start": 0, "width": 48, "color": "#3498db"},
        {"name": "ver\n(4 bits)", "start": 48, "width": 4, "color": "#e74c3c"},
        {"name": "rand_a\n(12 bits)", "start": 52, "width": 12, "color": "#2ecc71"},
        {"name": "var\n(2 bits)", "start": 64, "width": 2, "color": "#f39c12"},
        {"name": "rand_b\n(62 bits)", "start": 66, "width": 62, "color": "#9b59b6"}
    ]
    
    # Draw sections
    y_pos = 0.5
    height = 0.3
    total_bits = 128
    
    for section in sections:
        x = section["start"] / total_bits
        width = section["width"] / total_bits
        
        # Create fancy box
        box = FancyBboxPatch(
            (x, y_pos), width, height,
            boxstyle="round,pad=0.01",
            facecolor=section["color"],
            edgecolor="black",
            alpha=0.8,
            linewidth=2
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(x + width/2, y_pos + height/2, section["name"], 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Add bit positions
        ax.text(x, y_pos - 0.05, str(section["start"]), 
                ha='center', va='top', fontsize=8)
    
    # Add final bit position
    ax.text(1, y_pos - 0.05, "128", ha='center', va='top', fontsize=8)
    
    # UUID string representation
    uuid_str = "TTTTTTTT-TTTT-7RRR-VARR-RRRRRRRRRRRR"
    ax.text(0.5, 0.1, uuid_str, ha='center', va='center', 
            fontsize=14, fontfamily='monospace', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    # Labels
    ax.text(0.5, 0.9, "UUIDv7 Structure (128 bits)", 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Format axes
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('uuid_structure.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_implementation_flow_diagram():
    """Create a flow diagram showing the three implementations"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Function 1 & 2 flow
    flow1_steps = [
        "Generate UUIDv4",
        "Extract timestamp\n(ms since epoch)",
        "Convert to bytes",
        "Overlay timestamp\n(first 48 bits)",
        "Set version bits\n(make it v7)",
        "Convert to hex"
    ]
    
    y = 0.5
    x_start = 0.05
    x_step = 0.15
    box_width = 0.12
    box_height = 0.3
    
    for i, step in enumerate(flow1_steps):
        x = x_start + i * x_step
        
        # Box
        box = FancyBboxPatch(
            (x, y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.01",
            facecolor='lightblue' if i % 2 == 0 else 'lightgreen',
            edgecolor='black',
            linewidth=1.5
        )
        ax1.add_patch(box)
        
        # Text
        ax1.text(x + box_width/2, y, step, 
                ha='center', va='center', fontsize=9, wrap=True)
        
        # Arrow
        if i < len(flow1_steps) - 1:
            ax1.arrow(x + box_width, y, x_step - box_width - 0.01, 0,
                     head_width=0.05, head_length=0.01, fc='black', ec='black')
    
    ax1.text(0.5, 0.85, "Function 1 & 2: Overlay Method", 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Function 3 flow
    flow3_steps = [
        "Get timestamp\nwith sub-ms",
        "Split integer\nand fraction",
        "Create 48-bit\ntimestamp",
        "Create version\n+ sub-ms bits",
        "Generate\nrandom bytes",
        "Concatenate\nall parts"
    ]
    
    for i, step in enumerate(flow3_steps):
        x = x_start + i * x_step
        
        # Box
        box = FancyBboxPatch(
            (x, y - box_height/2), box_width, box_height,
            boxstyle="round,pad=0.01",
            facecolor='lightyellow' if i % 2 == 0 else 'lightcoral',
            edgecolor='black',
            linewidth=1.5
        )
        ax2.add_patch(box)
        
        # Text
        ax2.text(x + box_width/2, y, step, 
                ha='center', va='center', fontsize=9, wrap=True)
        
        # Arrow
        if i < len(flow3_steps) - 1:
            ax2.arrow(x + box_width, y, x_step - box_width - 0.01, 0,
                     head_width=0.05, head_length=0.01, fc='black', ec='black')
    
    ax2.text(0.5, 0.85, "Function 3: Construction Method (Sub-millisecond)", 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Comparison table
    comparison_data = [
        ["Feature", "Functions 1 & 2", "Function 3"],
        ["Method", "Overlay on UUIDv4", "Build from scratch"],
        ["Precision", "Millisecond", "Sub-millisecond"],
        ["Random bits", "74 bits", "62 bits"],
        ["Complexity", "Simple", "Moderate"],
        ["Performance", "Fast", "Slightly slower"]
    ]
    
    # Create table
    cell_colors = [['lightgray']*3] + [['white']*3]*5
    table = ax3.table(cellText=comparison_data, 
                     cellLoc='center',
                     loc='center',
                     cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    ax3.text(0.5, 0.9, "Implementation Comparison", 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('implementation_flow.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_collision_probability_chart():
    """Create a chart showing collision probabilities"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # UUID generation rates (per millisecond)
    rates = np.logspace(0, 9, 100)  # 1 to 1 billion
    
    # Collision probabilities
    # P(collision) ≈ n²/2^b where n = number of UUIDs, b = random bits
    prob_74_bits = rates**2 / (2**74)
    prob_62_bits = rates**2 / (2**62)
    
    # Plot
    ax.loglog(rates, prob_74_bits, 'b-', linewidth=2, label='Functions 1 & 2 (74 random bits)')
    ax.loglog(rates, prob_62_bits, 'r-', linewidth=2, label='Function 3 (62 random bits)')
    
    # Reference lines
    ax.axhline(y=1e-15, color='green', linestyle='--', alpha=0.5, label='1 in quadrillion')
    ax.axhline(y=1e-9, color='orange', linestyle='--', alpha=0.5, label='1 in billion')
    ax.axhline(y=1e-6, color='red', linestyle='--', alpha=0.5, label='1 in million')
    
    # Formatting
    ax.set_xlabel('UUIDs Generated per Millisecond', fontsize=12)
    ax.set_ylabel('Collision Probability', fontsize=12)
    ax.set_title('UUIDv7 Collision Probability by Generation Rate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotations
    ax.annotate('Safe for most applications', 
                xy=(1e6, 1e-15), xytext=(1e7, 1e-13),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('collision_probability.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_bit_manipulation_diagram():
    """Create a visual showing bit manipulation in Function 1"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Steps visualization
    steps = [
        {
            "title": "Step 1: Original UUIDv4",
            "bits": "RRRRRRRR-RRRR-4RRR-VARR-RRRRRRRRRRRR",
            "highlight": [],
            "y": 0.8
        },
        {
            "title": "Step 2: After timestamp overlay",
            "bits": "TTTTTTTT-TTTT-4RRR-VARR-RRRRRRRRRRRR",
            "highlight": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
            "y": 0.6
        },
        {
            "title": "Step 3: Set bit 52 (version bit 1)",
            "bits": "TTTTTTTT-TTTT-5RRR-VARR-RRRRRRRRRRRR",
            "highlight": [14],
            "y": 0.4
        },
        {
            "title": "Step 4: Set bit 53 (version bit 2) - Final UUIDv7",
            "bits": "TTTTTTTT-TTTT-7RRR-VARR-RRRRRRRRRRRR",
            "highlight": [14],
            "y": 0.2
        }
    ]
    
    for step in steps:
        # Background box
        box = FancyBboxPatch(
            (0.1, step["y"] - 0.05), 0.8, 0.1,
            boxstyle="round,pad=0.01",
            facecolor='lightgray',
            alpha=0.3,
            edgecolor='black'
        )
        ax.add_patch(box)
        
        # Title
        ax.text(0.05, step["y"] + 0.03, step["title"], 
                fontsize=11, fontweight='bold')
        
        # UUID bits
        x_pos = 0.15
        char_width = 0.045
        for i, char in enumerate(step["bits"]):
            color = 'red' if i in step["highlight"] else 'black'
            weight = 'bold' if i in step["highlight"] else 'normal'
            ax.text(x_pos + i * char_width, step["y"] - 0.02, char, 
                   fontsize=12, fontfamily='monospace', 
                   color=color, fontweight=weight)
    
    # Add bit position indicators
    positions = {
        "Bit 0": 0.15,
        "Bit 48": 0.15 + 13 * 0.045,
        "Bit 52": 0.15 + 14 * 0.045,
        "Bit 64": 0.15 + 18 * 0.045,
        "Bit 127": 0.15 + 35 * 0.045
    }
    
    for label, x in positions.items():
        ax.annotate(label, xy=(x, 0.1), xytext=(x, 0.05),
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5),
                   ha='center', fontsize=8, color='gray')
    
    ax.text(0.5, 0.95, "UUIDv7 Generation: Bit Manipulation Process", 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('bit_manipulation.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == "__main__":
    print("Creating UUID structure diagram...")
    create_uuid_structure_diagram()
    
    print("Creating implementation flow diagram...")
    create_implementation_flow_diagram()
    
    print("Creating collision probability chart...")
    create_collision_probability_chart()
    
    print("Creating bit manipulation diagram...")
    create_bit_manipulation_diagram()
    
    print("All diagrams created successfully!")