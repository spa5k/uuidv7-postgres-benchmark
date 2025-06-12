#!/usr/bin/env python3
"""
Create simplified explanation diagrams (not data visualizations)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def create_simple_architecture_diagram():
    """Create a clean implementation architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.5, 'ID Generation Implementation Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # UUIDv4 section
    uuid4_box = mpatches.FancyBboxPatch((0.5, 3.5), 2, 1.2, 
                                       boxstyle="round,pad=0.1", 
                                       facecolor='#FF6B6B', alpha=0.8)
    ax.add_patch(uuid4_box)
    ax.text(1.5, 4.1, 'UUIDv4\n(Baseline)', ha='center', va='center', 
            fontweight='bold', fontsize=12, color='white')
    ax.text(1.5, 3.7, '• Pure random\n• No time info\n• PostgreSQL native', 
            ha='center', va='center', fontsize=9, color='white')
    
    # UUIDv7 section
    uuid7_box = mpatches.FancyBboxPatch((3, 3.5), 4, 1.2, 
                                       boxstyle="round,pad=0.1", 
                                       facecolor='#4ECDC4', alpha=0.8)
    ax.add_patch(uuid7_box)
    ax.text(5, 4.1, 'UUIDv7 Implementations', ha='center', va='center', 
            fontweight='bold', fontsize=12, color='white')
    
    # UUIDv7 sub-boxes
    sub_boxes = [
        (3.3, 3.7, 'PL/pgSQL\nOverlay'),
        (5, 3.7, 'Pure SQL\nBit Ops'),
        (6.7, 3.7, 'Sub-ms\nPrecision')
    ]
    
    for x, y, text in sub_boxes:
        sub_box = mpatches.FancyBboxPatch((x-0.4, y-0.25), 0.8, 0.5, 
                                         boxstyle="round,pad=0.05", 
                                         facecolor='white', alpha=0.9)
        ax.add_patch(sub_box)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # ULID section
    ulid_box = mpatches.FancyBboxPatch((7.5, 3.5), 2, 1.2, 
                                      boxstyle="round,pad=0.1", 
                                      facecolor='#FECA57', alpha=0.8)
    ax.add_patch(ulid_box)
    ax.text(8.5, 4.1, 'ULID', ha='center', va='center', 
            fontweight='bold', fontsize=12, color='white')
    ax.text(8.5, 3.7, '• Base32 encoded\n• Human readable\n• Lexicographic sort', 
            ha='center', va='center', fontsize=9, color='white')
    
    # TypeID section
    typeid_box = mpatches.FancyBboxPatch((3.5, 1.8), 3, 1.2, 
                                        boxstyle="round,pad=0.1", 
                                        facecolor='#FF9FF3', alpha=0.8)
    ax.add_patch(typeid_box)
    ax.text(5, 2.4, 'TypeID', ha='center', va='center', 
            fontweight='bold', fontsize=12, color='white')
    ax.text(5, 2.0, '• Prefixed identifiers\n• Type safety\n• Based on UUIDv7', 
            ha='center', va='center', fontsize=9, color='white')
    
    # Performance arrows and notes
    ax.annotate('Fastest\nSingle-threaded', xy=(3.3, 3.5), xytext=(2, 2.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, ha='center', color='green', fontweight='bold')
    
    ax.annotate('Best\nConcurrent', xy=(1.5, 3.5), xytext=(1.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=10, ha='center', color='blue', fontweight='bold')
    
    ax.annotate('Most\nReadable', xy=(8.5, 3.5), xytext=(9.2, 2.5),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                fontsize=10, ha='center', color='orange', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('simple_implementation_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def create_performance_summary_diagram():
    """Create a visual performance summary"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'Performance Characteristics Summary', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Performance categories
    categories = [
        {
            'title': 'Single-threaded Speed',
            'winner': 'UUIDv7 (PL/pgSQL)',
            'value': '72.3 μs',
            'color': '#4ECDC4',
            'pos': (2, 6)
        },
        {
            'title': 'Concurrent Throughput',
            'winner': 'UUIDv4',
            'value': '29,492 IDs/sec',
            'color': '#FF6B6B',
            'pos': (8, 6)
        },
        {
            'title': 'Storage Efficiency',
            'winner': 'ULID',
            'value': '26 bytes',
            'color': '#FECA57',
            'pos': (2, 4)
        },
        {
            'title': 'Human Readability',
            'winner': 'ULID',
            'value': 'Base32 encoded',
            'color': '#FECA57',
            'pos': (8, 4)
        },
        {
            'title': 'Type Safety',
            'winner': 'TypeID',
            'value': 'Prefixed IDs',
            'color': '#FF9FF3',
            'pos': (2, 2)
        },
        {
            'title': 'PostgreSQL Native',
            'winner': 'UUIDv4 & UUIDv7 (v18+)',
            'value': 'Built-in support',
            'color': '#96CEB4',
            'pos': (8, 2)
        }
    ]
    
    for cat in categories:
        x, y = cat['pos']
        
        # Create category box
        box = mpatches.FancyBboxPatch((x-1.4, y-0.6), 2.8, 1.2, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=cat['color'], alpha=0.8)
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y+0.2, cat['title'], ha='center', va='center', 
                fontweight='bold', fontsize=12, color='white')
        ax.text(x, y-0.1, cat['winner'], ha='center', va='center', 
                fontweight='bold', fontsize=10, color='white')
        ax.text(x, y-0.4, cat['value'], ha='center', va='center', 
                fontsize=9, color='white', style='italic')
    
    plt.tight_layout()
    plt.savefig('performance_summary_diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def create_decision_flowchart():
    """Create a decision flowchart for choosing ID types"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'ID Generation Decision Flowchart', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Start node
    start_box = mpatches.FancyBboxPatch((6, 8.5), 2, 0.6, 
                                       boxstyle="round,pad=0.1", 
                                       facecolor='#E8E8E8', alpha=0.8)
    ax.add_patch(start_box)
    ax.text(7, 8.8, 'Choose ID Type', ha='center', va='center', 
            fontweight='bold', fontsize=12)
    
    # Decision nodes and paths
    decisions = [
        {
            'question': 'Need maximum\nconcurrent performance?',
            'pos': (3.5, 7.2),
            'yes_target': (1.5, 6),
            'yes_answer': 'UUIDv4',
            'no_pos': (10.5, 7.2)
        },
        {
            'question': 'Need time ordering?',
            'pos': (10.5, 7.2),
            'yes_target': (10.5, 5.5),
            'yes_answer': 'PostgreSQL 18?\nUse native uuidv7()',
            'no_target': (10.5, 4),
            'no_answer': 'Use UUIDv4'
        },
        {
            'question': 'Need human\nreadability?',
            'pos': (7, 5.5),
            'yes_target': (4.5, 4.5),
            'yes_answer': 'ULID',
            'no_target': (9.5, 4.5),
            'no_answer': 'UUIDv7\n(PL/pgSQL)'
        },
        {
            'question': 'Need type safety?',
            'pos': (7, 3.5),
            'yes_target': (7, 2.5),
            'yes_answer': 'TypeID',
            'no_target': (7, 1.5),
            'no_answer': 'UUIDv7\n(Pure SQL)'
        }
    ]
    
    # Draw decision tree (simplified version)
    # Main decision box
    main_decision = mpatches.FancyBboxPatch((5.5, 6.8), 3, 0.8, 
                                           boxstyle="round,pad=0.1", 
                                           facecolor='#FFE5B4', alpha=0.8)
    ax.add_patch(main_decision)
    ax.text(7, 7.2, 'Primary Requirement?', ha='center', va='center', 
            fontweight='bold', fontsize=11)
    
    # Outcome boxes
    outcomes = [
        {'name': 'UUIDv4\n(Max Performance)', 'pos': (2, 5.5), 'color': '#FF6B6B'},
        {'name': 'UUIDv7 (PL/pgSQL)\n(Balanced)', 'pos': (7, 5.5), 'color': '#4ECDC4'},
        {'name': 'ULID\n(Human Readable)', 'pos': (12, 5.5), 'color': '#FECA57'},
        {'name': 'TypeID\n(Type Safe)', 'pos': (4, 3.5), 'color': '#FF9FF3'},
        {'name': 'PostgreSQL 18\nnative uuidv7()', 'pos': (10, 3.5), 'color': '#96CEB4'},
    ]
    
    for outcome in outcomes:
        x, y = outcome['pos']
        box = mpatches.FancyBboxPatch((x-1, y-0.4), 2, 0.8, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=outcome['color'], alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, outcome['name'], ha='center', va='center', 
                fontweight='bold', fontsize=10, color='white')
    
    # Add arrows (simplified)
    arrow_props = dict(arrowstyle='->', lw=2, color='gray')
    
    # From main decision to outcomes
    ax.annotate('', xy=(2, 6), xytext=(6, 6.8), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 6), xytext=(7, 6.8), arrowprops=arrow_props)
    ax.annotate('', xy=(12, 6), xytext=(8, 6.8), arrowprops=arrow_props)
    
    # Labels for paths
    ax.text(4, 6.3, 'Concurrent\nSpeed', ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(7, 6.3, 'Time\nOrdering', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.text(9.5, 6.3, 'Special\nFeatures', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('decision_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    print("🎨 Creating simplified explanation diagrams...")
    
    create_simple_architecture_diagram()
    print("✅ Created simple_implementation_architecture.png")
    
    create_performance_summary_diagram()
    print("✅ Created performance_summary_diagram.png")
    
    create_decision_flowchart()
    print("✅ Created decision_flowchart.png")
    
    print("\n🎉 All explanation diagrams created successfully!")