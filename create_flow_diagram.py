"""
Enhanced Flow Diagram Generator for LangGraph Agent Architecture
===============================================================

Creates a clear, professional workflow diagram showing the multi-agent system
for e-commerce return prevention with detailed component descriptions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

def create_enhanced_workflow_diagram():
    """Create an enhanced, clearly readable workflow diagram"""
    
    # Create figure with high DPI for clarity
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme for better readability
    colors = {
        'start': '#4CAF50',      # Green
        'data': '#2196F3',       # Blue  
        'ml': '#FF9800',         # Orange
        'nlp': '#E91E63',        # Pink
        'decision': '#F44336',   # Red
        'action': '#9C27B0',     # Purple
        'end': '#607D8B'         # Blue Grey
    }
    
    # Title
    ax.text(8, 11.5, 'LangGraph Multi-Agent Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    ax.text(8, 11, 'E-commerce Return Prevention System', 
            fontsize=16, ha='center', style='italic')
    
    # Helper function to create rounded rectangles
    def create_agent_box(x, y, width, height, color, label, description):
        # Main box
        box = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Label
        ax.text(x + width/2, y + height/2 + 0.15, label,
                fontsize=12, fontweight='bold', ha='center', va='center')
        
        # Description
        ax.text(x + width/2, y + height/2 - 0.15, description,
                fontsize=9, ha='center', va='center', wrap=True)
    
    # Helper function to create arrows
    def create_arrow(x1, y1, x2, y2, label="", offset=0.1):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2 + offset
            ax.text(mid_x, mid_y, label, fontsize=9, ha='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # 1. Start Node
    start_circle = Circle((2, 9), 0.5, facecolor=colors['start'], 
                         edgecolor='black', linewidth=2)
    ax.add_patch(start_circle)
    ax.text(2, 9, 'START\nTransaction\nInitiated', fontsize=10, fontweight='bold', 
            ha='center', va='center')
    
    # 2. Data Collection Agent
    create_agent_box(1, 7, 2.5, 1.2, colors['data'], 
                    'Data Collection Agent',
                    'Gathers customer, product\n& seller information')
    
    # 3. Parallel Processing - Return Predictor
    create_agent_box(5, 8, 2.5, 1.2, colors['ml'],
                    'Return Prediction Agent',
                    'Neural Network with MFE\nPredicts return probability')
    
    # 4. Parallel Processing - Sentiment Analyzer  
    create_agent_box(5, 6, 2.5, 1.2, colors['nlp'],
                    'Sentiment Analysis Agent', 
                    'HuggingFace Transformers\nAnalyzes customer sentiment')
    
    # 5. Decision Diamond
    decision_diamond = mpatches.RegularPolygon((10, 7.5), 4, radius=0.8, 
                                             orientation=np.pi/4,
                                             facecolor=colors['decision'], 
                                             edgecolor='black', linewidth=2)
    ax.add_patch(decision_diamond)
    ax.text(10, 7.5, 'Risk\nAssessment\n& Decision', fontsize=10, fontweight='bold',
            ha='center', va='center')
    
    # 6. Notification Agent
    create_agent_box(12.5, 8.5, 2.5, 1.2, colors['action'],
                    'Notification Agent',
                    'Alerts customers\n& sellers')
    
    # 7. Recommendation Agent
    create_agent_box(12.5, 6, 2.5, 1.2, colors['action'],
                    'Recommendation Agent',
                    'Generates alternatives\n& improvements')
    
    # 8. End Node
    end_circle = Circle((14, 4.5), 0.5, facecolor=colors['end'], 
                       edgecolor='black', linewidth=2)
    ax.add_patch(end_circle)
    ax.text(14, 4.5, 'END\nProcess\nComplete', fontsize=10, fontweight='bold',
            ha='center', va='center')
    
    # Arrows with labels
    create_arrow(2.5, 9, 2.25, 8.2, "Initiate")
    create_arrow(3.5, 7.6, 5, 8.6, "Customer\nProduct\nSeller Data")
    create_arrow(3.5, 7.6, 5, 6.6, "Feedback\nHistory")
    create_arrow(7.5, 8.6, 9.2, 7.8, "Return\nProbability")
    create_arrow(7.5, 6.6, 9.2, 7.2, "Sentiment\nScore")
    create_arrow(10.8, 8, 12.5, 9, "High Risk")
    create_arrow(10.8, 7, 12.5, 6.5, "Generate\nRecommendations")
    create_arrow(14, 8.5, 14, 5, "")
    create_arrow(14, 6, 14, 5, "")
    
    # Technology Stack Information
    tech_box = Rectangle((0.5, 4), 7, 2.5, facecolor='lightblue', 
                        edgecolor='black', linewidth=1, alpha=0.3)
    ax.add_patch(tech_box)
    
    ax.text(4, 5.8, '🔧 Technology Stack', fontsize=14, fontweight='bold', ha='center')
    
    tech_info = [
        "🧠 Neural Network: TensorFlow/Keras with MFE features",
        "🤗 NLP Models: HuggingFace Transformers (RoBERTa, DistilRoBERTa)", 
        "🔗 Orchestration: LangGraph + LangChain multi-agent framework",
        "📊 Features: Customer behavior, Product data, Seller metrics",
        "🎯 Output: 0-100% return probability prediction",
        "📱 Alerts: Real-time notifications to stakeholders"
    ]
    
    for i, info in enumerate(tech_info):
        ax.text(0.7, 5.4 - i*0.25, info, fontsize=10, va='center')
    
    # Risk Level Classification
    risk_box = Rectangle((8.5, 4), 6.5, 2.5, facecolor='lightyellow', 
                        edgecolor='black', linewidth=1, alpha=0.3)
    ax.add_patch(risk_box)
    
    ax.text(11.75, 5.8, '⚠️ Risk Level Classification', fontsize=14, fontweight='bold', ha='center')
    
    risk_levels = [
        "🔴 CRITICAL (70%+): Immediate intervention",
        "🟠 HIGH (50-70%): Strong preventive action", 
        "🟡 MEDIUM (30-50%): Moderate guidance",
        "🟢 LOW (<30%): Standard processing"
    ]
    
    for i, level in enumerate(risk_levels):
        ax.text(8.7, 5.4 - i*0.25, level, fontsize=10, va='center')
    
    # Data Flow Annotations
    ax.text(1, 3, '📊 Data Flow Process:', fontsize=12, fontweight='bold')
    
    process_steps = [
        "1. Customer selects product → Transaction initiated",
        "2. Data Collection Agent gathers customer, product, seller data", 
        "3. Return Predictor uses Neural Network with MFE features",
        "4. Sentiment Analyzer uses HuggingFace Transformers",
        "5. Decision Point assesses risk level (Critical/High/Medium/Low)",
        "6. High Risk → Notifications sent to customer & seller",
        "7. Recommendations generated (alternatives, size guide, improvements)",
        "8. Process complete with enhanced customer experience"
    ]
    
    for i, step in enumerate(process_steps):
        ax.text(1, 2.6 - i*0.25, step, fontsize=9, va='center')
    
    # Legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['start'], label='Start/End Points'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['data'], label='Data Processing'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['ml'], label='ML Prediction'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['nlp'], label='NLP Analysis'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['decision'], label='Decision Point'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['action'], label='Action Agents')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Save with high quality
    plt.tight_layout()
    plt.savefig('enhanced_langgraph_workflow.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('enhanced_langgraph_workflow.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("✅ Enhanced workflow diagrams saved:")
    print("   📄 enhanced_langgraph_workflow.png (High-res PNG)")
    print("   📄 enhanced_langgraph_workflow.pdf (Vector PDF)")
    
    plt.show()

def create_detailed_architecture_diagram():
    """Create a detailed technical architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(9, 13.5, 'Detailed LangGraph Agent Architecture', 
            fontsize=22, fontweight='bold', ha='center')
    ax.text(9, 13, 'Multi-Agent System for E-commerce Return Prevention', 
            fontsize=16, ha='center', style='italic')
    
    # Layer 1: Input Layer
    input_box = Rectangle((1, 11), 16, 1.5, facecolor='lightgreen', 
                         edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(input_box)
    ax.text(9, 11.75, '📥 INPUT LAYER: Transaction Data', 
            fontsize=14, fontweight='bold', ha='center')
    
    inputs = ["Customer ID", "Product ID", "Seller ID", "Payment Method", "Order Details"]
    for i, inp in enumerate(inputs):
        ax.text(2 + i*3, 11.3, inp, fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white'))
    
    # Layer 2: Data Collection & Preprocessing
    data_box = Rectangle((1, 9), 16, 1.5, facecolor='lightblue', 
                        edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(data_box)
    ax.text(9, 9.75, '🔍 DATA COLLECTION & PREPROCESSING LAYER', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Data sources
    data_sources = [
        "Customer\nHistory DB", "Product\nCatalog", "Seller\nMetrics", 
        "Review\nSentiments", "Transaction\nLogs"
    ]
    for i, source in enumerate(data_sources):
        ax.text(2.5 + i*3, 9.3, source, fontsize=9, ha='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white'))
    
    # Layer 3: Parallel Processing
    ml_box = Rectangle((1, 6.5), 7.5, 2, facecolor='orange', 
                      edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(ml_box)
    ax.text(4.75, 7.8, '🧠 ML PREDICTION ENGINE', 
            fontsize=12, fontweight='bold', ha='center')
    
    nlp_box = Rectangle((9.5, 6.5), 7.5, 2, facecolor='pink', 
                       edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(nlp_box)
    ax.text(13.25, 7.8, '💭 NLP SENTIMENT ENGINE', 
            fontsize=12, fontweight='bold', ha='center')
    
    # ML Components
    ml_components = [
        "Feature\nExtraction", "MFE\nTransformation", "Neural\nNetwork", "Return\nPrediction"
    ]
    for i, comp in enumerate(ml_components):
        ax.text(1.8 + i*1.7, 7.2, comp, fontsize=9, ha='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow'))
    
    # NLP Components  
    nlp_components = [
        "Text\nPreprocessing", "Sentiment\nClassification", "Emotion\nDetection", "Concern\nIdentification"
    ]
    for i, comp in enumerate(nlp_components):
        ax.text(10.3 + i*1.7, 7.2, comp, fontsize=9, ha='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcyan'))
    
    # Layer 4: Decision & Action
    decision_box = Rectangle((7, 4.5), 4, 1.5, facecolor='red', 
                           edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(decision_box)
    ax.text(9, 5.25, '⚖️ DECISION ENGINE', 
            fontsize=12, fontweight='bold', ha='center', color='white')
    
    # Action Agents
    notification_box = Rectangle((1, 2.5), 7, 1.5, facecolor='purple', 
                               edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(notification_box)
    ax.text(4.5, 3.25, '📨 NOTIFICATION AGENT', 
            fontsize=12, fontweight='bold', ha='center', color='white')
    
    recommendation_box = Rectangle((10, 2.5), 7, 1.5, facecolor='indigo', 
                                 edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(recommendation_box)
    ax.text(13.5, 3.25, '💡 RECOMMENDATION AGENT', 
            fontsize=12, fontweight='bold', ha='center', color='white')
    
    # Output Layer
    output_box = Rectangle((1, 0.5), 16, 1.5, facecolor='lightgray', 
                          edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(output_box)
    ax.text(9, 1.25, '📤 OUTPUT LAYER: Enhanced Customer Experience', 
            fontsize=14, fontweight='bold', ha='center')
    
    outputs = ["Customer\nAlerts", "Seller\nNotifications", "Product\nAlternatives", 
               "Size\nGuidance", "Risk\nMitigation"]
    for i, out in enumerate(outputs):
        ax.text(2.5 + i*3, 0.8, out, fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white'))
    
    # Add arrows between layers
    arrow_props = dict(arrowstyle='->', lw=3, color='darkblue')
    
    # Input to Data Collection
    ax.annotate('', xy=(9, 9), xytext=(9, 11), arrowprops=arrow_props)
    
    # Data Collection to Parallel Processing
    ax.annotate('', xy=(4.75, 8.5), xytext=(6, 9), arrowprops=arrow_props)
    ax.annotate('', xy=(13.25, 8.5), xytext=(12, 9), arrowprops=arrow_props)
    
    # Parallel Processing to Decision
    ax.annotate('', xy=(8.5, 5.5), xytext=(4.75, 6.5), arrowprops=arrow_props)
    ax.annotate('', xy=(9.5, 5.5), xytext=(13.25, 6.5), arrowprops=arrow_props)
    
    # Decision to Actions
    ax.annotate('', xy=(4.5, 4), xytext=(8, 4.5), arrowprops=arrow_props)
    ax.annotate('', xy=(13.5, 4), xytext=(10, 4.5), arrowprops=arrow_props)
    
    # Actions to Output
    ax.annotate('', xy=(6, 2), xytext=(4.5, 2.5), arrowprops=arrow_props)
    ax.annotate('', xy=(12, 2), xytext=(13.5, 2.5), arrowprops=arrow_props)
    
    plt.tight_layout()
    plt.savefig('detailed_architecture_diagram.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("✅ Detailed architecture diagram saved:")
    print("   📄 detailed_architecture_diagram.png")
    
    plt.show()

if __name__ == "__main__":
    print("🎨 Creating Enhanced Flow Diagrams...")
    print("=" * 50)
    
    # Create enhanced workflow diagram
    create_enhanced_workflow_diagram()
    
    print("\n" + "=" * 50)
    
    # Create detailed architecture diagram
    create_detailed_architecture_diagram()
    
    print("\n🎉 All diagrams created successfully!")
    print("Files generated:")
    print("  📄 enhanced_langgraph_workflow.png - Main workflow diagram")
    print("  📄 enhanced_langgraph_workflow.pdf - Vector version")
    print("  📄 detailed_architecture_diagram.png - Technical architecture")
