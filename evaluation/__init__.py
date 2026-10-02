"""
Evaluation and Visualization module.
Includes Evaluator, Visualizer, Comparison, and Report Generator.
"""

from .evaluator import Evaluator
from .visualization import generate_four_panel_visualization, generate_overlay, generate_attention_plot
from .comparison import compare_baseline_models
from .report_generator import generate_academic_report, generate_architecture_diagram

__all__ = [
    "Evaluator",
    "generate_four_panel_visualization",
    "generate_overlay",
    "generate_attention_plot",
    "compare_baseline_models",
    "generate_academic_report",
    "generate_architecture_diagram",
]
