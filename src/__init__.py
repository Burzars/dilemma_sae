"""dilemma_sae — пайплайн интерпретации hidden states LLM через Sparse Autoencoder.

Содержание модулей:
  - data        : ReasoningSample, load_samples, save_samples, load_saved_samples
  - extract     : ActivationConfig, _build_chat_text, extract_activations
  - sae         : STEHeaviside, heaviside_ste, SparseAutoencoder, train_sae
  - analysis    : encode_all, neuron_statistics, select_top_neurons,
                  top_contexts_for_neuron, label_contrast, sample_level_contrast
  - interpret   : LLM-судья (OpenRouter) → темы нейронов
  - visualize   : графики (train curves, Pareto, fire_rate hist и т.п.)
  - io_utils    : save/load активаций, SAE, отчётов; конфиги
"""

from . import analysis, data, extract, interpret, io_utils, sae, visualize

__all__ = [
    "analysis",
    "data",
    "extract",
    "interpret",
    "io_utils",
    "sae",
    "visualize",
]
