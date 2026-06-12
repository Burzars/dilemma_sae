"""dilemma_sae — пайплайн интерпретации hidden states LLM через Sparse Autoencoder.

Содержание модулей:
  - data        : ReasoningSample, load_samples, save_samples, load_saved_samples
  - extract     : ActivationConfig, _build_chat_text, extract_activations
  - sae         : SparseAutoencoder, train_sae, sae_kwargs_for_mode,
                  JumpReLUFunction, HeavisideFunction
  - analysis    : encode_all, recon_ev, neuron_statistics, select_top_neurons,
                  top_contexts_for_neuron, label_contrast, sample_level_contrast
  - interpret   : LLM-судья (OpenRouter) → темы нейронов; классификатор Yes/No/?
  - steering    : forward-hook стиринг резстрима, батч-генерация (эксперимент 07)
  - visualize   : графики (train curves, Pareto, fire_rate hist, EV, steering)
  - io_utils    : save/load активаций, SAE, отчётов; конфиги
"""

# Ленивый импорт подмодулей (PEP 562): `from src.X import Y` и `src.X` работают,
# но пакет НЕ тянет torch/numpy при импорте, если они не нужны. Это важно для
# инструментов генерации данных (generate_answers, build_datasets, interpret) —
# они ходят только в OpenRouter и не должны требовать GPU-стек.
import importlib

__all__ = [
    "analysis",
    "data",
    "extract",
    "interpret",
    "io_utils",
    "sae",
    "steering",
    "visualize",
    "scenarios",
    "generate_answers",
    "build_datasets",
]


def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
