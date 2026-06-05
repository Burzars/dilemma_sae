"""Sparse Autoencoder с тремя архитектурами на выбор.

Три режима в одном классе (mode):

  "relu_l1"   — стандартный ReLU + L1-штраф (Bricken et al. 2023, как в FAST).
                Параметры: l1_coef.
                ВНИМАНИЕ: на этих данных не достигает нужной разреженности
                (L0 застревает на ~8400 нейронов даже при l1_coef=5e-2).

  "topk"      — TopK без L1 (Gao et al. 2024). Sparsity = K by construction.
                Параметры: k. РАБОТАЕТ ХОРОШО.

  "jumprelu"  — JumpReLU + STE + прямая оптимизация L0 (Rajamanoharan et al. 2024).
                Параметры: l0_coef, theta_init, ste_eps.
                STE реализован по статье (см. JumpReLUFunction / HeavisideFunction):
                активация и L0-ступенька — отдельные autograd.Function с
                ПРАВИЛЬНЫМИ псевдо-градиентами по порогу θ.
                ВНИМАНИЕ ПО МАСШТАБУ: θ и ε живут в пространстве пре-активаций
                (норма (x−b_dec) здесь ~150, проекции на learned-направления
                порядка единиц–десятков), поэтому theta_init≈1–3 и ste_eps≈0.5–2.
                Старые значения 0.1/0.1 давали почти всегда пустое STE-окно
                |z−θ|<ε/2 → θ не получал градиента → деградация обучения.

Общие для всех режимов трюки:
  * вычитание b_dec ПЕРЕД энкодером (привязка к среднему облака),
  * нормализация колонок W_dec на единицу,
  * удаление параллельной составляющей градиента W_dec в train-loop
    (чтобы Adam не пытался удлинить декодерные направления, длина
    всё равно фиксирована нормализацией).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


def _rectangle(x: torch.Tensor) -> torch.Tensor:
    """Прямоугольное ядро K ширины 1, центрированное в 0: 1 при |x|<1/2, иначе 0.

    Через него Rajamanoharan et al. 2024 задают псевдо-градиенты по порогу:
    K((z−θ)/ε) ненулевой ровно в окне |z−θ| < ε/2.
    """
    return ((x > -0.5) & (x <= 0.5)).to(x.dtype)


def _sum_to_theta(grad: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Свернуть градиент формы (..., d_hidden) к форме θ = (d_hidden,).

    θ один на нейрон, а grad приходит поэлементно по батчу — autograd.Function
    обязан вернуть градиент ТОЧНО формы входа θ, поэтому суммируем лишние оси.
    """
    extra = grad.dim() - theta.dim()
    if extra > 0:
        grad = grad.sum(dim=tuple(range(extra)))
    return grad


class JumpReLUFunction(torch.autograd.Function):
    """JumpReLU-активация f(z) = z · H(z − θ) (Rajamanoharan et al. 2024).

    Forward точный. Backward (псевдо-градиенты, прил. B статьи):
      ∂f/∂z = H(z − θ)                  — точная производная почти всюду (= гейт);
      ∂f/∂θ = −(θ/ε) · K((z − θ)/ε)     — высота скачка функции в точке θ равна θ.

    Важно: коэффициент при ядре — именно θ (высота разрыва), а НЕ z; и в ∂f/∂z
    нет «паразитного» члена z·δ(z−θ). Прошлая реализация (f = z·heaviside_ste(z,θ))
    через autograd порождала и лишний член z·K/ε в градиенте к W_enc, и коэффициент
    z вместо θ.
    """

    @staticmethod
    def forward(ctx, z, theta, eps):
        ctx.save_for_backward(z, theta)
        ctx.eps = eps
        return z * (z > theta).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        z, theta = ctx.saved_tensors
        eps = ctx.eps
        grad_z = grad_output * (z > theta).to(grad_output.dtype)
        k = _rectangle((z - theta) / eps)
        grad_theta = _sum_to_theta(grad_output * (-(theta / eps) * k), theta)
        return grad_z, grad_theta, None


class HeavisideFunction(torch.autograd.Function):
    """Ступенька H(z − θ) для L0-штрафа (Rajamanoharan et al. 2024).

    Forward точный. Backward (прил. B статьи):
      ∂H/∂z = 0                         — L0 НЕ должен давать градиент энкодеру;
      ∂H/∂θ = −(1/ε) · K((z − θ)/ε)     — двигает только сам порог θ.
    """

    @staticmethod
    def forward(ctx, z, theta, eps):
        ctx.save_for_backward(z, theta)
        ctx.eps = eps
        return (z > theta).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        z, theta = ctx.saved_tensors
        eps = ctx.eps
        k = _rectangle((z - theta) / eps)
        grad_theta = _sum_to_theta(grad_output * (-(1.0 / eps) * k), theta)
        # ∂/∂z = 0 (None) — ключевой момент: L0 двигает порог, а не W_enc.
        return None, grad_theta, None


class SparseAutoencoder(nn.Module):
    """Объединённый SAE c режимами relu_l1 / topk / jumprelu.

    Аргументы lr1_coef / k / l0_coef / theta_init / ste_eps активны
    только в соответствующих режимах; в остальных режимах хранятся,
    но в forward не используются.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        mode: str = "relu_l1",
        l1_coef: float = 5e-4,
        k: int = 64,
        l0_coef: float = 1e-2,
        theta_init: float = 1.0,   # в пространстве пре-активаций (норма ~единицы-десятки)
        ste_eps: float = 1.0,      # ширина STE-окна по порогу; см. JumpReLUFunction
    ):
        super().__init__()
        if mode not in ("relu_l1", "topk", "jumprelu"):
            raise ValueError(f"unknown mode: {mode}")

        self.d_in, self.d_hidden = d_in, d_hidden
        self.mode = mode
        self.l1_coef = l1_coef
        self.k = k
        self.l0_coef = l0_coef
        self.ste_eps = ste_eps

        self.W_enc = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Обучаемый порог для JumpReLU (по одному на нейрон).
        # log-параметризация: θ = exp(log_theta), чтобы θ > 0 всегда.
        # Для других режимов параметр существует, но в forward не используется.
        self.log_theta = nn.Parameter(
            torch.full((d_hidden,), math.log(theta_init))
        )

        # Инициализация: W_dec — случайные единичные векторы, W_enc = W_dec.T.
        with torch.no_grad():
            W = torch.randn(d_hidden, d_in)
            W = W / W.norm(dim=1, keepdim=True)
            self.W_dec.copy_(W)
            self.W_enc.copy_(W.t())

    def encode(self, x, return_gate: bool = False):
        """Преобразование hidden → разреженный feature vector.

        Если return_gate=True, дополнительно возвращает индикатор
        активности нейронов — нужен в loss для JumpReLU, чтобы не
        считать forward дважды.
        """
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        gate = None

        if self.mode == "relu_l1":
            f = F.relu(pre)

        elif self.mode == "topk":
            topk_vals, topk_idx = pre.topk(self.k, dim=-1)
            f = torch.zeros_like(pre)
            f.scatter_(-1, topk_idx, F.relu(topk_vals))

        elif self.mode == "jumprelu":
            theta = self.log_theta.exp()
            f = JumpReLUFunction.apply(pre, theta, self.ste_eps)
            # Отдельный гейт для L0-штрафа — со своим (нулевым по z) backward.
            if return_gate:
                gate = HeavisideFunction.apply(pre, theta, self.ste_eps)

        if return_gate:
            return f, gate
        return f

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

    def loss(self, x) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        f, gate = self.encode(x, return_gate=True)
        x_hat = self.decode(f)
        recon = (x_hat - x).pow(2).sum(dim=-1).mean()

        # Диагностические метрики (всегда, для логов).
        with torch.no_grad():
            l0 = (f > 0).float().sum(dim=-1).mean()
            l1 = f.abs().sum(dim=-1).mean()

        if self.mode == "relu_l1":
            total = recon + self.l1_coef * f.abs().sum(dim=-1).mean()

        elif self.mode == "topk":
            total = recon       # L0 уже зафиксирован = k

        else:  # jumprelu
            # Прямой L0-штраф: gate имеет STE-градиенты, gate.sum даёт L0,
            # её градиент через STE толкает theta вверх/вниз.
            l0_for_loss = gate.sum(dim=-1).mean()
            total = recon + self.l0_coef * l0_for_loss

        return total, {"recon": recon.detach(), "l1": l1.detach(), "l0": l0.detach()}

    @torch.no_grad()
    def normalize_decoder(self):
        """Нормируем колонки W_dec в единичную длину."""
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)


def sae_kwargs_for_mode(sae_cfg: dict) -> dict:
    """Собирает только релевантные для режима kwargs SAE из секции cfg.sae.

    Зеркалит логику scripts/02_train_sae.py::_sae_kwargs, но живёт в src/,
    чтобы новые скрипты (06_eval_ev, 07_steering) переиспользовали её, не
    дублируя и не импортируя из скрипта с цифрами в имени.
    """
    mode = sae_cfg["mode"]
    kw = {"mode": mode}
    if mode == "relu_l1":
        kw["l1_coef"] = sae_cfg["l1_coef"]
    elif mode == "topk":
        kw["k"] = sae_cfg["k"]
    elif mode == "jumprelu":
        kw["l0_coef"] = sae_cfg["l0_coef"]
        kw["theta_init"] = sae_cfg["theta_init"]
        kw["ste_eps"] = sae_cfg["ste_eps"]
    return kw


def train_sae(
    hidden: np.ndarray,
    expansion: int = 8,
    lr: float = 5e-4,
    epochs: int = 40,
    batch_size: int = 1024,
    device: str = "cuda",
    seed: int = 0,
    verbose: bool = True,
    **sae_kwargs,
) -> Tuple[SparseAutoencoder, Dict[str, list]]:
    """Обучает SAE одного из режимов на матрице активаций (N, d_in).

    Все параметры конкретной архитектуры (mode, l1_coef, k, l0_coef и т.п.)
    передаются через **sae_kwargs и прокидываются в конструктор SAE.

    Returns
    -------
    (SparseAutoencoder, history)
        history — dict со списками значений recon / l1 / l0 / total по эпохам.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, d_in = hidden.shape
    d_hidden = d_in * expansion
    X = torch.from_numpy(hidden.astype(np.float32))

    sae = SparseAutoencoder(d_in, d_hidden, **sae_kwargs).to(device)
    with torch.no_grad():
        # Стартуем с центра облака — значительно ускоряет сходимость.
        sae.b_dec.copy_(X.mean(dim=0).to(device))

    if verbose:
        n_params = sum(p.numel() for p in sae.parameters())
        log.info(
            "  mode=%s, d_hidden=%d, params=%.1fM, epochs=%d",
            sae.mode, d_hidden, n_params / 1e6, epochs,
        )

    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    history = {"recon": [], "l1": [], "l0": [], "total": []}
    for epoch in range(epochs):
        ep = {"recon": 0., "l1": 0., "l0": 0., "total": 0., "n": 0}
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            total, comp = sae.loss(xb)

            opt.zero_grad(set_to_none=True)
            total.backward()

            # КРИТИЧНО: убираем компоненту градиента W_dec, удлиняющую/
            # укорачивающую колонки — нормализация их всё равно отменит,
            # не тратим шаги Adam.
            with torch.no_grad():
                W = sae.W_dec
                g = W.grad
                proj = (W * g).sum(dim=1, keepdim=True) * W
                g.sub_(proj)

            opt.step()
            sae.normalize_decoder()

            bs = xb.shape[0]
            ep["recon"] += comp["recon"].item() * bs
            ep["l1"]    += comp["l1"].item()    * bs
            ep["l0"]    += comp["l0"].item()    * bs
            ep["total"] += total.item()         * bs
            ep["n"]     += bs

        for key in ("recon", "l1", "l0", "total"):
            history[key].append(ep[key] / ep["n"])
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            log.info(
                "  эп. %3d/%d  recon=%.3f  L1=%.1f  L0=%.1f  total=%.3f",
                epoch + 1, epochs,
                history["recon"][-1], history["l1"][-1],
                history["l0"][-1], history["total"][-1],
            )
    return sae, history
