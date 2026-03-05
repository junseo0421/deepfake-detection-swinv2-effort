import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# SVD Residual Linear
# -------------------------
class SVDResidualLinear(nn.Module):
    """
    W ≈ W_main (frozen) + U_res @ diag(S_res) @ V_res  (trainable)
    - W_main: top-r singular components
    - residual: remaining (rank - r) components
    """
    def __init__(self, in_features, out_features, r_keep_top: int, bias=True,
                 init_weight: torch.Tensor = None, init_bias: torch.Tensor = None, train_main: bool = False, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_keep_top = r_keep_top

        # frozen main weight
        self.weight_main = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype), requires_grad=train_main)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        # bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
            if init_bias is not None:
                self.bias.data.copy_(init_bias.to(device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # residual components (trainable params; 없으면 None)
        self.S_residual = None
        self.U_residual = None
        self.V_residual = None

    def compute_current_weight(self):
        if (self.S_residual is not None) and (self.U_residual is not None) and (self.V_residual is not None):
            US = self.U_residual * self.S_residual.unsqueeze(0)  # (out,k)
            residual_weight = US @ self.V_residual  # (out,in)  (V_residual: (k,in))
            return self.weight_main + residual_weight
        return self.weight_main

    @property
    def weight(self):
        # Swin 코드가 self.qkv.weight 를 직접 참조하므로 호환용
        return self.compute_current_weight()

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def compute_orthogonal_loss(self):
        # residual이 없으면 0
        if (self.U_residual is None) or (self.V_residual is None):
            return torch.zeros((), device=self.weight_main.device)

        U = self.U_residual  # (out, k)
        V = self.V_residual  # (k, in)  == Vh_res

        k = U.shape[1]
        device = U.device
        dtype = U.dtype

        Iu = torch.eye(k, device=device, dtype=dtype)
        UtU = U.transpose(0, 1) @ U
        loss_u = torch.norm(UtU - Iu, p='fro')

        Iv = torch.eye(V.shape[0], device=device, dtype=dtype)
        VVt = V @ V.transpose(0, 1)
        loss_v = torch.norm(VVt - Iv, p='fro')

        return 0.5 * (loss_u + loss_v)


def set_svd_bias_trainable(model: nn.Module, train_bias: bool):
    """
    freeze_others=False (전체 FT) 상황에서도 SVDResidualLinear.bias 학습 여부를 의도대로 강제.
    """
    for m in model.modules():
        if isinstance(m, SVDResidualLinear) and (m.bias is not None):
            m.bias.requires_grad = bool(train_bias)


def _linear_to_svd_residual(linear: nn.Linear, k_residual: int) -> SVDResidualLinear:
    """
    nn.Linear -> SVDResidualLinear
    k_residual: trainable residual singular component 개수
    """
    W = linear.weight.data
    device, dtype = W.device, W.dtype

    out_features, in_features = W.shape
    rank = min(out_features, in_features)

    # residual 개수 = k_residual (가능하면), keep_top = rank - k_residual
    k = min(k_residual, max(rank - 1, 0))  # 최소 1개는 keep_top 남기기
    r_keep_top = max(rank - k, 1)

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # main (frozen): top r_keep_top
    U_r = U[:, :r_keep_top]
    S_r = S[:r_keep_top]
    Vh_r = Vh[:r_keep_top, :]
    W_main = U_r @ torch.diag(S_r) @ Vh_r

    new = SVDResidualLinear(
        in_features=in_features,
        out_features=out_features,
        r_keep_top=r_keep_top,
        bias=(linear.bias is not None),
        init_weight=W_main,
        init_bias=(linear.bias.data.clone() if linear.bias is not None else None),
        train_main=False,  # W_r 은 학습하지 않고, 잔차인 ΔW 만 학습
        device=device, dtype=dtype,
    )

    # residual (trainable): remaining
    U_res = U[:, r_keep_top:]
    S_res = S[r_keep_top:]
    Vh_res = Vh[r_keep_top:, :]

    if S_res.numel() > 0:
        new.U_residual = nn.Parameter(U_res.clone().to(device=device, dtype=dtype))
        new.S_residual = nn.Parameter(S_res.clone().to(device=device, dtype=dtype))
        new.V_residual = nn.Parameter(Vh_res.clone().to(device=device, dtype=dtype))
    else:
        new.U_residual = None
        new.S_residual = None
        new.V_residual = None

    return new


def _freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_svd_residual(model: nn.Module, train_bias: bool = False):
    """
    SVDResidualLinear의 residual 파라미터만 학습 가능하게.
    train_bias=True면 SVDResidualLinear.bias도 풀어줌(보통은 False 추천).
    """
    for m in model.modules():
        if isinstance(m, SVDResidualLinear):
            if m.U_residual is not None:
                m.U_residual.requires_grad = True
            if m.S_residual is not None:
                m.S_residual.requires_grad = True
            if m.V_residual is not None:
                m.V_residual.requires_grad = True
            if train_bias and (m.bias is not None):
                m.bias.requires_grad = True


def _unfreeze_head(model: nn.Module, head_attr: str = "head"):
    """
    model.head (classifier) 학습 가능하게.
    """
    head = getattr(model, head_attr, None)
    if head is None:
        return
    for p in head.parameters():
        p.requires_grad = True


# ------------------------------------------------------------
# 마지막 stage: Attention(qkv, proj)만 SVDResidual로 교체
# ------------------------------------------------------------
def apply_effort_swin_last_stage_attn_only(model: nn.Module,
                                          k_residual: int = 8,
                                          freeze_others: bool = True,
                                          train_svd_bias: bool = False,
                                          head_attr: str = "head"):
    """
    SwinTransformerV2 기준:
      - model.layers[-1].blocks[*].attn.qkv, attn.proj 만 교체
      - 기본: 마지막 stage만
    """
    assert hasattr(model, "layers"), "model.layers 가 없는데 SwinV2 구조가 맞는지 확인 필요"
    last_layer = model.layers[-1]
    assert hasattr(last_layer, "blocks"), "model.layers[-1].blocks 가 없음"

    # 1) 교체
    for blk in last_layer.blocks:
        # WindowAttention: qkv / proj
        if hasattr(blk, "attn"):
            if hasattr(blk.attn, "qkv") and isinstance(blk.attn.qkv, nn.Linear):
                blk.attn.qkv = _linear_to_svd_residual(blk.attn.qkv, k_residual=k_residual)
            if hasattr(blk.attn, "proj") and isinstance(blk.attn.proj, nn.Linear):
                blk.attn.proj = _linear_to_svd_residual(blk.attn.proj, k_residual=k_residual)

    # 2) requires_grad 세팅
    if freeze_others:
        _freeze_all(model)
        _unfreeze_svd_residual(model, train_bias=train_svd_bias)
        _unfreeze_head(model, head_attr=head_attr)

    return model


# ---------------------------------------------------------------------
# 마지막 stage: Attention(qkv, proj) + MLP(fc1, fc2)까지 교체
# ---------------------------------------------------------------------
def apply_effort_swin_last_stage_attn_and_mlp(model: nn.Module,
                                              k_residual: int = 8,
                                              freeze_others: bool = True,
                                              train_svd_bias: bool = False,
                                              head_attr: str = "head"):
    """
    SwinTransformerV2 기준:
      - model.layers[-1].blocks[*].attn.qkv, attn.proj + mlp.fc1, mlp.fc2 교체
    """
    assert hasattr(model, "layers"), "model.layers 가 없는데 SwinV2 구조가 맞는지 확인 필요"
    last_layer = model.layers[-1]
    assert hasattr(last_layer, "blocks"), "model.layers[-1].blocks 가 없음"

    # 1) 교체
    for blk in last_layer.blocks:
        # Attention
        if hasattr(blk, "attn"):
            if hasattr(blk.attn, "qkv") and isinstance(blk.attn.qkv, nn.Linear):
                blk.attn.qkv = _linear_to_svd_residual(blk.attn.qkv, k_residual=k_residual)
            if hasattr(blk.attn, "proj") and isinstance(blk.attn.proj, nn.Linear):
                blk.attn.proj = _linear_to_svd_residual(blk.attn.proj, k_residual=k_residual)

        # MLP (SwinTransformerBlock.mlp.fc1 / fc2)
        if hasattr(blk, "mlp"):
            if hasattr(blk.mlp, "fc1") and isinstance(blk.mlp.fc1, nn.Linear):
                blk.mlp.fc1 = _linear_to_svd_residual(blk.mlp.fc1, k_residual=k_residual)
            if hasattr(blk.mlp, "fc2") and isinstance(blk.mlp.fc2, nn.Linear):
                blk.mlp.fc2 = _linear_to_svd_residual(blk.mlp.fc2, k_residual=k_residual)

    # 2) requires_grad 세팅
    if freeze_others:
        _freeze_all(model)
        _unfreeze_svd_residual(model, train_bias=train_svd_bias)
        _unfreeze_head(model, head_attr=head_attr)

    return model


def compute_effort_regularizers(model, lambda_orth=0.0, lambda_s=0.0):
    loss_orth = 0.0
    loss_s = 0.0
    n_orth = 0
    n_s = 0

    for m in model.modules():
        if isinstance(m, SVDResidualLinear) and getattr(m, "S_residual", None) is not None:
            if lambda_orth > 0:
                loss_orth = loss_orth + m.compute_orthogonal_loss()
                n_orth += 1
            if lambda_s > 0:
                loss_s = loss_s + m.S_residual.abs().mean()
                n_s += 1

    if n_orth > 0:
        loss_orth = loss_orth / n_orth
    if n_s > 0:
        loss_s = loss_s / n_s

    total = lambda_orth * loss_orth + lambda_s * loss_s
    return total, loss_orth, loss_s
