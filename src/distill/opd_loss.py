"""Token-level distillation losses between student and teacher action-token distributions.

Logits have shape (B, 56, 256): 56 action tokens decoded in parallel, each a categorical over 256 bins.
Parallel decoding makes the chunk distribution a product over tokens, so KL terms are exact sums over
tokens and bins; no sampling is needed.
"""
import torch
import torch.nn.functional as F


def opd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mode: str = "rkl",
             bins: torch.Tensor = None, temperature: float = 1.0) -> torch.Tensor:
    """Mean per-token loss.

    rkl     exact reverse KL(student || teacher): the expectation of VLA-OPD's single-sample estimator
    rkl_pg  VLA-OPD Eq. 6-7 as published: REINFORCE on the token the student sampled (`bins`), with reward
            r = -(log pi_S(a) - log pi_T(a)) under stop-gradient
    fkl     forward KL(teacher || student)
    ce      cross-entropy to the teacher's argmax token (DAgger with the teacher's greedy action as label)
    """
    ls = F.log_softmax(student_logits.float() / temperature, dim=-1)
    lt = F.log_softmax(teacher_logits.float() / temperature, dim=-1).detach()
    if mode == "rkl":
        return (ls.exp() * (ls - lt)).sum(-1).mean()
    if mode == "fkl":
        return (lt.exp() * (lt - ls)).sum(-1).mean()
    if mode == "ce":
        return F.nll_loss(ls.flatten(0, 1), lt.argmax(-1).flatten())
    if mode == "rkl_pg":
        idx = bins.long().unsqueeze(-1)
        ls_a, lt_a = ls.gather(-1, idx).squeeze(-1), lt.gather(-1, idx).squeeze(-1)
        reward = -(ls_a - lt_a).detach()
        return -(ls_a * reward).mean()
    raise ValueError(mode)


@torch.no_grad()
def token_metrics(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> dict:
    ls = F.log_softmax(student_logits.float(), dim=-1)
    lt = F.log_softmax(teacher_logits.float(), dim=-1)
    return {
        "rkl": float((ls.exp() * (ls - lt)).sum(-1).mean()),
        "fkl": float((lt.exp() * (lt - ls)).sum(-1).mean()),
        "entropy_s": float(-(ls.exp() * ls).sum(-1).mean()),
        "entropy_t": float(-(lt.exp() * lt).sum(-1).mean()),
        "agree": float((ls.argmax(-1) == lt.argmax(-1)).float().mean()),
    }
