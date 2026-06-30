# ADR-003: 提交版纯 ZeroBP 与研究分支隔离

- **Status**: Accepted (2026-06-30) — LOCKED
- **Context**: Phase E/F 引入了少量 BP 的研究脚本与 flag。竞赛要求**零全局梯度**。必须保证官方提交**绝不含 BP**，且不被研究改动污染。另发现提交 kernel 曾内嵌 **pre-D-1 旧快照**（word-level，非 bpe+mlp）。
- **Decision**:
  - **官方提交 = `kaggle_run/` kernel = 纯 ZeroBP** `kaggle_zerograd_moe.py` 默认路径（bpe+mlp+Phase C），**不设任何 `ZG_*` env**。提交文件内**零** `.backward()/autograd.grad/enable_grad`。
  - **所有 BP 研究**（Phase E/F）放**独立脚本**（`phase_e*.py`/`phasee_nli_4b.py`/`h1_attn.py`/`f1_data.py`/`f2_aux.py`）+ **独立 kernel**（`kaggle_c1`/`adapt`/`phase_e`/`phasee_nli`）+ **默认-off flag**，提交从不导入。
  - 修复（commit `ef7f213`）：`build_kaggle_kernels.py` 从当前代码重生成 `kaggle_run.ipynb`（单一真源）。完整性 checklist 见 `SUBMISSION.md`。
- **Consequences**: 提交可证明纯 ZeroBP（run() 断言零 autograd + 7 门）。**维护规则（LOCKED）**：改 `kaggle_zerograd_moe.py` 必须①保持默认零 autograd / `6.251` 不变（研究 flag 默认 no-op）②重生成提交 notebook ③过 `SUBMISSION.md` 完整性表。
