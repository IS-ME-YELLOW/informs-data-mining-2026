# ============================================================
# logger.py — 实验日志记录器
# ============================================================
# 职责: 将每次运行的配置、CV结果、基线对比、特征重要性、预测分布
#   以纯文本追加写入 logs/experiments.log
#
# 日志格式(每段一次运行):
#   RUN 时间戳 | FEATURE_VERSION | SEED
#   --- Config --- (超参数、样本数等)
#   [osi_target_t01h] 逐折RMSE/MAE + best_iter
#   [Feature Importance top 15] 特征名+gain值
#   [预测分布] min/max/mean/zero%
#   --- Baselines --- Zero/Mean/Persistence 对比表
#   --- Notes --- 手动备注
#
# 快速查询:
#   grep "RUN" logs/experiments.log     — 列出所有运行
#   grep "Mean RMSE" logs/experiments.log — 横向对比各次CV
# ============================================================

import os
import time
import numpy as np


class ExperimentLogger:
    """实验日志: 纯文本追加写, 段落式, 人读+grep友好"""

    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.path = log_path
        self.lines = []  # 先攒在内存, run_footer时一次性写入

    def _write(self, text):
        self.lines.append(text)

    def _flush(self):
        """一次性将积攒的所有内容追加写入文件"""
        with open(self.path, 'a') as f:
            f.write('\n'.join(self.lines) + '\n')
        self.lines = []

    def run_header(self, config_dict):
        """记录运行头: 时间戳、feature版本、seed、超参数摘要"""
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        sep = '=' * 80
        self._write(sep)
        self._write(f"RUN {ts} | FEATURE_VERSION={config_dict.get('feature_version','?')} "
                    f"| SEED={config_dict.get('seed','?')}")
        self._write(sep)
        self._write('')
        self._write('--- Config ---')
        for k, v in config_dict.items():
            self._write(f'  {k}: {v}')
        self._write('')

    def cv_results(self, horizon, fold_metrics, summary):
        """记录单个horizon的逐折指标+汇总"""
        self._write(f'[{horizon}]')
        for i, fm in enumerate(fold_metrics):
            self._write(f'  Fold {i+1}: RMSE={fm["rmse"]:.6f}, MAE={fm["mae"]:.6f}  '
                        f'(best_iter={fm.get("best_iter","?")})')
        self._write(f'  Mean RMSE={summary["rmse"]:.6f} ± {summary["rmse_std"]:.6f}, '
                    f'Mean MAE={summary["mae"]:.6f} ± {summary["mae_std"]:.6f}')
        self._write('')

    def baselines(self, baseline_metrics):
        """记录基线对比表: Zero/Mean/Persistence vs Model"""
        self._write('--- Baselines ---')
        horizons = list(baseline_metrics.keys())
        models = list(baseline_metrics[horizons[0]].keys())
        header = f'{"":20s} ' + ' '.join(f'{h:>10s}' for h in horizons)
        self._write(header)
        for m in models:
            vals = ' '.join(f'{baseline_metrics[h][m]["rmse"]:.6f}' for h in horizons)
            self._write(f'  {m:18s} {vals}')
        self._write('')

    def feature_importance(self, horizon, top_features):
        """记录top-N特征重要性(按gain排序)"""
        self._write(f'[Feature Importance top {len(top_features)} — {horizon}]')
        for i, (name, gain) in enumerate(top_features):
            self._write(f'  {i+1}. {name:30s} gain={gain:.1f}')
        self._write('')

    def prediction_summary(self, horizon, stats):
        """记录预测值分布: min/max/mean/zero%"""
        self._write(f'[{horizon}] min={stats["min"]:.4f}, max={stats["max"]:.4f}, '
                    f'mean={stats["mean"]:.4f}, zero%={stats["zero_pct"]:.1f}%')
        self._write('')

    def notes(self, text):
        """记录手动备注(观察、错误、下一步计划)"""
        self._write('--- Notes ---')
        self._write(text)
        self._write('')

    def run_footer(self):
        """记录运行尾: 分隔符 + 一次性flush到文件"""
        sep = '=' * 80
        self._write(sep)
        self._write('')
        self._flush()

    def log(self, text):
        """记录任意文本行"""
        self._write(text)
