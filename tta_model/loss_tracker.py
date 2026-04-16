import os, csv, json, time
from collections import deque

def _is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

class LossTracker:
    def __init__(self, adapt_method: str, logger, window_size: int = 50, log_every: int = 3):
        self.adapt_method = adapt_method
        self.logger = logger
        self.window_size = int(window_size)
        self.log_every = int(log_every)
        
        self.records = []
        self.loss_keys = None
        self.primary_key = None
        self._win = deque(maxlen=self.window_size)
        
        self.ctx = {
            "dataset_name": None,
            "base_model": None,
            "retrieval_type": None,
            "perturbation": None,
            "severity": None,
            "exp_dir": None,
            "log_dir": None,
        }
        
    def set_run_context(self, *, dataset_name, base_model, retrieval_type,
                        perturbation, severity, output_dir, total_inter):
        exp_dir = "_".join([dataset_name.lower(), base_model.lower(), retrieval_type.lower()])
        log_dir = os.path.join(output_dir, exp_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.ctx.update(dict(
            dataset_name=dataset_name,
            base_model=base_model,
            retrieval_type=retrieval_type,
            perturbation=perturbation,
            severity=severity,
            total_inter=total_inter,
            exp_dir=exp_dir,
            log_dir=log_dir
        ))
        self._win.clear()
        self.loss_keys = None
        self.primary_key = None
        self._current_run_start_idx = len(self.records)
        
    def _init_loss_keys_if_needed(self, loss_dict: dict):
        if self.loss_keys is not None:
            return
        numeric_items = {k: v for k, v in loss_dict.items() if _is_number(v)}
        if not numeric_items:
            self.loss_keys = []
            self.primary_key = None
            return
        
        keys = list(numeric_items.keys())
        keys.sort()
        if 'loss_total' in keys:
            keys.remove('loss_total')
            keys = ['loss_total'] + keys
        self.loss_keys = keys
        
        self.primary_key = 'loss_total' if 'loss_total' in keys else keys[0]
        
    def add(self, *, iter_idx: int, step_idx: int, steps_per_iter: int, loss_dict: dict):
        self._init_loss_keys_if_needed(loss_dict)
        
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        rec = {
            "time": now,
            "adapt_method": self.adapt_method,
            "dataset_name": self.ctx["dataset_name"],
            "base_model": self.ctx["base_model"],
            "retrieval_type": self.ctx["retrieval_type"],
            "perturbation": self.ctx["perturbation"],
            "severity": self.ctx["severity"],
            "iter": int(iter_idx),
            "step": int(step_idx),
            "steps_per_iter": int(steps_per_iter),
        }
        
        for k in (self.loss_keys or []):
            v = loss_dict.get(k, float("nan"))
            try:
                v = float(v)
            except Exception:
                v = float("nan")
            rec[k] = v

        self.records.append(rec)
        
        v = rec.get(self.primary_key, None)
        if v is not None and _is_number(v):
            self._win.append(float(v))
        
        if step_idx == steps_per_iter - 1 and self.primary_key is not None:
            if (iter_idx + 1) % self.log_every == 0 and len(self._win) > 0:
                current_run_records = self.records[self._current_run_start_idx:]
                recent_records = []
                for r in reversed(current_run_records):
                    recent_records.append(r)
                    if len(recent_records) >= self.window_size:
                        break
                if recent_records:
                    loss_info = []
                    for k in self.loss_keys:
                        if k != self.primary_key:
                            values = [r[k] for r in recent_records if k in r and _is_number(r[k])]
                            if values:
                                mean_val = sum(values) / len(values)
                                loss_info.append(f"{k}={mean_val:.4f}")
                mean_win = sum(self._win) / len(self._win)
                if loss_info:
                    loss_str = ", ".join(loss_info)
                    self.logger.info(
                        f"{self.ctx['perturbation']}(sev={self.ctx['severity']}) "
                        f"iter={iter_idx+1}/{self.ctx['total_inter']} "
                        f"{self.primary_key}={mean_win:.6f} "
                        f"{loss_str}"
                    )
                else:
                    self.logger.info(
                        f"{self.ctx['perturbation']}(sev={self.ctx['severity']}) "
                        f"iter={iter_idx+1}/{self.ctx['total_inter']} "
                        f"{self.primary_key}={mean_win:.6f} "
                    )
                    
    def dump_csv(self, filename: str = None):
        log_dir = self.ctx["log_dir"] or "."
        if filename is None:
            filename = f"{self.adapt_method}_losses.csv"
        fpath = os.path.join(log_dir, filename)

        base_fields = [
            "time", "adapt_method", "dataset_name", "base_model", "retrieval_type",
            "perturbation", "severity", "iter", "step", "steps_per_iter",
        ]
        loss_fields = self.loss_keys or []
        fieldnames = base_fields + loss_fields

        write_header = not os.path.exists(fpath)
        with open(fpath, "a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for r in self.records:
                row = {k: r.get(k, float("nan")) for k in fieldnames}
                writer.writerow(row)
        return fpath

    def dump_jsonl(self, filename: str = None):
        log_dir = self.ctx["log_dir"] or "."
        if filename is None:
            filename = f"{self.adapt_method}_losses.jsonl"
        fpath = os.path.join(log_dir, filename)
        with open(fpath, "a", encoding="utf-8") as fp:
            for r in self.records:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        return fpath