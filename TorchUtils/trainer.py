import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

class CNN_trainer:
    """
    Binary classification trainer (single-logit head) for BCEWithLogitsLoss.

    Key behavior:
    - Can use a FIXED threshold or SEARCH the best threshold on the validation set.
    - Selects the best checkpoint by **Val Macro-F1** evaluated at the current threshold.
    - Early stopping by patience on Val Macro-F1.
    """

    def __init__(
        self, net, device, criterion, optimizer, scheduler = None, patience = 10, threshold_mode = "fixed",
        fixed_threshold = 0.50, threshold_grid = None, selection_metric = "macro_f1"
    ):
        self.net = net
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.equal_tol = 1e-8

        self.selection_metric = selection_metric.lower()
        self.maximize = self.selection_metric not in {"val_loss"}

        self.best_score = -np.inf if self.maximize else np.inf
        self.best_epoch_index = 0
        self.epochs_no_improve = 0

        self.threshold_mode = threshold_mode
        self.fixed_threshold = float(fixed_threshold)
        
        if threshold_grid is None:
            self.threshold_grid = [i / 100.0 for i in range(30, 71)]
        else:
            self.threshold_grid = list(threshold_grid)
        self.best_threshold = self.fixed_threshold

        self.main_path = ""

    @staticmethod
    def _to_probs(logits: torch.Tensor) -> torch.Tensor:

        return torch.sigmoid(logits.float().view(-1))

    @staticmethod
    def _metrics_at_threshold(y_true_np, y_prob_np, thr=0.5):
        y_pred = (y_prob_np >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred, average = None, labels = [0, 1]
        )
        acc = accuracy_score(y_true_np, y_pred)
        macro_f1 = float((f1[0] + f1[1]) / 2.0)

        try:
            roc = roc_auc_score(y_true_np, y_prob_np)
        except ValueError:
            roc = float("nan")

        pr_auc = average_precision_score(y_true_np, y_prob_np)

        return {
            "precision_0": prec[0], "recall_0": rec[0], "f1_0": f1[0],
            "precision_1": prec[1], "recall_1": rec[1], "f1_1": f1[1],
            "accuracy": acc, "macro_f1": macro_f1,
            "roc_auc": roc, "pr_auc": pr_auc, "threshold": thr
        }

    @staticmethod
    def _find_best_threshold(y_true_np, y_prob_np, grid):
        best_t, best_m = None, -np.inf
        best_metrics = None

        for t in grid:
            m = CNN_trainer._metrics_at_threshold(y_true_np, y_prob_np, thr = t)
            if m["macro_f1"] > best_m:
                best_m = m["macro_f1"]
                best_t = t
                best_metrics = m

        return best_t, best_metrics
    
    def _pick_score(self, val_loss, val_metrics):
        if self.selection_metric == "val_loss":
            return float(val_loss)
        
        v = val_metrics.get(self.selection_metric, None)

        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):

            return val_metrics["macro_f1"]
        
        return float(v)
    
    # def _is_improvement(self, score):

    #     return (score > self.best_score) if self.maximize else (score < self.best_score)

    def _is_improvement(self, score, val_loss):
        """
        Primary: better selection_metric
        Tie: lower validation loss
        Returns: (is_better: bool, reason: str)
        """
        if self.maximize:
            if score > self.best_score + self.equal_tol:
                return True, "primary metric improved"
            if abs(score - self.best_score) <= self.equal_tol and val_loss < self.best_val_loss - self.equal_tol:
                return True, "tie on primary metric, lower validation loss"
            return False, ""
        else:
            if score < self.best_score - self.equal_tol:
                return True, "primary metric improved (lower is better)"
            if abs(score - self.best_score) <= self.equal_tol and val_loss < self.best_val_loss - self.equal_tol:
                return True, "tie on primary metric, lower validation loss"
            return False, ""

    def run_process(self, train_loader, val_loader, main_path):
        self.main_path = main_path

        train_accuracies, val_accuracies = [], []
        train_losses, val_losses = [], []
        train_macro_f1_hist, val_macro_f1_hist = [], []

        train_ds = getattr(train_loader, 'dataset', None)

        for epoch in range(200):
            if hasattr(train_ds, "set_epoch"):
                train_ds.set_epoch(epoch)

            # if self.aug_logger is not None:
            #     self.aug_logger.reset()

            self.net.train()
            total_loss = 0.0
            all_probs, all_labels = [], []

            for batch in train_loader:
                x = batch["image"].to(self.device)
                y = batch["label"].to(self.device).view(-1).float()

                logits = self.net(x)
                loss = self.criterion(logits.view(-1), y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                all_probs.append(self._to_probs(logits).detach().cpu())
                all_labels.append(y.detach().cpu())

            train_loss = total_loss / max(1, len(train_loader))
            train_probs = torch.cat(all_probs).numpy()
            train_labels = torch.cat(all_labels).int().numpy()

            train_metrics_fixed = self._metrics_at_threshold(train_labels, train_probs, thr = 0.50)
            train_losses.append(train_loss)
            train_accuracies.append(round(train_metrics_fixed["accuracy"] * 100, 2))

            self.net.eval()
            v_total_loss = 0.0
            v_probs, v_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["image"].to(self.device)
                    y = batch["label"].to(self.device).view(-1).float()

                    logits = self.net(x)
                    loss = self.criterion(logits.view(-1), y)

                    v_total_loss += loss.item()
                    v_probs.append(self._to_probs(logits).cpu())
                    v_labels.append(y.cpu())

            val_loss = v_total_loss / max(1, len(val_loader))
            val_probs = torch.cat(v_probs).numpy()
            val_labels = torch.cat(v_labels).int().numpy()

            if self.threshold_mode == "search":
                thr_current, val_metrics = self._find_best_threshold(val_labels, val_probs, self.threshold_grid)
            else:
                thr_current = self.fixed_threshold
                val_metrics = self._metrics_at_threshold(val_labels, val_probs, thr = thr_current)

            train_metrics = self._metrics_at_threshold(train_labels, train_probs, thr = thr_current)

            print(f"Epoch {epoch+1} | thr={thr_current:.2f}")
            print(f"  Train:  loss={train_loss:.4f}  acc@thr={train_metrics['accuracy']*100:.2f}%  macroF1={train_metrics['macro_f1']:.4f}")
            print(f"  Valid:  loss={val_loss:.4f}    acc@thr={val_metrics['accuracy']*100:.2f}%  macroF1={val_metrics['macro_f1']:.4f}  "
                  f"ROC-AUC={val_metrics['roc_auc']:.4f}  PR-AUC={val_metrics['pr_auc']:.4f}")

            val_losses.append(val_loss)
            val_accuracies.append(round(val_metrics["accuracy"] * 100, 2))
            train_macro_f1_hist.append(train_metrics["macro_f1"])
            val_macro_f1_hist.append(val_metrics["macro_f1"])

            sel_score = self._pick_score(val_loss, val_metrics)
            if self.scheduler is not None:
                try:
                    self.scheduler.step(sel_score)
                except TypeError:
                    self.scheduler.step()

            # if self._is_improvement(sel_score):
            improved, reason = self._is_improvement(sel_score, val_loss)
            if improved:
                self.best_score = sel_score
                self.best_val_loss = float(val_loss)
                self.best_epoch_index = epoch
                self.epochs_no_improve = 0
                self.best_threshold = thr_current

                # print(f"  ↳ New best @ epoch {epoch+1}: {self.selection_metric}={self.best_score:.4f} @ thr={self.best_threshold:.2f}")
                print(
                    f"  ↳ New best @ epoch {epoch+1}: "
                    f"{self.selection_metric}={self.best_score:.4f}, "
                    f"val_loss={self.best_val_loss:.4f} "
                    f"({reason}) @ thr={self.best_threshold:.2f}"
                )

                torch.save(self.net.state_dict(), f"{self.main_path}_BEST_ITERATION.pth")
                # with open(f"{self.main_path}_BEST_THRESHOLD.json", "w") as f:
                #     json.dump({"threshold": self.best_threshold, "save_metric": self.selection_metric}, f)
                with open(f"{self.main_path}_BEST_THRESHOLD.json", "w") as f:
                    json.dump(
                        {"threshold": self.best_threshold, "save_metric": self.selection_metric,
                        "best_score": float(self.best_score), "best_val_loss": float(self.best_val_loss)}, f)

                self.plot_confusion_matrix(
                    val_labels.tolist(),
                    (val_probs >= self.best_threshold).astype(int).tolist(),
                    f"{self.main_path}_val_Confusion_matrix.png"
                )
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch+1}: no improvement in {self.selection_metric} for {self.patience} epochs.")
                break

        # self.epoch_plot(
        #     train_accuracies[: self.best_epoch_index + 1],
        #     val_accuracies[: self.best_epoch_index + 1],
        #     f"{self.main_path}_Accuracy.png",
        # )

        # self.epoch_plot(
        #     train_losses[: self.best_epoch_index + 1],
        #     val_losses[: self.best_epoch_index + 1],
        #     f"{self.main_path}_Loss.png",
        # )

        # self.epoch_plot(
        #     [m * 100 for m in train_macro_f1_hist[: self.best_epoch_index + 1]],
        #     [m * 100 for m in val_macro_f1_hist[: self.best_epoch_index + 1]],
        #     f"{self.main_path}_MacroF1.png",
        #     ylabel = "Macro-F1 (%)",
        #     ylim = (0, 100),
        # )

        total_epochs_ran = len(train_losses)
        last_k = 10
        last_start = max(0, total_epochs_ran - last_k)

        epoch_idxs = sorted(set(
            list(range(0, self.best_epoch_index + 1)) +
            list(range(last_start, total_epochs_ran))
        ))

        self.epoch_plot(
            train_accuracies,
            val_accuracies,
            f"{self.main_path}_Accuracy.png",
            epoch_idxs=epoch_idxs,
            ylabel="Accuracy (%)",
            ylim=(0, 100),
        )

        self.epoch_plot(
            train_losses,
            val_losses,
            f"{self.main_path}_Loss.png",
            epoch_idxs=epoch_idxs,
            ylabel="Loss",
        )

        self.epoch_plot(
            [m * 100 for m in train_macro_f1_hist],
            [m * 100 for m in val_macro_f1_hist],
            f"{self.main_path}_MacroF1.png",
            epoch_idxs=epoch_idxs,
            ylabel="Macro-F1 (%)",
            ylim=(0, 100),
        )

    def epoch_plot(self, train_values, validation_values, path, ylabel=None, ylim=None, epoch_idxs=None):
        plot_name = path.split("_")[-1].replace(".png", "")

        # If not provided, plot everything
        if epoch_idxs is None:
            epoch_idxs = list(range(len(train_values)))

        # Convert to 1-based epoch numbers for display
        x = [i + 1 for i in epoch_idxs]
        y_train = [train_values[i] for i in epoch_idxs]
        y_val = [validation_values[i] for i in epoch_idxs]

        plt.figure(figsize=(10, 6))
        plt.plot(x, y_train, label="train")
        plt.plot(x, y_val, label="validation")
        plt.title(f"Cross-Validation {plot_name}")
        plt.xlabel("Epochs")
        plt.ylabel(ylabel or plot_name)

        if ylim is not None:
            plt.ylim(*ylim)

        plt.legend()
        plt.grid(True)
        plt.savefig(path)
        plt.close()

    # def epoch_plot(self, train_values, validation_values, path, ylabel = None, ylim = None):
    #     plot_name = path.split("_")[-1].replace(".png", "")
    #     plt.figure(figsize = (10, 6))
    #     plt.plot(range(1, len(train_values) + 1), train_values, label="train")
    #     plt.plot(range(1, len(validation_values) + 1), validation_values, label="validation")
    #     plt.title(f"Cross-Validation {plot_name}")
    #     plt.xlabel("Epochs")
    #     plt.ylabel(ylabel or plot_name)

    #     if ylim is not None:
    #         plt.ylim(*ylim)
    #     elif plot_name == "Accuracy":
    #         plt.ylim(0, 100)

    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(path)
    #     plt.close()

    def plot_confusion_matrix(self, all_labels, all_preds, path):
        cm = confusion_matrix(all_labels, all_preds)
        cm_percent = cm.astype("float") / cm.sum(axis = 1, keepdims = True) * 100.0
        disp = ConfusionMatrixDisplay(confusion_matrix = cm_percent, display_labels=[0, 1])
        fig, ax = plt.subplots(figsize = (6, 5))
        disp.plot(cmap = "Blues", ax = ax, values_format = ".2f")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i - 0.2, f"({cm[i, j]})", ha = "center", va = "center", color = "black")

        plt.title("Confusion Matrix (Percentages)")
        plt.savefig(path)
        plt.close()
