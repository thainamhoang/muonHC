"""Simple training loop for downscaling models."""

import os
import torch
import torch.nn as nn

from utils.metrics import log_frequency_distance


def _resolve_amp_dtype(dtype_name):
    dtype_name = str(dtype_name).lower()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {dtype_name}")


def _build_grad_scaler(enabled):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader=None,
                 optimizer=None, scheduler=None, device='cuda',
                 max_epochs=50, patience=10, save_dir='checkpoints',
                 spectral_lambda=0.0, target_mean=278.45, target_std=21.25,
                 wandb_run=None, wandb_run_id=None, scheduler_step_by='epoch',
                 log_interval=50, resume_path=None, grad_accum_steps=1,
                 amp_enabled=False, amp_dtype='bfloat16', loss_cfg=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.save_dir = save_dir
        self.spectral_lambda = spectral_lambda
        self.loss_cfg = dict(loss_cfg or {})
        if self.spectral_lambda > 0 and "spectral_lambda" not in self.loss_cfg:
            self.loss_cfg["spectral_lambda"] = self.spectral_lambda
        self.target_mean = target_mean
        self.target_std = target_std
        self.wandb_run = wandb_run
        self.wandb_run_id = wandb_run_id
        self.scheduler_step_by = scheduler_step_by
        self.log_interval = log_interval
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.device_type = str(device).split(':', maxsplit=1)[0]
        self.amp_enabled = bool(amp_enabled and self.device_type == 'cuda')
        self.amp_dtype = _resolve_amp_dtype(amp_dtype)
        self.grad_scaler = _build_grad_scaler(
            enabled=self.amp_enabled and self.amp_dtype is torch.float16
        )
        self.start_epoch = 1

        os.makedirs(save_dir, exist_ok=True)
        self.best_val_rmse = float('inf')
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.train_losses = []
        self.val_losses = []
        self.val_rmses_k = []
        self.test_metrics = {}
        if resume_path is not None:
            self.load_checkpoint(resume_path)
        if self.wandb_run is not None:
            self.wandb_run.summary['config/spectral_lambda'] = float(
                self.loss_cfg.get("spectral_lambda", self.spectral_lambda)
            )
            if "laplacian_lambda" in self.loss_cfg:
                self.wandb_run.summary['config/loss/laplacian_lambda'] = float(
                    self.loss_cfg["laplacian_lambda"]
                )
            self.wandb_run.summary['config/grad_accum_steps'] = int(self.grad_accum_steps)
            self.wandb_run.summary['config/effective_batch_size'] = (
                int(self.train_loader.batch_size) * int(self.grad_accum_steps)
            )
            self.wandb_run.summary['config/amp_enabled'] = self.amp_enabled
            self.wandb_run.summary['config/amp_dtype'] = str(self.amp_dtype).replace('torch.', '')

    def autocast(self):
        return torch.autocast(
            device_type=self.device_type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        )

    @property
    def latest_checkpoint_path(self):
        return os.path.join(self.save_dir, 'latest_checkpoint.pt')

    @property
    def best_checkpoint_path(self):
        return os.path.join(self.save_dir, 'best_checkpoint.pt')

    @property
    def best_model_path(self):
        return os.path.join(self.save_dir, 'best_model.pt')

    def checkpoint_state(self, epoch):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'best_val_rmse': self.best_val_rmse,
            'best_epoch': self.best_epoch,
            'epochs_no_improve': self.epochs_no_improve,
            'wandb_run_id': self.wandb_run_id,
        }
        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        return state

    def save_checkpoint(self, path, epoch):
        torch.save(self.checkpoint_state(epoch), path)

    def current_lr(self):
        return float(self.optimizer.param_groups[0]['lr'])

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        if 'model_state' in ckpt:
            self.model.load_state_dict(ckpt['model_state'])
        else:
            self.model.load_state_dict(ckpt)
        if self.optimizer is not None and 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler is not None and 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.best_val_rmse = ckpt.get('best_val_rmse', self.best_val_rmse)
        self.best_epoch = ckpt.get('best_epoch', self.best_epoch)
        self.epochs_no_improve = ckpt.get('epochs_no_improve', self.epochs_no_improve)
        self.wandb_run_id = ckpt.get('wandb_run_id', self.wandb_run_id)
        self.start_epoch = ckpt.get('epoch', 0) + 1
        print(
            f"Resumed from epoch {self.start_epoch - 1} | "
            f"best_val_rmse={self.best_val_rmse:.4f} K"
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        n_batches = len(self.train_loader)
        print(
            f"Starting epoch {epoch}/{self.max_epochs} "
            f"({n_batches} train batches, grad_accum_steps={self.grad_accum_steps})",
            flush=True,
        )
        self.optimizer.zero_grad(set_to_none=True)
        seen_samples = 0
        for batch_idx, (lr, hr) in enumerate(self.train_loader):
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            seen_samples += lr.size(0)
            with self.autocast():
                pred = self.model(lr)

            # Keep FFT/MSE loss math in fp32 even when model forward uses AMP.
            pred_loss = pred.float()
            hr_loss = hr.float()
            mse = nn.functional.mse_loss(pred_loss, hr_loss)
            from losses.spectral_loss import mse_spectral_laplacian_loss
            loss = mse_spectral_laplacian_loss(
                pred_loss,
                hr_loss,
                loss_cfg=self.loss_cfg,
                mse=mse,
            )
            scaled_loss = loss / self.grad_accum_steps
            if self.grad_scaler.is_enabled():
                self.grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (
                (batch_idx + 1) % self.grad_accum_steps == 0
                or batch_idx + 1 == n_batches
            )
            if should_step:
                if self.grad_scaler.is_enabled():
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None and self.scheduler_step_by == 'step':
                    self.scheduler.step()

            total_loss += loss.detach() * lr.size(0)
            if (
                self.log_interval > 0
                and ((batch_idx + 1) % self.log_interval == 0 or batch_idx + 1 == n_batches)
            ):
                avg_loss = (total_loss / seen_samples).item()
                print(
                    f"  batch {batch_idx + 1:4d}/{n_batches} | "
                    f"loss {avg_loss:.6f} | lr {self.current_lr():.3e}",
                    flush=True,
                )
        return (total_loss / len(self.train_loader.dataset)).item()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = torch.zeros((), device=self.device)
        sum_sq_z = torch.zeros((), device=self.device)
        n_values = 0
        for lr, hr in self.val_loader:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            with self.autocast():
                pred = self.model(lr)
            pred = pred.float()
            hr = hr.float()
            loss = nn.functional.mse_loss(pred, hr)
            total_loss += loss * lr.size(0)
            sum_sq_z += ((pred - hr) ** 2).sum()
            n_values += hr.numel()
        avg_loss = (total_loss / len(self.val_loader.dataset)).item()
        val_rmse_z = ((sum_sq_z / n_values) ** 0.5).item()
        val_rmse_k = val_rmse_z * self.target_std
        return avg_loss, val_rmse_k, val_rmse_z

    def train(self):
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_rmse_k, val_rmse_z = self.validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_rmses_k.append(val_rmse_k)

            print(f"Epoch {epoch:3d}/{self.max_epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Val RMSE (K): {val_rmse_k:.4f} | Val RMSE (z): {val_rmse_z:.6f}")

            # Scheduler step
            if self.scheduler is not None and self.scheduler_step_by == 'epoch':
                self.scheduler.step()
            current_lr = self.current_lr()

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        'epoch': epoch,
                        'lr': current_lr,
                        'train/loss': float(train_loss),
                        'val/loss': float(val_loss),
                        'val/rmse_k': float(val_rmse_k),
                        'val/rmse_z': float(val_rmse_z),
                        'best/val_rmse_k': float(min(self.best_val_rmse, val_rmse_k)),
                    },
                    step=epoch,
                    commit=True,
                )

            # Early stopping
            if val_rmse_k < self.best_val_rmse:
                self.best_val_rmse = val_rmse_k
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                # Save best model
                torch.save(self.model.state_dict(), self.best_model_path)
                self.save_checkpoint(self.best_checkpoint_path, epoch)
                print(f"  --> New best model (RMSE {val_rmse_k:.4f} K)")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                    break

            self.save_checkpoint(self.latest_checkpoint_path, epoch)

        # Load best model
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        print(f"Training finished. Best epoch: {self.best_epoch}, Best Val RMSE: {self.best_val_rmse:.4f} K")
        if self.wandb_run is not None:
            self.wandb_run.summary['best/epoch'] = self.best_epoch
            self.wandb_run.summary['best/val_rmse_k'] = float(self.best_val_rmse)
        # Optional test evaluation
        if self.test_loader is not None:
            test_metrics = self.test()
            print(
                f"Test RMSE: {test_metrics['rmse_k']:.4f} K | "
                f"Test LFD: {test_metrics['lfd']:.4f}"
            )
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        'test/rmse_k': float(test_metrics['rmse_k']),
                        'test/rmse_z': float(test_metrics['rmse_z']),
                        'test/lfd': float(test_metrics['lfd']),
                    },
                    commit=True,
                )
                for name, value in test_metrics.items():
                    self.wandb_run.summary[f'test/{name}'] = float(value)
        return self.best_val_rmse

    @torch.no_grad()
    def test(self):
        self.model.eval()
        sum_sq_z = torch.zeros((), device=self.device)
        lfd_sum = torch.zeros((), device=self.device)
        n_values = 0
        n_samples = 0
        target_mean, target_std = self._test_target_stats()
        for lr, hr in self.test_loader:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            with self.autocast():
                pred = self.model(lr)
            pred = pred.float()
            hr = hr.float()
            sum_sq_z += ((pred - hr) ** 2).sum()
            n_values += hr.numel()
            lfd_sum += log_frequency_distance(
                pred,
                hr,
                mean=target_mean,
                std=target_std,
                crop_border=1,
                reduction="sum",
            )
            n_samples += hr.size(0)

        rmse_z = ((sum_sq_z / n_values) ** 0.5).item()
        metrics = {
            'rmse_k': rmse_z * self.target_std,
            'rmse_z': rmse_z,
            'lfd': (lfd_sum / n_samples).item(),
        }
        self.test_metrics = metrics
        return metrics

    def _test_target_stats(self):
        dataset = getattr(self.test_loader, 'dataset', None)
        target_mean = getattr(dataset, 'hr_mean', self.target_mean)
        target_std = getattr(dataset, 'hr_std', self.target_std)
        return target_mean, target_std
