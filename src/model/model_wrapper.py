from dataclasses import dataclass
from pathlib import Path
import gc
import random
import time
from typing import Literal, Optional, Protocol, runtime_checkable, Any

import json
import moviepy.editor as mpy
import torch
import torchvision
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim
import torch.nn.functional as F

from loss.loss_lpips import LossLpips
from loss.loss_mse import LossMse
from model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri

from ..loss.loss_distill import DistillLoss
from src.utils.render import generate_path
from src.utils.point import get_normal_map

from ..loss.loss_huber import HuberLoss, extri_intri_to_pose_encoding


from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim, abs_relative_difference, delta1_acc
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_point import Regr3D
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose, rotation_6d_to_matrix
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .ply_export import export_ply

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    generate_video: bool
    mode: Literal["inference", "evaluation"]
    image_folder: str
    #  Global prune percentage used at test/val time.
    global_prune_percent: int


@dataclass
class TrainCfg:
    output_path: Path
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    pose_loss_alpha: float = 1.0
    pose_loss_delta: float = 1.0
    cxt_depth_weight: float = 0.01
    weight_pose: float = 1.0
    weight_depth: float = 1.0
    weight_normal: float = 1.0
    render_ba: bool = False
    render_ba_after_step: int = 0


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model: nn.Module
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        model: nn.Module,
        losses: list[Loss],
        step_tracker: StepTracker | None
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.global_prune_percent = self.test_cfg.global_prune_percent

        # Set up the model.
        self.encoder_visualizer = None
        self.model = model
        self.data_shim = get_data_shim(self.model.encoder)
        self.losses = nn.ModuleList(losses)
        
        if self.model.encoder.pred_pose:
            self.loss_pose = HuberLoss(alpha=self.train_cfg.pose_loss_alpha, delta=self.train_cfg.pose_loss_delta)
        
        if self.model.encoder.distill:
            self.loss_distill = DistillLoss(
                delta=self.train_cfg.pose_loss_delta,
                weight_pose=self.train_cfg.weight_pose,
                weight_depth=self.train_cfg.weight_depth,
                weight_normal=self.train_cfg.weight_normal
            )

        # This is used for testing.
        self.benchmarker = Benchmarker()
        
    def on_train_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.datamodule.train_loader.dataset, "set_epoch"):
            self.trainer.datamodule.train_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.train_loader.sampler, "set_epoch"):
            self.trainer.datamodule.train_loader.sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        print(f"Validation epoch start on rank {self.trainer.global_rank}")
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.datamodule.val_loader.dataset, "set_epoch"):
            self.trainer.datamodule.val_loader.dataset.set_epoch(self.current_epoch)
        if hasattr(self.trainer.datamodule.val_loader.sampler, "set_epoch"):
            self.trainer.datamodule.val_loader.sampler.set_epoch(self.current_epoch)
        
    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        # torch.cuda.empty_cache()
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined
        
        batch: BatchedExample = self.data_shim(batch)
        b, v, c, h, w = batch["context"]["image"].shape
        context_image = (batch["context"]["image"] + 1) / 2
        
        # Run the model.
        visualization_dump = None

        encoder_output, output = self.model(context_image, self.global_step, visualization_dump=visualization_dump)
        gaussians, pred_pose_enc_list, depth_dict = encoder_output.gaussians, encoder_output.pred_pose_enc_list, encoder_output.depth_dict
        pred_context_pose = encoder_output.pred_context_pose
        infos = encoder_output.infos
        distill_infos = encoder_output.distill_infos
        
        num_context_views = pred_context_pose['extrinsic'].shape[1]

        using_index = torch.arange(num_context_views, device=gaussians.means.device)
        batch["using_index"] = using_index
        
        target_gt = (batch["context"]["image"] + 1) / 2
        scene_scale = infos["scene_scale"]
        self.log("train/scene_scale", infos["scene_scale"])
        self.log("train/voxelize_ratio", infos["voxelize_ratio"])

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        consis_absrel = abs_relative_difference(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
            rearrange(distill_infos['conf_mask'], "b v h w -> (b v) h w"),
        )
        self.log("train/consis_absrel", consis_absrel.mean())

        consis_delta1 = delta1_acc(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
            rearrange(distill_infos['conf_mask'], "b v h w -> (b v) h w"),
        )
        self.log("train/consis_delta1", consis_delta1.mean())
        
        # Compute and log loss.
        total_loss = 0

        depth_dict['distill_infos'] = distill_infos
        with torch.amp.autocast('cuda', enabled=False):
            for loss_fn in self.losses:
                loss = loss_fn.forward(output, batch, gaussians, depth_dict, self.global_step)
                self.log(f"loss/{loss_fn.name}", loss)
                total_loss = total_loss + loss

            if depth_dict is not None and "depth" in get_cfg()["loss"].keys() and self.train_cfg.cxt_depth_weight > 0:
                depth_loss_idx = list(get_cfg()["loss"].keys()).index("depth")
                depth_loss_fn = self.losses[depth_loss_idx].ctx_depth_loss
                loss_depth = depth_loss_fn(depth_dict["depth_map"], depth_dict["depth_conf"], batch, cxt_depth_weight=self.train_cfg.cxt_depth_weight)
                self.log("loss/ctx_depth", loss_depth)
                total_loss = total_loss + loss_depth

            if distill_infos is not None:
                # distill ctx pred_pose & depth & normal
                loss_distill_list = self.loss_distill(distill_infos, pred_pose_enc_list, output, batch)
                self.log("loss/distill", loss_distill_list['loss_distill'])
                self.log("loss/distill_pose", loss_distill_list['loss_pose'])
                self.log("loss/distill_depth", loss_distill_list['loss_depth'])
                self.log("loss/distill_normal", loss_distill_list['loss_normal'])
                total_loss = total_loss + loss_distill_list['loss_distill']
        
        self.log("loss/total", total_loss)

        # Skip batch if loss is too high after certain step
        SKIP_AFTER_STEP = 1000  
        LOSS_THRESHOLD = 0.2
        if self.global_step > SKIP_AFTER_STEP and total_loss > LOSS_THRESHOLD:
            print(f"Skipping batch with high loss ({total_loss:.6f}) at step {self.global_step} on Rank {self.global_rank}")
            # set to a really small number
            return total_loss * 1e-10

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}; "
            )
            
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor
        
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        
        del batch
        if self.global_step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        return total_loss
    
    def on_after_backward(self):
        total_norm = 0.0
        counter = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                counter += 1
        total_norm = (total_norm / counter) ** 0.5
        self.log("loss/grad_norm", total_norm)
        
    #  Test step
    def test_step(self, batch, batch_idx):
        """Novel-view-synthesis evaluation on the test set.

        This is intentionally kept fully separate from ``validation_step``:
        it renders to disk (honouring the ``test_cfg`` save flags), computes
        PSNR/SSIM/LPIPS scores and times the encoder/decoder with the
        benchmarker, but performs no wandb image logging and no
        distillation-dependent depth-consistency metrics.

        Because the model is pose-free, Gaussians and camera poses are
        predicted from the context views and the target views (a subset of
        the context views) are rendered from those predicted poses.
        """
        batch: BatchedExample = self.data_shim(batch)

        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1, "test_step assumes batch size 1."
        if batch_idx % 100 == 0 and self.global_rank == 0:
            print(f"Test step {batch_idx:0>6}.")

        context_image = (batch["context"]["image"] + 1) / 2
        device = context_image.device
        near, far = 0.01, 100.0
        global_prune_percent = self.global_prune_percent

        # Predict Gaussians + camera poses from the context views.
        with self.benchmarker.time("encoder"):
            encoder_output = self.model.encoder(
                context_image,
                global_prune_percent,
                self.global_step,
                visualization_dump={},
            )
        gaussians = encoder_output.gaussians
        pred_context_pose = encoder_output.pred_context_pose

        # Render the context views from the predicted poses. (The model is
        # pose-free, so GT target extrinsics are in a different frame and are
        # not used for rendering; target views are matched below as a subset
        # of the context views.)
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.model.decoder.forward(
                gaussians,
                pred_context_pose["extrinsic"],
                pred_context_pose["intrinsic"],
                torch.ones(1, v, device=device) * near,
                torch.ones(1, v, device=device) * far,
                (h, w),
                "depth",
            )

        # Select the predicted views that correspond to the target views.
        context_indices = batch["context"]["index"][0]
        target_indices = batch["target"]["index"][0]
        match_idxs = []
        for t_idx in target_indices:
            matches = (context_indices == t_idx).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                match_idxs.append(matches[0])
        if len(match_idxs) == 0:
            print(f"Warning: no target views found in context views for scene "
                  f"{batch['scene']}. Skipping test step.")
            return
        match_idxs = torch.stack(match_idxs).to(device)

        rgb_pred = output.color[0][match_idxs].float()
        rgb_gt = batch["target"]["image"][0].float()

        # Guard against a mismatch in the number of matched views.
        if rgb_pred.shape[0] != rgb_gt.shape[0]:
            min_len = min(rgb_pred.shape[0], rgb_gt.shape[0])
            rgb_pred, rgb_gt = rgb_pred[:min_len], rgb_gt[:min_len]

        # Compute scores.
        if self.test_cfg.compute_scores:
            overlap = batch["context"].get("overlap", None)
            overlap_tag = get_overlap_tag(overlap[0]) if overlap is not None else None

            all_metrics = {
                "psnr_ours": compute_psnr(rgb_gt, rgb_pred).mean(),
                "ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                "lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
            }
            self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics, methods=["ours"], overlap_tag=overlap_tag)

        # Save results to disk.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], rgb_pred):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in rgb_pred],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        if self.test_cfg.save_compare:
            context_img = inverse_normalize(batch["context"]["image"][0])
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
            )
            save_image(comparison, path / f"{scene}.png")

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        visualization_dump = {}

        context_image = (batch["context"]["image"] + 1) / 2

        global_step = self.global_step
        visualization_dum = visualization_dump
        near = 0.01
        far = 100.0
    
        b, v, c, h, w = context_image.shape
        device = context_image.device

        global_prune_percent = self.global_prune_percent

        start_time = time.time()

        encoder_output = self.model.encoder(context_image, global_prune_percent, global_step, visualization_dump=visualization_dump)
        gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose
        output = self.model.decoder.forward(
            gaussians,
            pred_context_pose['extrinsic'],
            pred_context_pose["intrinsic"],
            torch.ones(1, v, device=device) * near,
            torch.ones(1, v, device=device) * far,
            (h, w),
            "depth",
        )

        gaussians, pred_pose_enc_list, depth_dict = encoder_output.gaussians, encoder_output.pred_pose_enc_list, encoder_output.depth_dict
        


        self.log(f"val/time_taken", time.time() - start_time)

        pred_context_pose, distill_infos = encoder_output.pred_context_pose, encoder_output.distill_infos
        infos = encoder_output.infos

        # Identify subset of context views that match target views
        context_indices = batch["context"]["index"][0]
        target_indices = batch["target"]["index"][0]
        
        match_idxs = []
        for t_idx in target_indices:
            # find index where context_indices == t_idx
            # this handles reordering if target is subset but shuffled (unlikely but safe)
            matches = (context_indices == t_idx).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                match_idxs.append(matches[0])
        
        if len(match_idxs) == 0:
            print(f"Warning: No target views found in context views for scene {batch['scene']}. Skipping metrics.")
            return

        match_idxs = torch.stack(match_idxs).to(self.device)
        
        # Filter predictions to match target set
        rgb_pred = output.color[0][match_idxs].float()
        depth_pred = vis_depth_map(output.depth[0][match_idxs])
        

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0].float()
        
        # Ensure shapes match before metric computation
        if rgb_pred.shape[0] != rgb_gt.shape[0]:
            print(f"Warning: Number of matched predicted views ({rgb_pred.shape[0]}) != target views ({rgb_gt.shape[0]}). Metric computation might fail.")
            # Trim to common length? Or fail?
            min_len = min(rgb_pred.shape[0], rgb_gt.shape[0])
            rgb_pred = rgb_pred[:min_len]
            rgb_gt = rgb_gt[:min_len]
            
        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        # Compute validation metrics.
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim)

        

        # depth metrics (consistency check on full context set)
        consis_absrel = abs_relative_difference(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
        )
        self.log("val/consis_absrel", consis_absrel.mean())
        
        consis_delta1 = delta1_acc(
            rearrange(output.depth, "b v h w -> (b v) h w"),
            rearrange(depth_dict['depth'].squeeze(-1), "b v h w -> (b v) h w"),
            valid_mask=rearrange(torch.ones_like(output.depth, device=output.depth.device, dtype=torch.bool), "b v h w -> (b v) h w"),
        )
        self.log("val/consis_delta1", consis_delta1.mean())

        diff_map = torch.abs(output.depth - depth_dict['depth'].squeeze(-1))
        self.log("val/consis_mse", diff_map[distill_infos['conf_mask']].mean())

        # Construct comparison image.
        # Filter all components to match target set for alignment
        
        # inverse_normalize assumes standard normalization
        context_img_full = inverse_normalize(batch["context"]["image"][0])
        context = [context_img_full[i] for i in range(len(context_img_full))]
        
        colored_diff_map = vis_depth_map(diff_map[0][match_idxs], near=torch.tensor(1e-4, device=diff_map.device), far=torch.tensor(1.0, device=diff_map.device))
        
        model_depth_pred = depth_dict["depth"].squeeze(-1)[0][match_idxs]
        model_depth_pred = vis_depth_map(model_depth_pred)
        
        # Normals also need filtering
        # get_normal_map usually returns (B*V, H, W, 3) or similar.
        # output.depth has shape (B, V, H, W) -> flatten to (B*V, H, W)
        # We need to compute normals on subset? Or compute on all then filter?
        # Compute on all first is safer for shapes?
        # Actually get_normal_map takes (B*V, H, W) depth and (B*V, 3, 3) intrinsics.
        # We can filter inputs to get_normal_map
        
        subset_depth = output.depth.flatten(0, 1)[match_idxs] # (V_subset, H, W) where V_subset = len(match_idxs) since B=1
        subset_intrinsics = batch["context"]["intrinsics"].flatten(0, 1)[match_idxs]
        render_normal = (get_normal_map(subset_depth, subset_intrinsics).permute(0, 3, 1, 2) + 1) / 2.
        
        subset_model_depth = depth_dict['depth'].flatten(0, 1).squeeze(-1)[match_idxs]
        pred_normal = (get_normal_map(subset_model_depth, subset_intrinsics).permute(0, 3, 1, 2) + 1) / 2.


        save_list = [vcat(*context), vcat(*rgb_gt), vcat(*rgb_pred)]

        for l, file_name in zip(save_list, ["Context", "Target (Ground Truth)", "Target (Prediction)"]):
            comparison = l

            comparison = torch.nn.functional.interpolate(
                comparison.unsqueeze(0), 
                scale_factor=0.5, 
                mode='bicubic', 
                align_corners=False
            ).squeeze(0)
            
            self.logger.log_image(
                file_name,
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )


        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)
        

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        encoder_output = self.model.encoder((batch["context"]["image"]+1)/2, self.global_step)
        gaussians, pred_pose_enc_list = encoder_output.gaussians, encoder_output.pred_pose_enc_list

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.model.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]


    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if "gaussian_param_head" in name or "interm" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)
        
        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
