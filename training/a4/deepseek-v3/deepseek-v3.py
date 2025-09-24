"""Nemo2 pretraining recipe for Deepseek v3 model."""

from nemo.collections import llm
from nemo.collections.llm.recipes import deepseek_v3
from nemo.lightning.pytorch.callbacks import NsysCallback
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
import nemo_run as run
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
import os



def recipe(
    profile_enabled: bool = False,
    profile_start_step: int = 0,
    profile_end_step: int = 0,
    profile_ranks: str = "0",
) -> run.Partial:
  """Returns a Nemo2 training recipe for Deepseek v3 model.

  Args:
      profile_enabled: Whether to enable Nsys profiling.
      profile_start_step: The step to start profiling.
      profile_end_step: The step to end profiling.
      profile_ranks: The ranks to profile, comma separated.

  Returns:
      A Nemo2 training recipe.
  """
  print("LOCAL_RANK: ", os.environ["LOCAL_RANK"])
  local_rank=os.environ['LOCAL_RANK']
  os.environ['NVSHMEM_ENABLE_NIC_PE_MAPPING'] = '1'
  os.environ['NVSHMEM_HCA_LIST'] = f'mlx5_{local_rank}:1'

  # Start from the Nemo standard recipe.
  pretrain = deepseek_v3.pretrain_recipe(
      num_nodes=1, num_gpus_per_node=8, performance_mode=True
  )

  pretrain.trainer.limit_val_batches = 0.0
  pretrain.trainer.val_check_interval = 100

  # Add the Nsys profiling callback if enabled.
  if profile_enabled:
    pretrain.trainer.callbacks.append(
        run.Config(
            NsysCallback,
            start_step=profile_start_step,
            end_step=profile_end_step,
            ranks=[int(x) for x in profile_ranks.split(",")],
            gen_shape=False,
        )
    )

  # Add the FLOPs measurement callback.
  pretrain.trainer.callbacks.append(
      run.Config(
          FLOPsMeasurementCallback,
          model_name="deepseekv3",
          model_config=pretrain.model.config,
          data_config=pretrain.data,
      )
  )

  # Disable checkpointing.
  pretrain.log.ckpt = None
  pretrain.trainer.enable_checkpointing = False

  pretrain.trainer.strategy.pipeline_model_parallel_size = 16
  pretrain.trainer.strategy.tensor_model_parallel_size = 1
  pretrain.trainer.strategy.expert_model_parallel_size = 8
  pretrain.trainer.strategy.virtual_pipeline_model_parallel_size = None

  pretrain.data.global_batch_size = 2048
  pretrain.trainer.plugins = bf16_with_fp8_mixed()
  pretrain.trainer.plugins.grad_reduce_in_fp32 = False

  pretrain.model.config.recompute_modules = ["mla_up_proj"]

  # Log every step.
  pretrain.trainer.log_every_n_steps = 1

  return pretrain


if __name__ == "__main__":
  run.cli.main(llm.pretrain, default_factory=recipe)
