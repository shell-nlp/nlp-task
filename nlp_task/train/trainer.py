from typing import Callable, Dict, List, Tuple, Union, Optional
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BaseImageProcessor,
    FeatureExtractionMixin,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.data.data_collator import DataCollator
from torch.utils.data import Dataset, IterableDataset
import torch
import torch.nn as nn
import datasets
from loguru import logger
from .adversarial_training import FGM


class Trainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[
            Union[Dataset, IterableDataset, "datasets.Dataset"]
        ] = None,
        eval_dataset: Optional[
            Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]
        ] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        use_fgm: bool = False,
        epsilon: float = 1.0,
        emb_name: str = "word_embeddings",
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.fgm = None
        if use_fgm:
            logger.info("开启FGM...")
            self.fgm = FGM(model, epsilon=epsilon, emb_name=emb_name)

    def training_step(self, model, inputs, num_items_in_batch=None):
        # 正常前向传播
        loss = super().training_step(
            model, inputs, num_items_in_batch=num_items_in_batch
        )
        if self.fgm:
            # 添加对抗扰动
            self.fgm.attack()
            loss_adv = super().training_step(
                model, inputs, num_items_in_batch=num_items_in_batch
            )

            # 恢复原始权重
            self.fgm.restore()
            loss = loss + loss_adv

        # 返回总损失
        return loss
