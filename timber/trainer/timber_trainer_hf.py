import os
import pathlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Union, Any

import datasets
import torch
import torch.onnx
import transformers
from torch import nn
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch.utils.checkpoint
import torch.autograd
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

# torch.autograd.set_detect_anomaly(True)

from timber.models.modeling_llama import LlamaForCausalLM, LlamaConfig, LlamaDecoderLayer

from timber.utils import seed
from timber.dataset.labdataset import LabDataset
from timber.dataset.booksum import BookSumDataset
from timber.dataset.alpaca import AlpacaDataset
from timber.dataset.openwebtext import OpenWebTextDataset
from timber.dataset.lmsys import LmsysChatDataset
from torch.utils.data import DataLoader, random_split

torch.set_float32_matmul_precision('high')


@dataclass
class TrainConfig:
    disable_kd: bool = False
    using_fsdp: bool = False
    lr: float = 5e-5
    batch_size: int = 1
    accumulation_steps: int = 2
    lora_r: int = 32
    save_steps: int = 100
    dense_queries: int = None
    seq_len: int = 4096
    max_steps: int = 1000000
    model_checkpoint_dir: str = "./saves/dev/checkpoint"
    dataset: str = 'wikitext103'
    load_from_checkpoint: str = None
    k: int = 512
    block_size_q: int = 8
    block_size_k: int = 8
    init_from_checkpoint: str = None
    method: str = 'timber'
    model: str = 'llama32k'


class LabDataModule:
    def __init__(
            self,
            config: TrainConfig,
            num_workers: int = 0,
            data_dir: Path = "data",
            download: bool = True,
            train_size: float = 0.9,
    ):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.block_size = config.seq_len
        self.download = download
        self.num_workers = num_workers
        self.train_size = train_size
        self.dataset = None
        self.tokenizer = load_tokenizer()
        self.bsize = config.batch_size

    def get_dataset(self):
        if self.config.dataset in ['wikitext2', 'wikitext103']:
            dataset = LabDataset(
                data_dir=self.data_dir,
                block_size=self.block_size,
                download=self.download,
                tokenizer=self.tokenizer,
                dataset=self.config.dataset,
            )
        elif self.config.dataset in ['alpaca']:
            dataset = AlpacaDataset(
                tokenizer=self.tokenizer,
            )
        elif self.config.dataset in ['booksum']:
            dataset = BookSumDataset(
                tokenizer=self.tokenizer,
            )
        elif self.config.dataset in ['openwebtext']:
            dataset = OpenWebTextDataset(
                tokenizer=self.tokenizer,
                stride=self.block_size,
            )
        elif self.config.dataset == 'lmsys':
            dataset = LmsysChatDataset(
                tokenizer=self.tokenizer,
            )
        else:
            raise Exception()
        return dataset

    def prepare_data(self):
        self.get_dataset()

    def setup(self, stage: str):
        self.dataset = self.get_dataset()
        if self.config.dataset in ['wikitext2', 'wikitext103']:
            if stage == "fit" or stage is None:
                test_size = min(100, len(self.dataset) * (1 - self.train_size))
                train_size = int(len(self.dataset) - test_size)
                self.train_data, self.val_data = random_split(self.dataset, lengths=[train_size, test_size])
            if stage == "test" or stage is None:
                self.test_data = self.val_data
        elif self.config.dataset in ['booksum', 'alpaca', 'openwebtext', 'lmsys']:
            if stage == "fit" or stage is None:
                def train_val_dataset(dataset, val_split=0.05):
                    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
                    train = Subset(dataset, train_idx)
                    valid = Subset(dataset, val_idx)
                    return train, valid

                self.train_data, self.val_data = train_val_dataset(self.dataset)
            if stage == "test" or stage is None:
                self.test_data = self.val_data
        else:
            raise Exception()

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers, batch_size=self.bsize)

    def val_dataloader(self):
        return DataLoader(self.val_data, num_workers=self.num_workers, batch_size=self.bsize)

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.num_workers, batch_size=self.bsize)


from peft import LoraConfig, TaskType, PeftModel
from peft import get_peft_model, prepare_model_for_kbit_training


def get_hf_dataset(ds):
    def gen():
        for idx in range(len(ds)):
            inputs, targets = ds[idx]
            yield {'input_ids': inputs, 'labels': targets}

    return datasets.IterableDataset.from_generator(gen)


def load_model(
        train_config: TrainConfig = None,
        method='timber',
        device='cuda:0',
        is_teacher=False,
):
    if train_config.using_fsdp:
        device = 'cpu'

    MODELS = {
        'llama32k': 'togethercomputer/LLaMA-2-7B-32K',
        'llama13b': 'meta-llama/Llama-2-13b-hf',
    }
    assert train_config.model in MODELS, MODELS.keys()
    model_id = MODELS[train_config.model]

    config = LlamaConfig.from_pretrained(model_id)
    config._attn_implementation = config.attn_implementation = 'sdpa'

    quant_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_skip_modules=['tree_avgpool_scaler'],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    if train_config.using_fsdp:
        quant_config = None

    if method == 'timber':
        ModelClass = LlamaForCausalLM
    else:
        ModelClass = transformers.LlamaForCausalLM

    print(ModelClass)

    model = ModelClass.from_pretrained(
        model_id,
        config=config,
        device_map="auto",
        # device_map={"" : device} if device != 'cpu' else 'cpu',
        load_in_4bit=None if quant_config is not None else True,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    for m in model.modules():
        if hasattr(m, 'attention_method'):
            m.attention_method = method
            m.tree_k = train_config.k
            m.tree_block_size_q = train_config.block_size_q
            m.tree_block_size_k = train_config.block_size_k
            if train_config.dense_queries is None:
                train_config.dense_queries = train_config.k
            m.tree_dense_queries = train_config.dense_queries
        if hasattr(m, 'gradient_checkpointing'):
            m.gradient_checkpointing = True
            if train_config.using_fsdp:
                # m._gradient_checkpointing_func = deepspeed.checkpointing.checkpoint
                m._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
            else:
                m._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    if not is_teacher:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_config.lora_r,
            lora_alpha=train_config.lora_r // 2,
            lora_dropout=0.05,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
                # 'input_layernorm', 'post_attention_layernorm'
            ],
            modules_to_save=[
                'tree_avgpool_scaler',
                'input_layernorm', 'post_attention_layernorm'
            ]
        )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        if train_config.init_from_checkpoint is not None:

            print(f"Loading peft model from {train_config.init_from_checkpoint}")

            if pathlib.Path(train_config.init_from_checkpoint).is_dir():
                model = PeftModel.from_pretrained(model, train_config.init_from_checkpoint)
                model.print_trainable_parameters()

            else:
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

                print('loading from', train_config.init_from_checkpoint)
                state_dict = torch.load(train_config.init_from_checkpoint, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                keys = list(state_dict.keys())
                for key in keys:
                    x = state_dict[key]
                    state_dict[key.strip('model.')] = x
                    del state_dict[key]
                try:
                    result = model.load_state_dict(state_dict, strict=False)
                    print('load result', result)
                except RuntimeError as e:
                    pass
                print('lora checkpoint loaded from', train_config.init_from_checkpoint)

        else:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    return model


def load_tokenizer():
    model_id = 'togethercomputer/LLaMA-2-7B-32K'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'
    return tokenizer


class Trainer(Seq2SeqTrainer):
    def __init__(
            self,
            config=None,
            model=None,
            teacher=None,
            args=None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=(None, None),
            preprocess_logits_for_metrics=None,
    ):
        super().__init__(
            model, args, data_collator, train_dataset, eval_dataset,
            tokenizer, model_init, compute_metrics, callbacks,
            optimizers, preprocess_logits_for_metrics
        )

        self.model = model
        self.teacher = teacher

        self.validation_preds = []
        self.validation_targets = []

        self.config = config
        self.pad_token_id = tokenizer.pad_token_id

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        result = super().training_step(model, inputs)

        total_max_mem = sum(
            torch.cuda.max_memory_allocated(device)
            for device in range(torch.cuda.device_count())
        )

        return result

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.teacher is not None:
            self.teacher.eval()
        self.model.train()

        inputs, target = inputs['input_ids'], inputs['labels']
        inputs = inputs[..., :self.config.seq_len]
        target = target[..., :self.config.seq_len]

        if not self.config.disable_kd:
            with torch.no_grad():  # , torch.autocast('cuda', torch.bfloat16):
                output_teacher = self.teacher(inputs, output_hidden_states=not self.config.disable_kd)
        # with torch.autocast('cuda', torch.bfloat16):
        output = self.model(
            inputs,
            attention_mask=(inputs.ne(self.pad_token_id)).to(inputs.dtype),
            labels=target,
            output_hidden_states=not self.config.disable_kd
        )
        logits = output.logits

        loss_model = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).to(torch.float32),
            target.reshape(-1)
        )

        loss_kd_hidden = 0
        loss_kd_logits = 0
        if not self.config.disable_kd:
            for teacher_layer, student_layer in zip(output_teacher.hidden_states, output.hidden_states):
                loss_kd_hidden += torch.nn.functional.mse_loss(student_layer.to(torch.float32),
                                                               teacher_layer.to(torch.float32))
            loss_kd_hidden = loss_kd_hidden / len(output_teacher.hidden_states)

            loss_kd_logits = torch.nn.functional.kl_div(
                output.logits.view(-1, logits.shape[-1]).to(torch.float32).log_softmax(-1),
                output_teacher.logits.view(-1, logits.shape[-1]).to(torch.float32).softmax(-1),
                reduction='batchmean',
            )

        if not self.config.disable_kd:
            loss = loss_model * 0.1 + (loss_kd_hidden + loss_kd_logits) * 2.5
        else:
            loss = loss_model

        log_dict = dict()
        log_dict["training/loss_model"] = loss_model.item()
        if not self.config.disable_kd:
            if loss_kd_hidden > 0:
                log_dict["training/loss_kd_hidden"] = loss_kd_hidden.item()
            if loss_kd_logits > 0:
                log_dict["training/loss_kd_logits"] = loss_kd_logits.item()
        log_dict["training/loss"] = loss.item()
        self.log(log_dict)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        inputs, target = batch
        inputs = inputs[..., :self.config.seq_len]
        target = target[..., :self.config.seq_len]

        with torch.no_grad():  # , torch.autocast('cuda', torch.bfloat16):
            # print('asdfasdf', inputs.shape, target.shape, flush=True)
            output = self.model(
                inputs,
                attention_mask=(inputs != self.pad_token_id).to(inputs.dtype),
                labels=target,
            ).logits
            loss = torch.nn.functional.cross_entropy(
                output.view(-1, output.shape[-1]),
                target.view(-1)
            )
        self.log({"val/loss": loss.item()})

        self.validation_preds.append(output.cpu())
        self.validation_targets.append(target.cpu())

    def on_validation_epoch_end(self):
        from torchmetrics.text.perplexity import Perplexity
        with torch.no_grad():
            device = 'cpu'
            if self.config.using_fsdp:
                device = 'cuda'
            calculator = Perplexity(ignore_index=-100).to(device)
            for preds, target in zip(self.validation_preds, self.validation_targets):
                calculator.update(preds.to(device), target.to(device))
            ppl = calculator.compute()
        ppl = ppl.item()
        print('val/ppl', ppl)
        self.log({"val/ppl": ppl})

        self.validation_preds.clear()
        self.validation_targets.clear()


def main(config: TrainConfig):
    os.environ["WANDB_PROJECT"] = "timber-attention"

    os.makedirs('./saves/dev/wandb', exist_ok=True)
    os.makedirs('./saves/dev/checkpoint', exist_ok=True)

    if config.method == 'timber':
        filename = f'{config.model}-{config.dataset}-{config.seq_len}-bq{config.block_size_q}-bk{config.block_size_k}-k{config.k}'
    elif config.method == 'none':
        filename = f'{config.model}-{config.dataset}-{config.seq_len}'
    elif config.method == 'reformer':
        filename = f'{config.model}-{config.method}-{config.dataset}-{config.seq_len}-k{config.k}'
    elif config.method == 'performer':
        filename = f'{config.model}-{config.method}-{config.dataset}-{config.seq_len}'
    else:
        raise Exception()

    config.model_checkpoint_dir = config.model_checkpoint_dir + '/' + filename

    model = load_model(train_config=config, method=config.method)
    if not config.disable_kd:
        teacher = load_model(train_config=config, method='none', is_teacher=True)
    else:
        teacher = None

    datamodule = LabDataModule(config=config)
    datamodule.setup("fit")

    trainer_config = Seq2SeqTrainingArguments(
        logging_steps=1,
        fp16=True,
        output_dir=config.model_checkpoint_dir,
        gradient_accumulation_steps=config.accumulation_steps,
        #num_train_epochs=20,
        max_steps=config.max_steps,
        report_to=["wandb"],
        gradient_checkpointing=True,
        save_total_limit=3,
        save_steps=config.save_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        ignore_data_skip=True
    )

    trainer = Trainer(
        config=config,
        model=model,
        teacher=teacher,
        args=trainer_config,
        train_dataset=get_hf_dataset(datamodule.train_data),
        eval_dataset=get_hf_dataset(datamodule.val_data),
        tokenizer=datamodule.tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=datamodule.tokenizer,
            padding='longest',
            pad_to_multiple_of=16,
        ),
    )

    trainer.train(resume_from_checkpoint=config.load_from_checkpoint)


def run():
    seed()

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='llama32k', type=str)
    parser.add_argument('--method', default='timber', type=str)
    parser.add_argument('--dense_queries', default=None, type=int)
    parser.add_argument('--using_fsdp', action='store_true')
    parser.add_argument('--disable_kd', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=-1, type=int)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--lora_r', default=-1, type=int)
    parser.add_argument('--lr', default=-1, type=float)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--dataset', default='wikitext103', type=str)
    parser.add_argument('--seq_len', default=-1, type=int)
    parser.add_argument('--save_steps', default=-1, type=int)
    parser.add_argument('--init_checkpoint', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--k', default=512, type=int)
    parser.add_argument('--block_size_q', default=16, type=int)
    parser.add_argument('--block_size_k', default=2, type=int)

    args = parser.parse_args()

    train_config = TrainConfig(
        using_fsdp=args.using_fsdp,
        disable_kd=args.disable_kd,
        dataset=args.dataset,
        load_from_checkpoint=args.checkpoint,
        k=args.k,
        block_size_q=args.block_size_q,
        block_size_k=args.block_size_k,
        method=args.method,
        model=args.model,
    )
    if args.gradient_accumulation_steps > 0:
        train_config.accumulation_steps = args.gradient_accumulation_steps
    if args.lora_r > 0:
        train_config.lora_r = args.lora_r
    if args.lr > 0:
        train_config.lr = args.lr
    if args.batch_size > 0:
        train_config.batch_size = args.batch_size
    if args.max_steps > 0:
        train_config.max_steps = args.max_steps
    if args.seq_len > 0:
        train_config.seq_len = args.seq_len
    if args.save_steps > 0:
        train_config.save_steps = args.save_steps
    if args.dense_queries is not None:
        train_config.dense_queries = args.dense_queries
    if args.init_checkpoint is not None:
        train_config.init_from_checkpoint = args.init_checkpoint

    main(train_config)


if __name__ == "__main__":
    run()
