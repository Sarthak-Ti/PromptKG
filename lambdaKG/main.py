import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
from lit_models.utils import EMA
import os

import sys
sys.setrecursionlimit(5000) #for longer tries

# import warnings

# warnings.filterwarnings("ignore", message="Be aware, overflowing tokens are not returned*")
# warnings.filterwarnings("ignore", message="noisy*")
# warnings.filterwarnings("always")
# #to ignore this annoying warning

import transformers
transformers.logging.set_verbosity_error()


os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 

# In order to ensure reproducible experiments, we must set random seeds.


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--lit_model_class", type=str, default="TransformerLitModel")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_class", type=str, default="KGC")
    parser.add_argument("--model_class", type=str, default="RobertaUseLabelWord")
    parser.add_argument("--early_stop", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt",type=str,default=" ")
    parser.add_argument("--wandb_name", type=str, default='kgc')
    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    lit_model_class = _import_class(f"lit_models.{temp_args.lit_model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)
    model_group = parser.add_argument_group("Model Args")
    if hasattr(model_class, "add_to_argparse"):
        model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_model_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

metric_list = {"knnkge" : "hits10",
                "simkgc" : "acc1",
                "t5kbqa" : "hits1"}

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    args.gpus = torch.cuda.device_count()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    data_class = _import_class(f"data.{args.data_class}") #chooses module based on args, this is KGT5DataModule even for bart?
    litmodel_class = _import_class(f"lit_models.{args.lit_model_class}") #this one based on it like kgbartlit or whatever
    

    # perfered , warp the transformers encoder
    method_name = args.model_class.lower().replace("model","") #like bartkgc
    if method_name in metric_list: #has list of ones that use different metrics, otherwise use hits10
        metric_name = metric_list[method_name]
    else:
        metric_name = "hits10" #for bart kgc

    if "csk" in args.dataset: #if you do csk, use auc. We don't
        metric_name = "auc"

    data = data_class(args)#KGT5DataModule
    print(data)
    tokenizer = data.tokenizer#KGT5DataModule
    lit_model = litmodel_class(args=args, tokenizer=tokenizer, num_relation=data.num_relation, num_entity = data.num_entity) #defines trie
    #this is the KGBART lit model class, which is a pytorch lightning module

    # if args.checkpoint:
    #     params_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
    #     for k in list(params_dict.keys()):
    #         if "wte" in k:
    #             params_dict.pop(k)

    #     lit_model.load_state_dict(params_dict, strict=False)
    



    # ----- set up all the callbacks for training and logging ---

    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="kgc", name=args.wandb_name)
        logger.log_hyperparams(vars(args))

    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor=metric_name, mode="max",
        filename='{epoch}-{acc1:.2f}',
        dirpath=os.path.join("output", args.dataset),
        save_weights_only=True,
        every_n_train_steps= None
    )
    callbacks = [model_checkpoint]
    if args.early_stop:
        early_callback = pl.callbacks.EarlyStopping(monitor=metric_name, mode="max", patience=4)
        callbacks.append(early_callback)
    if hasattr(args, "ema_decay") and args.ema_decay != 0.0:
        callbacks.append(EMA(args.ema_decay, ema_device="cuda"))
    
    if "mix" in args.model_class.lower(): #Now sure what mix is, something else
        print("running on cpu-gpu mode...")
        callbacks.append(EmbUpdateCallback(lit_model.model.ent_embeddings))

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs")
    
    # if args.checkpoint:
    #     lit_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")['state_dict'])
    #     trainer.test(lit_model, datamodule=data)
    #     return
    
    #here's some code if you simply want to test the data using the litmodel class
    #trie access using litmodel.entity_trie.trie_dict
    #it follows the format of </s> <s> <entity short form> </s>
    
    #to access an individual data point, use 
    #data.setup()
    #train_loader = data_module.train_dataloader()
    #train_loader.dataset[0]
    #line 720 of transformers.py has the actual training loop

    trainer.fit(lit_model, datamodule=data) #so it does indeed seem to run...? And speeds up quite a bit. Decently fast on 139 million parameters
    
    # make sure use one device to test
    args.devices = 1
    args.accumulate_grad_batches = None
    tester = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs", gpus=args.gpus)
    result = tester.test(lit_model, data)

    # path = model_checkpoint.best_model_path

    # lit_model.load_state_dict(torch.load(path)["state_dict"])
    # print(path)

    # result = trainer.test(lit_model, data)
    print(result)

if __name__ == "__main__":

    main()
