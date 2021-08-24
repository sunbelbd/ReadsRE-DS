import argparse
from paddlenlp.transformers import BertTokenizer
from dist_trainer import train
from utils import init_signal_handler, init_distributed_mode, initialize_exp
import paddle
import random
import os
import numpy as np
from paddle.distributed import fleet


def main(params):
    # initialize the multi-GPU / multi-node training
    # init_distributed_mode(params)

    strategy = fleet.DistributedStrategy()
    fleet.init(is_collective=True, strategy=strategy)

    # initialize the experiment
    # logger = initialize_exp(params)

    # trainer = Trainer(args)
    # trainer.train()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6666'  # 8888, 7777

    passage_data = open(args.passage_path, "r").readlines()
    if "base" in args.model_name_or_path:
        doc_embed = np.zeros([len(passage_data), 768], dtype="float32")
    else:  # large
        doc_embed = np.zeros([len(passage_data), 1024], dtype="float32")
    shared_doc_embed = doc_embed
    args.shared_doc_embed = shared_doc_embed

    train(args)
    # paddle.distributed.spawn(train, nprocs=args.gpus, args=(args, ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="NYT", type=str, help="The name of the task to train")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased", # bert-large-uncased # "bert-base-uncased"
        help="Model Name or Path",
    )
    parser.add_argument("--passage_path", default="../nyt10/train_passage.txt", type=str, help="passage path")
    # "../nyt10/train_passage.txt"
    parser.add_argument("--bag_path", default="../nyt10/train_bag.pkl", type=str, help="bag path")
    # "../nyt10/train_bag.pkl"
    parser.add_argument("--topK", default=5, type=int, help="number of retrieved candidates")
    parser.add_argument("--model_dir", default="./model_best", type=str, help="Path to model")
    # "./model_best_dist9"
    parser.add_argument("--seed", type=int, default=77, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=12, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len",
        default=120,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--learning_rate",
        default=3e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=5.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--weight_decay", default=1e-3, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout for fully-connected layers",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=5000, help="Log every X updates steps.")
    parser.add_argument("--update_MIPS_steps", type=int, default=250, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # float16 / AMP API
    parser.add_argument("--fp16", action="store_true", help="Run model with float16")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")

    # multi-gpu / multi-node
    parser.add_argument("--multi_gpu", type=bool, default=True,
                        help="Is a Multi-GPU task?")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()
    main(args)
