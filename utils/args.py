import argparse

import torch


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--load_ckpt", action="store_true", help="load checkpoint")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--exp_keys", type=str, default="")

    # optuna
    parser.add_argument("--sampler", type=str, default="tpe", choices=["rand", "tpe"])
    parser.add_argument("--num_trials", type=int, default=1)

    # federated arguments
    parser.add_argument(
        "--num_rounds", type=int, default=3000, help="communication rounds"
    )  # learnit: 100k for 1 outer bs
    parser.add_argument("--print_train", type=int, default=10)
    parser.add_argument("--print_test", type=int, default=100, help="tto0 & tto steps = 0")
    parser.add_argument("--save_period", type=int, default=100, help="save ckpt")
    parser.add_argument("--tto_period", type=int, default=1000, help="tto steps = args.tto_steps")
    parser.add_argument("--num_clients", type=int, default=50, help="number of clients")
    parser.add_argument("--num_ood_clients", type=int, default=0, help="number of ood clients")
    parser.add_argument("--num_participants", type=int, default=5, help="number of participants in each round")

    # model
    parser.add_argument("--algo", type=str, default="fedavg_maml", help="ml algorithm")
    parser.add_argument("--model", type=str, default="simplenerf")
    parser.add_argument("--num_layers", type=int, default=6)  # learnit
    parser.add_argument("--hidden_dim", type=int, default=256)  # learnit
    parser.add_argument("--num_oml_layers", type=int, default=2)  # OML

    # nerf model
    parser.add_argument("--max_freq", type=int, default=8)
    parser.add_argument("--num_freqs", type=int, default=20)

    # server
    parser.add_argument("--server_lr", type=float, default=0.9, help="parameter for proximal global SGD")
    parser.add_argument("--server_gradclip", type=float, default=5)

    # client
    parser.add_argument("--client_lr", type=float, default=5e-5)  # learnit (5e-4 for chairs)
    parser.add_argument("--client_inner_lr", type=float, default=0.1)  # learnit: 0.1
    parser.add_argument("--client_gradclip", type=float, default=5)
    parser.add_argument("--adapt_optim", type=str, default="sgd")

    parser.add_argument("--task_bs", type=int, default=1)
    parser.add_argument(
        "--test_task_bs",
        type=int,
        default=1,
        help="object bs for test. irrelevant to performance",
    )
    parser.add_argument("--client_epochs", type=int, default=1)  # learnit: 30 (meta_epochs)
    parser.add_argument("--epoch_to_iter", type=int, default=0, help="if 1, run (client_epochs) outer loops")
    parser.add_argument("--client_ray_bs", type=int, default=128)  # learnit
    parser.add_argument("--inner_loop_steps", type=int, default=8)  # learnit: 32 (task_bs: 8 -> oom)
    parser.add_argument("--use_q", type=int, default=0, help="use query views (reptile)")

    parser.add_argument("--tto_steps", type=int, default=512)  # learnit: 2000
    parser.add_argument("--tto_lr", type=float, default=0.1)  # learnit: 0.1
    parser.add_argument("--tto_step_period", type=int, default=-1)

    parser.add_argument(
        "--test_ray_bs",
        type=int,
        default=4096 * 2 * 2,
        help="ray bs for test. irrelevant to performance",
    )  # 4096

    parser.add_argument("--num_points_per_ray", type=int, default=128, help="num_samples")

    # dataset
    parser.add_argument("--dataset", type=str, default="cars", help="name of dataset")
    parser.add_argument("--alpha", type=float, default=1.0, help="dirichlet distribution parameter")
    parser.add_argument("--iid", action="store_true", help="whether i.i.d or not")
    parser.add_argument("--views_iid", action="store_true", help="whether views are i.i.d or not")
    parser.add_argument("--views_alpha", type=float, default=1.0, help="dirichlet distribution parameter")

    parser.add_argument(
        "--test_dist",
        type=str,
        default="consistent",
        choices=["uniform", "dirichlet", "consistent"],
    )
    parser.add_argument(
        "--num_objects_per_client",
        type=int,
        default=2,
        help="average number of objects per client.",
    )
    parser.add_argument(
        "--num_test_objects_per_client",
        type=int,
        default=2,
        help="average number of objects per client.",
    )
    parser.add_argument(
        "--num_support_views", type=int, default=20, help="number of support views for each scene"
    )  # learnit = 25
    parser.add_argument(
        "--num_query_views", type=int, default=10, help="number of query views for each scene"
    )  # learnit = 25
    parser.add_argument(
        "--num_test_support_views",
        type=int,
        default=-1,
        help="number of support views for each scene",
    )  # learnit = 25
    parser.add_argument(
        "--num_test_query_views", type=int, default=-1, help="number of query views for each scene"
    )  # learnit = 25

    parser.add_argument("--mean_num_support_views", type=int, default=4, help="number of support views for each scene")
    parser.add_argument("--mean_num_query_views", type=int, default=4, help="number of query views for each scene")
    parser.add_argument(
        "--mean_num_test_support_views",
        type=int,
        default=-1,
        help="number of support views for each scene",
    )
    parser.add_argument(
        "--mean_num_test_query_views", type=int, default=-1, help="number of query views for each scene"
    )

    parser.add_argument("--res_scale", type=int, default=2)

    # local training (no FL, no meta)
    parser.add_argument("--participant_ids", type=str, default="")
    parser.add_argument("--participant_period", type=int, default=1)

    # triplet loss
    parser.add_argument("--triplet_alpha", type=float, default=0.0, help="triplet loss alpha")
    parser.add_argument(
        "--triplet_gamma",
        type=float,
        default=0.75,
        help="triplet loss gamma. coefficient for global loss",
    )

    # fedexp
    parser.add_argument("--fedexp_epsilon", type=float, default=0.001)
    parser.add_argument("--fedexp_type", type=str, default="paper")

    # fedprox
    parser.add_argument("--fedprox_mu", type=float, default=0.1)

    args = parser.parse_args()
    args.num_total_clients = args.num_clients + args.num_ood_clients
    if args.num_test_support_views == -1:
        args.num_test_support_views = args.num_support_views
    if args.num_test_query_views == -1:
        args.num_test_query_views = args.num_query_views
    if args.mean_num_test_support_views == -1:
        args.mean_num_test_support_views = args.mean_num_support_views
    if args.mean_num_test_query_views == -1:
        args.mean_num_test_query_views = args.mean_num_query_views
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args
