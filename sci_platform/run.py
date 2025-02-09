from sci_plat import Platform
from utils.scientist_utils import read_txt_files_as_dict
import os
import argparse
# import pydevd_pycharm
# pydevd_pycharm.settrace('172.31.165.80', port=12345, stdoutToServer=True, stderrToServer=True)
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiments")
    # parser.add_argument(
    #     "--skip-idea-generation",
    #     action="store_true",
    #     help="Skip idea generation and load existing ideas",
    # )
    # parser.add_argument(
    #     "--skip-novelty-check",
    #     action="store_true",
    #     help="Skip novelty check and use existing ideas",
    # )

    # how many runs
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Calculate average on how many runs.",
    )
    # team limit
    parser.add_argument(
        "--team_limit",
        type=int,
        default=2,
        help="Max number of teams for a scientist.",
    )
    parser.add_argument(
        "--max_discuss_iteration",
        type=int,
        default=3,
        help="Max discuss iteration.",
    )
    parser.add_argument(
        "--max_team_member",
        type=int,
        default=4,
        help="Max team member of a team, actual team size is max_team_member.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Epochs.",
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    args.save_dir = f'team_info/{args.max_discuss_iteration}_itrs_{args.max_team_member}_members'
    args.log_dir = f'team_log/{args.max_discuss_iteration}_itrs_{args.max_team_member}_members'

    end =False
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    while end == False:
        print(f'{len(os.listdir(args.save_dir))} files are created...')
        # 每次循环创建新的平台实例
        platform_example = Platform(
            team_limit=args.team_limit,
            group_max_discuss_iteration=args.max_discuss_iteration,
            max_teammember=args.max_team_member - 1,
            log_dir=args.log_dir,
            info_dir=args.save_dir
        )
        # 运行平台
        platform_example.running(args.epochs)
        if len(os.listdir(args.save_dir)) >= args.team_limit * args.runs:
            end = True
