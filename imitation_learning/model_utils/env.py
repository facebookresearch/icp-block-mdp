# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import os
from argparse import Namespace
from typing import List, Union

import utils


def make_train_and_eval_envs(args: Namespace):
    """Method to make the train and eval envs"""

    (
        fns_to_make_train_envs,
        fns_to_make_eval_envs,
    ) = make_fns_to_make_train_and_eval_envs(args=args)
    train_envs = [fn() for fn in fns_to_make_train_envs]

    eval_envs = [fn() for fn in fns_to_make_eval_envs]

    return train_envs, eval_envs


def _get_resource_files_for_train_and_eval_envs(args: Namespace):
    def _split_files_across_envs(resource_files: Union[List[str], str], num_envs: int):
        """Method to split the resource files across environments"""
        resource_files = glob.glob(os.path.expanduser(resource_files))
        num_files_per_env = int(len(resource_files) / num_envs)
        if num_files_per_env == 0:
            return [[] for i in range(0, num_envs)]
        return [
            resource_files[i : i + num_files_per_env]
            for i in range(0, len(resource_files), num_files_per_env)
        ]

    resource_files = {
        "train": _split_files_across_envs(
            resource_files=args.train_resource_files, num_envs=args.num_train_envs
        ),
        "eval": _split_files_across_envs(
            resource_files=args.eval_resource_files, num_envs=args.num_eval_envs
        ),
    }

    return resource_files


def make_fns_to_make_train_and_eval_envs(args):
    """Method to make the train and eval envs"""

    if args.change_angle:
        fns_to_make_train_envs = [
            utils.fn_to_make_env(args, seed=i, resource_files=None, camera_id=i + 1)
            for i in range(args.num_train_envs)
        ]
        fns_to_make_eval_envs = [
            utils.fn_to_make_env(
                args, seed=args.num_train_envs + i, resource_files=None, camera_id=0,
            )
            for i in range(args.num_eval_envs)
        ]
    else:
        fns_to_make_train_envs = [
            utils.fn_to_make_env(
                args, seed=i, resource_files=resource_files["train"][i], camera_id=0
            )
            for i in range(args.num_train_envs)
        ]
        fns_to_make_eval_envs = [
            utils.fn_to_make_env(
                args,
                seed=args.num_train_envs + i,
                resource_files=resource_files["eval"][i],
                camera_id=0,
            )
            for i in range(args.num_eval_envs)
        ]
    return fns_to_make_train_envs, fns_to_make_eval_envs
