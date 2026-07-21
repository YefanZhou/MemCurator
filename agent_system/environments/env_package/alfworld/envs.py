# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward


# ---------------------------------------------------------------------------------------------- #
# Startup-scan cache (A-axis fix; see docs/memcurator_startup_scan_optimization.md).              #
# ---------------------------------------------------------------------------------------------- #
# AlfredTWEnv.__init__ -> collect_game_files() does list(os.walk(data_path)) over ~8810 trial dirs
# on /fsx (~7-min silent network-stat storm) + ~17.6k json.load (~2-3-min visible bar) = ~9 min,
# EVERY env construction (train + val => 2x/run). The result (self.game_files) is DETERMINISTIC for
# a fixed (data_path, task_types, goal_desc_human_anns_prob, train_eval, num_games) key, so we
# disk-cache it: first run pays ~9 min, every run after is sub-second. We MONKEYPATCH the method
# (installed once, idempotent) rather than edit the vendored alfred_tw_env.py — keeps the
# runtime_env.yaml upload story clean and avoids drift with the eval-side copy. Falls back to the
# original scan on ANY error, so it can never wedge a run. Cache lives on /fsx (persistent).
import hashlib as _hashlib
import json as _json

_GAME_FILES_CACHE_DIR = os.environ.get(
    "ALFWORLD_GAMEFILES_CACHE_DIR",
    "/fsx/home/yefan.zhou/mem-evolve/data/alfworld_gamefiles_cache",
)
_GAME_FILES_CACHE_VERSION = "v1"  # bump to invalidate ALL caches manually (dataset/logic change)


def _install_game_files_cache():
    """Idempotently monkeypatch AlfredTWEnv.collect_game_files to disk-cache its game_files list."""
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (
        AlfredTWEnv,
    )
    if getattr(AlfredTWEnv, "_collect_game_files_cached", False):
        return  # already installed this process
    _orig_collect = AlfredTWEnv.collect_game_files

    def _cache_key(self):
        te = self.train_eval
        ds = self.config['dataset']
        if te == "train":
            data_path = os.path.expandvars(ds['data_path']); num = ds['num_train_games']
        elif te == "eval_in_distribution":
            data_path = os.path.expandvars(ds['eval_id_data_path']); num = ds['num_eval_games']
        elif te == "eval_out_of_distribution":
            data_path = os.path.expandvars(ds['eval_ood_data_path']); num = ds['num_eval_games']
        else:
            raise ValueError(f"unknown train_eval={te!r}")
        payload = {
            "version": _GAME_FILES_CACHE_VERSION,
            "data_path": data_path,
            "train_eval": te,
            "task_types": sorted(self.config['env']['task_types']),
            "goal_desc_human_anns_prob": self.config['env']['goal_desc_human_anns_prob'],
            "num_games": num,
        }
        h = _hashlib.md5(_json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        return data_path, os.path.join(_GAME_FILES_CACHE_DIR, f"gamefiles_{te}_{h}.json")

    def collect_game_files(self, verbose=False):
        try:
            data_path, cache_path = _cache_key(self)
        except Exception as e:
            print(f"[alfworld-cache] key error ({e!r}); scanning fresh", flush=True)
            return _orig_collect(self, verbose=verbose)

        # --- cache HIT (with root-mtime invalidation) ---
        if os.path.exists(cache_path):
            try:
                blob = _json.load(open(cache_path))
                cur_mtime = os.path.getmtime(data_path)
                if abs(blob.get("data_path_mtime", -1.0) - cur_mtime) < 1.0:
                    self.game_files = list(blob["game_files"])
                    self.num_games = len(self.game_files)
                    print(f"[alfworld-cache] game_files loaded from cache "
                          f"({self.num_games} games, split={self.train_eval}, "
                          f"key={os.path.basename(cache_path)})", flush=True)
                    return
                print(f"[alfworld-cache] STALE (root mtime changed: cached="
                      f"{blob.get('data_path_mtime')} cur={cur_mtime}); rescanning", flush=True)
            except Exception as e:
                print(f"[alfworld-cache] read failed ({e!r}); rescanning", flush=True)

        # --- cache MISS: run the real scan, then persist ---
        _orig_collect(self, verbose=verbose)
        try:
            os.makedirs(_GAME_FILES_CACHE_DIR, exist_ok=True)
            tmp = f"{cache_path}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                _json.dump({"data_path_mtime": os.path.getmtime(data_path),
                            "game_files": self.game_files}, f)
            os.replace(tmp, cache_path)  # atomic
            print(f"[alfworld-cache] game_files scanned fresh + cached "
                  f"({self.num_games} games, split={self.train_eval} -> "
                  f"{os.path.basename(cache_path)})", flush=True)
        except Exception as e:
            print(f"[alfworld-cache] cache write failed ({e!r}); continuing uncached", flush=True)

    AlfredTWEnv.collect_game_files = collect_game_files
    AlfredTWEnv._collect_game_files_cached = True
    print("[alfworld-cache] collect_game_files disk-cache installed "
          f"(dir={_GAME_FILES_CACHE_DIR})", flush=True)

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """

    def __init__(self, config, seed, base_env):
        self.seed = seed
        self.base_env = base_env  # own deserialized copy (Ray serializes on actor creation)
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)

    def filter_game_files_by_task_type(self, task_type: str):
        """Reinitialize this worker's env to only sample games of `task_type`."""
        filtered = [gf for gf in self.base_env.game_files if task_type in gf]
        if not filtered:
            raise ValueError(f"No game files found for task_type={task_type!r}")
        self.base_env.game_files = filtered
        self.base_env.num_games = len(filtered)
        self.env.close()
        self.env = self.base_env.init_env(batch_size=1)
        self.env.seed(self.seed)

    def pin_game_file(self, game_file: str):
        """[MemCurator] Reinitialize this worker to serve EXACTLY one game_file.

        Additive helper (mirrors filter_game_files_by_task_type). Sets game_files=[game_file]
        and re-registers the textworld gym env, so subsequent reset() always loads THIS game —
        used by dataset-driven training to pin a slot to a specific target task. No effect on
        the default (random-sampling) path, which never calls this.
        """
        self.base_env.game_files = [game_file]
        self.base_env.num_games = 1
        self.env.close()
        self.env = self.base_env.init_env(batch_size=1)
        self.env.seed(self.seed)

    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        # Install the game_files disk-cache BEFORE constructing the env (the scan fires inside the
        # constructor). Patches AlfredTWEnv.collect_game_files only; AlfredThorEnv is a SEPARATE class
        # (not a subclass) so it's simply left un-patched (Thor is unused here; the walk cost is a
        # TW-split concern). Guard covers both names so the install is attempted iff relevant.
        if env_type in ('AlfredTWEnv', 'AlfredThorEnv'):
            try:
                _install_game_files_cache()
            except Exception as e:
                print(f"[alfworld-cache] install failed ({e!r}); using uncached scan", flush=True)
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def set_task_types_per_slot(self, task_types_per_slot: list):
        """Reconfigure each worker to only sample its assigned task type (parallel Ray calls)."""
        futures = [
            self.workers[i].filter_game_files_by_task_type.remote(task_types_per_slot[i])
            for i in range(self.num_processes)
            if task_types_per_slot[i] is not None
        ]
        ray.get(futures)

    def set_game_files_per_slot(self, game_files_per_slot: list):
        """[MemCurator] Pin each worker slot to an exact game_file (parallel Ray calls).

        Additive (mirrors set_task_types_per_slot). ``game_files_per_slot`` has one entry per
        slot; None entries are left unchanged. Used by dataset-driven training to pin slots to
        specific target games before reset(). Unused by the default random-sampling path.
        """
        assert len(game_files_per_slot) == self.num_processes, \
            f"expected {self.num_processes} game_files, got {len(game_files_per_slot)}"
        futures = [
            self.workers[i].pin_game_file.remote(game_files_per_slot[i])
            for i in range(self.num_processes)
            if game_files_per_slot[i] is not None
        ]
        ray.get(futures)

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)