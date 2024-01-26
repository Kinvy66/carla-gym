import argparse

import carla_gym
import supersuit as ss
import ray
import ray.tune as tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.catalog import ModelCatalog
from ray.tune import register_env
import tensorflow as tf
from tensorflow.keras import layers, models
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.air import RunConfig

from carla_gym.env_wrappers.multi_carla import MultiCarlaEnv

argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
argparser.add_argument("--xml_config_path", default="stop_sign_3c_town03.xml", help="Path to the xml config file")
argparser.add_argument("--maps_path", default="/Game/Carla/Maps/", help="Path to the CARLA maps")
argparser.add_argument("--render_mode", default="human", help="Path to the CARLA maps")

config_args = vars(argparser.parse_args())
config_args["discrete_action_space"] = True

class Mnih15(TFModelV2):
    def __init__(
            self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(
            observation_space, action_space, num_outputs, model_config, name
        )
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='same', input_shape=(84, 84, 3)))
        self.model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu',  padding='same'))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(num_outputs, activation=None))

    def forward(self, input_dict, state, seq_lens):
        out, self._value_out = self.model(input_dict["obs"])
        return out, []

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

def env_creator(config):
    env = carla_gym.parallel_env(**config_args)

    env = ss.resize_v1(env, x_size=84, y_size=84)
    return env

if __name__ == "__main__":

    env_name = "HomoNcomIndePOIntrxMASS3CTWN3-V0"

    # ISSUE: local_mode
    ray.init(local_mode=True,num_gpus=1)

    register_env(env_name, lambda config: MultiCarlaEnv(env_creator(config)))
    ModelCatalog.register_custom_model("mnih15", Mnih15)

    config = (
        PPOConfig()
        .environment(env_name)
        # .environment(env=StopSign3CarTown03)
        .framework(framework="tf2")
        .multi_agent(
            policies={"car1", "car2", "car3"},
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
            # policies_to_train=["car1", "car2", "car3"]
        )
        .rollouts(num_rollout_workers=1)
        .training(
            # model={
            #     "custom_model": "mnih15"
            # },
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .resources(num_gpus=1))

    stop = {
        "training_iteration": 100,
        "timesteps_total": 4000000,
        "episodes_total": 1024
    }

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=RunConfig(stop=stop),
    )
    results = tuner.fit()

    ray.shutdown()