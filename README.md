# MISA

## Abstract

Generalization across environments is critical to the successful application of reinforcement learning algorithms to real-world challenges. In this paper, we consider the problem of learning abstractions that generalize in block MDPs, families of environments with a shared latent state space and dynamics structure over that latent space, but varying observations. We leverage tools from causal inference to propose a method of invariant prediction to learn model-irrelevance state abstractions (MISA) that generalize to novel observations in the multi-environment setting. We prove that for certain classes of environments, this approach outputs with high probability a state abstraction corresponding to the causal feature set with respect to the return. We further provide more general bounds on model error and generalization error in the multi-environment setting, in the process showing a connection between causal variable selection and the state abstraction framework for MDPs. We give empirical evidence that our methods work in both linear and nonlinear settings, attaining improved generalization over single-and multi-task baselines.

## Citation

```
@inproceedings{zhang2020invariant,
    title={Invariant Causal Prediction for Block MDPs},
    author={Amy Zhang and Clare Lyle and Shagun Sodhani and Angelos Filos and Marta Kwiatkowska and Joelle Pineau and Yarin Gal and Doina Precup},
    year={2020},
    booktitle={International Conference on Machine Learning (ICML)},
}
```

## Experiments

The three sets of experiments on model learning, imitation learning, and reinforcement learning can be found in their respective folder. To install requirements, create a new conda environment and run
```
pip install -e requirements.txt
```

In model learning, there are two sets of experiments, linear MISA and nonlinear MISA. The code is in `model_learning`. First `cd model_learning`.

The main experiment with linear MISA can be run with
```
ICPAbstractMDP.ipynb
```
The main experiment with nonlinear MISA can be run with
```
python main.py
```

For running the imitation learning experiments, first `cd imitation_learning`. Then install the `baselines` by running `cd baselines && pip install tensorflow==1.14 && pip install -e .` The main experiments can be run in `imitation_learning` directory with:

```
python train_expert.py --save_model --save_model_path models # Training the expert model

#Lets say the model was trained for 150K steps.

mkdir -p buffers/train/0 buffers/train/1 buffers/eval/0 # Directory to hold the buffer data

python collect_data_using_expert_policy.py --load_model_path models_150000 --save_buffer --save_buffer_path buffers  # Collecting the trajectories using the expert model

python train.py --use_single_encoder_decoder --num_train_envs 1 --num_eval_envs 1 --load_buffer_path buffers # MISA One Env

python train.py --use_single_encoder_decoder --num_train_envs 2 --num_eval_envs 1 --load_buffer_path buffers # Baseline One Decoder 

python train.py --use_discriminator --num_train_envs 2 --num_eval_envs 1 --load_buffer_path buffers # Proposed Approach

python train.py --use_irm_loss --num_train_envs 2 --num_eval_envs 1 --load_buffer_path buffers # IRM

```

In reinforcement learning, the main experiment can be run in `reinforcement_learning` directory with
```
./run_local.sh
```

# LICENSE

[Attribution-NonCommercial 4.0 International](/LICENSE)