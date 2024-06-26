import gym
import torch
from gym import spaces
from torch.distributions import Bernoulli

from TBWorldUtils import *
from PatientWorldUtils import *


class PatientWorld(gym.Env):
    """An environment for simulating disease treatment through MDP. Based off of TB Adherence"""

    def __init__(
        self,
        transition_prob=None,  # predictions for the transition matrix
        true_transition_prob=None,  # for forward compatability with TBBeliefWorld
        DATA_PATH=None,  # location of file with patient data
        NUM_STATES = 2,  # total states (min 2)
        NUM_MEDICINES = 4, # number
        # MIN_SEQ_LEN=30,  # Data pre-processing step
        # EFFECT_SIZE=0.2,  # size of action effect
        EPSILON=0.05,  # smoothing for transition probabilities
        # EPISODE_LEN=30,  # how many steps consist of an episode
        # START_ADHERING=1,  # whether patients start in an adhering state or not
    ):
        super(PatientWorld, self).__init__()

        #
        self.NUM_MEDICINES = NUM_MEDICINES

        # DEFINE STATE SPACE
        self.observation_space = spaces.Box(low=0, high=NUM_STATES, dtype=int)

        # DEFINE ACTION SPACE
        self.action_space = spaces.Discrete(NUM_MEDICINES)

        # DEFINE TRANSITIONS
        # patient_models = [current_state][sickness][current_action][next_state]
        # Note sickness is the belief sickness?
        self.true_patient_models = torch.from_numpy(create_transition_matrix(NUM_MEDICINES - 1))
        # # patient_models[patient][action][current_state][next_state] --> probability of event
        # if true_transition_prob is not None:
        #     self.true_patient_models = true_transition_prob
        # elif transition_prob is not None:
        #     self.true_patient_models = transition_prob
        # else:
        #     if DATA_PATH is not None:
        #         # Generate raw transition matrices from patient data
        #         T_matrices = generate_T_matrices_from_data(DATA_PATH, MIN_SEQ_LEN)
        #     else:
        #         # Generate random matrices
        #         T_matrices = generate_T_matrices_random()
        #
        #     # Add action effects to transitions
        #     patient_models = generate_action_effects(T_matrices, NUM_PATIENTS, EFFECT_SIZE, EPSILON)
        #     self.true_patient_models = torch.from_numpy(patient_models)

        # Initialise internal variables
        self.reset()

    def reset(self):
        """
        Reset the position to a starting position.
        Starts with either all adhering or not...
        """
        self.current_step = 0
        # In this case only one patient so state is a single value
        self.state = torch.zeros(1, dtype=torch.int)
        return self.observe()

    def is_terminal(self):
        """Check if the given state is a terminal state."""
        return self.state.item() == self.observation_space.high.item() - 1

    def observe(self):
        """This is the MDP version so just return state (i.e. full observability)"""
        return self.state

    # def _get_action_onehot(self, action):
    #     if not torch.is_tensor(action):
    #         action = torch.tensor(action)
    #     action_onehot = torch.nn.functional.one_hot(action, num_classes=self.NUM_MEDICINES).flatten()
    #
    #     return action

    def _get_next_state(self, action):
        # Get the next state
        # next_state_probabilities = torch.cat([self.true_patient_models[i, action_onehot[i], self.state[i]].view(1, -1) for i in range(self.NUM_MEDICINES)], dim=0)
        # next_state_probabilities = torch.tensor([self.true_patient_models[self.state[i], medicine, action_onehot[i]] for i in range(self.NUM_MEDICINES)])

        next_state_probabilities = torch.cat(
            [self.true_patient_models[self.state.item(), sickness, action] for sickness in
             range(self.NUM_MEDICINES - 1)], dim = 0).view(1,-1)
        # print("n_s_p MDP: ", next_state_probabilities.shape)

        distribution = Bernoulli(probs=next_state_probabilities[:, 1])
        next_state = distribution.sample().to(torch.int64).detach()

        # Get the relevant transition probabilities
        logprob = torch.log(torch.gather(next_state_probabilities, 1, next_state.view(-1, 1))).sum()

        return next_state, logprob

    def step(self, action):
        """Perform an action, yielding a new state and reward."""

        # Get next state
        next_state, logprob = self._get_next_state(action)
        # print("logprob: ", logprob)

        # Update relevant variables
        reward = next_state.sum().item()
        self.state = next_state
        self.current_step += 1

        return next_state, reward, self.is_terminal(), {'logprob': logprob}

    @staticmethod
    def loss_fn(trajectories, transition_prob):
        N = len(trajectories)
        T = len(trajectories[0])

        # Get data
        states = torch.cat([trajectories[n][t][0].view(1, -1) for n in range(N) for t in range(T)], dim=0).to(torch.int64)
        next_states = torch.cat([trajectories[n][t][3].view(1, -1) for n in range(N) for t in range(T)], dim=0).to(torch.int64)
        actions = torch.cat([trajectories[n][t][1] for n in range(N) for t in range(T)], dim=0)

        NUM_PATIENTS = states.shape[-1]
        actions_onehot = torch.nn.functional.one_hot(actions, num_classes=NUM_PATIENTS)  # to one-hot

        # Get relevant probabilities
        likelihoods = torch.cat([transition_prob[torch.full((N * T,), i, dtype=torch.int64), actions_onehot[:, i], states[:, i], next_states[:, i]].view(-1, 1) for i in range(NUM_PATIENTS)], dim=1)
        NLL = (-torch.log(likelihoods)).sum() / N
        return NLL


class PatientBeliefWorld(PatientWorld):
    """A version of TBWorld with partial observability.
    Instead of operating on the true state, we return the
    `belief state' as the observation."""

    def __init__(
        self,
        transition_prob=None,  # transition matrix for updating beliefs
        true_transition_prob=None,  # transition matrix for updating the true state
        DATA_PATH=None,  # location of file with patient data
        # NUM_PATIENTS=20,  # number of patients to sample
        # MIN_SEQ_LEN=30,  # Data pre-processing step
        NUM_STATES=2,  # total states (min 2)
        NUM_MEDICINES=4,  # number
        # EFFECT_SIZE=0.2,  # size of action effect
        EPSILON=0.05,  # smoothing for transition probabilities
        # EPISODE_LEN=30,  # how many steps consist of an episode
        # START_ADHERING=1,  # whether patients start in an adhering state or not
    ):
        # Load the info using TBWorld's constructor
        super(PatientBeliefWorld, self).__init__(transition_prob, true_transition_prob, DATA_PATH, NUM_STATES, NUM_MEDICINES, EPSILON)

        # Dealing with `true_transition_prob' and `true_transition_prob'
        if true_transition_prob is not None:
            assert transition_prob is not None
            self.patient_models = transition_prob.clone()
        else:
            self.patient_models = self.true_patient_models.clone()

    def reset(self):
        """
        Reset the position to a starting position.
        """
        self.current_step = 0
        # In this case only one patient so state is a single value
        self.state = torch.zeros(1, dtype=torch.int)
        self.belief_state = self.state.clone().to(torch.double)
        return self.observe()

    def observe(self):
        """
        Return belief state as the observation (instead of state)
        """
        return self.belief_state

    def step(self, action):
        """Perform an action, yielding a new state and reward."""
        # # Parse Action
        # action_onehot = self._get_action_onehot(action)

        # Get next state
        next_state, _ = self._get_next_state(action)

        # Get Next Belief State
        #   For patients not acted on, calculate expected next state
        # next_state_probabilities = self.patient_models[torch.arange(self.NUM_PATIENTS), action_onehot]
        # TODO shape of belief state not working
        next_state_probabilities = torch.cat(
            [self.true_patient_models[self.state.item(), sickness, action] for sickness in
             range(self.NUM_MEDICINES - 1)], dim = 0).view(1,-1)
        # print('n_s_p: ', next_state_probabilities[1 , 1])
        next_belief_state = self.belief_state * next_state_probabilities[:, 1, 1] + (1 - self.belief_state) * next_state_probabilities[:, 0, 1]

        #   For patients acted on, collapse the belief state
        #   Note: This is slightly different from Jackson and Aditya's implementation
        observed_next_acted_state = next_state[action].to(next_belief_state.dtype)
        logprob = next_belief_state[action] * observed_next_acted_state + (1 - next_belief_state[action]) * (1 - observed_next_acted_state)
        next_belief_state[action] = observed_next_acted_state

        # Update relevant variables
        # Is this the right reward? Should it be next_state.sum() or next_belief_state.sum()? How would the logprobs change?
        reward = next_state.sum().item()
        self.state = next_state
        self.belief_state = next_belief_state
        self.current_step += 1

        return next_belief_state, reward, self.is_terminal(), {'logprob': logprob}

    @staticmethod
    def loss_fn(trajectories, transition_prob):
        # Get data
        N = len(trajectories)
        T = len(trajectories[0])
        states = torch.cat([trajectories[n][t][0].view(1, -1) for n in range(N) for t in range(T)], dim=0)
        next_states = torch.cat([trajectories[n][t][3].view(1, -1) for n in range(N) for t in range(T)], dim=0)
        actions = torch.cat([trajectories[n][t][1] for n in range(N) for t in range(T)], dim=0).unsqueeze(-1)

        # Get relevant states
        relevant_states = torch.gather(states, 1, actions)
        relevant_next_states = torch.gather(next_states, 1, actions)

        # Get likelihood
        # TODO: Check why the likelihoods definition was messing with backprop
        pr_next_state_1 = relevant_states * transition_prob[actions, 1, 1, 1] + (1 - relevant_states) * transition_prob[actions, 1, 0, 1]
        likelihoods = torch.where(torch.round(relevant_next_states) == 1, pr_next_state_1, 1 - pr_next_state_1)
        NLL = (-torch.log(likelihoods)).sum() / N

        return NLL


if __name__ == '__main__':
    import random

    # Test Environments
    envs = [PatientWorld()]#  , PatientBeliefWorld()]

    for env in envs:
        print(f"\nTesting {env.__class__.__name__}:")

        obs = env.reset()
        done = False
        while not done:
            action = random.randrange(env.NUM_MEDICINES)
            next_obs, reward, done, _ = env.step(action)
            print(f"State: {obs.tolist()}, Action: {action}, Reward: {reward}, Next State: {next_obs.tolist()}")

            obs = next_obs
