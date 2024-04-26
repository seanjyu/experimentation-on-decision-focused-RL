import numpy as np
import torch

def create_transition_matrix(num_sicknesses, medicine_effectiveness = 0.9, num_states = 2):
    # Define the number of states (sick, not sick) and actions (do nothing, take medicine)
    num_actions = num_sicknesses + 1

    # Define transition probabilities
    # TransitionProb[current_state][current_sickness][current_action][next_state]
    transition_probabilities = np.zeros((num_states - 1, num_sicknesses, num_actions, num_states))

    # Populate transition probabilities
    # In the following case only have one state (sick) if the correct action is selected then patient will become not sick
    for sickness in range(num_sicknesses):
        # TODO adapt this for more states
        transition_probabilities[0][sickness][sickness + 1][1] = medicine_effectiveness
        transition_probabilities[0][sickness][sickness + 1][0] = 1 - medicine_effectiveness

        # If the current state is "sick" and the action is doing nothing, stay in "sick" state with probability 0.9
        transition_probabilities[0][sickness][0][0] = 0.95
        transition_probabilities[0][sickness][0][1] = 0.05

        # # If the current state is "not sick", both actions lead to staying in "not sick" state with probability 1
        # transition_probabilities[1][sickness][0][1] = 1.0
        # transition_probabilities[1][sickness][sickness + 1][1] = 1.0

        # fill in other values (use wrong medicine for disease)
        for i in range(1, num_actions):
            if i != sickness + 1:
                transition_probabilities[0][sickness][i][1] = 0.1
                transition_probabilities[0][sickness][i][0] = 0.9

    return np.array(transition_probabilities)

if __name__ == '__main__':
    # Test if it works
    num_sicknesses = 4
    transition_matrix = create_transition_matrix(num_sicknesses)

    # Print transition probability matrix
    print("Transition Probability Matrix:")
    print(transition_matrix)
