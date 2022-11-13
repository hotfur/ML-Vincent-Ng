# HW3 Markov Decision Process
# Vuong Kha Sieu
# Date 2/11/2022

import sys


def main():
    train, gamma = sys.argv[3], float(sys.argv[4])
    with open(train) as file:
        data = file.read().splitlines()
    r, prob = read_input(data)
    solve_mdp(r, prob, gamma)


def read_input(data):
    prob = dict()
    r = []
    for state in data:
        action_dict = dict()
        while state.find("(") != -1:
            start = state.find("(")
            end = state.find(")")
            temp = state[start + 1:end].split()
            if temp[0] in action_dict:
                action_dict[temp[0]].append(temp[1:])
            else:
                action_dict[temp[0]] = list([temp[1:]])
            state = state[:start] + state[end + 1:]
        state_arr = state.split()
        r.append(float(state_arr[1]))
        prob[state_arr[0]] = action_dict
    return r, prob


def solve_mdp(r, prob, gamma):
    J = list(r)
    for i in range(1, 21):
        new_J = list(J)
        # Formatting
        if i != 1:
            print()
        print("After iteration " + str(i) + ":")
        for state in prob:
            state_dict = prob[state]
            max_gain = -99999
            best_action = ""
            for action in state_dict:
                gain = 0
                action_dict = state_dict[action]
                for trans in action_dict:
                    new_state = int(trans[0][1:]) - 1
                    gain += float(trans[1]) * gamma * J[new_state]
                if gain > max_gain:
                    max_gain = gain
                    best_action = action
            new_J[int(state[1:]) - 1] = max_gain + r[int(state[1:]) - 1]
            # Printer
            print("(" + state + " " + best_action + " " + str(round(J[int(state[1:]) - 1], 4)) + ") ", end="")
        # Update J
        J = new_J


if __name__ == "__main__":
    main()
