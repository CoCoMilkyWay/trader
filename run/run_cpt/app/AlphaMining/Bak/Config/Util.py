import numpy as np

def get_all_UNARY_ACTIONS(start_idx, end_idx, operator_tks, operand_tks, null_idx):
    action_list, token_action_list = [], []
    for i in range(start_idx, end_idx):
        operator_tk = operator_tks[i]
        for j, operand_tk in enumerate(operand_tks):
            # check validity of the action and add to the action_list
            pass
    return action_list, token_action_list


def initialize_UNARY_actions(operator_tks, operand_tks, null_idx, num_splits=1):
    num_operator_tokens = len(operator_tks)
    interval = num_operator_tokens / num_splits
    refs = []
    for i in range(num_splits):
        start_index, end_index = i*interval, (i+1)*interval
        start_index = int(np.floor(start_index))
        end_index = int(np.floor(end_index))
        if i == num_splits - 1:
            end_index = num_operator_tokens
        refs.append(get_all_UNARY_ACTIONS(
            start_index, end_index, operator_tks, operand_tks, null_idx))
    action_lists = refs
    unary_action_list, unary_token_action_list = [], []
    for action_list, token_action_list in action_lists:
        unary_action_list.extend(action_list)
        unary_token_action_list.extend(token_action_list)
    return unary_action_list, unary_token_action_list


def get_all_BINARY_ACTIONS(start_idx, end_idx, operator_tks, operand_tks, action_shift):
    action_list, token_action_list = [], []
    for i in range(start_idx, end_idx):
        operator_tk = operator_tks[i]
        for j, operand_tk1 in enumerate(operand_tks):
            for k, operand_tk2 in enumerate(operand_tks):
                # check validity of the action and add to the action_list
                pass
    return action_list, token_action_list


def initialize_BINARY_actions(operator_tks, operand_tks, action_shift,  num_splits=1):
    num_operator_tokens = len(operator_tks)
    interval = num_operator_tokens / num_splits
    refs = []
    for i in range(num_splits):
        start_index, end_index = i*interval, (i+1)*interval
        start_index = int(np.floor(start_index))
        end_index = int(np.floor(end_index))
        if i == num_splits - 1:
            end_index = num_operator_tokens
        refs.append(get_all_BINARY_ACTIONS(
            start_index, end_index, operator_tks, operand_tks, action_shift))
    action_lists = refs

    binary_action_list, binary_token_action_lists = [], []
    for action_list, token_action_list in action_lists:
        binary_action_list.extend(action_list)
        binary_token_action_lists.extend(token_action_list)
    return binary_action_list, binary_token_action_lists


def get_all_TERNARY_actions(start_idx, end_idx, operator_tks, operand_tks, action_shift):
    action_list, token_action_list = [], []
    for i in range(start_idx, end_idx):
        operator_tk = operator_tks[i]
        for j, operand_tk1 in enumerate(operand_tks):
            for k, operand_tk2 in enumerate(operand_tks):
                for l, operand_tk3 in enumerate(operand_tks):
                    # check validity of the action and add to the action_list
                    pass
    return action_list, token_action_list


def initialize_TERNARY_actions(operator_tks, operand_tks, action_shift,  num_splits=1):
    num_operator_tokens = len(operator_tks)
    interval = num_operator_tokens / num_splits
    refs = []
    for i in range(num_splits):
        start_index, end_index = i*interval, (i+1)*interval
        start_index = int(np.floor(start_index))
        end_index = int(np.floor(end_index))
        if i == num_splits - 1:
            end_index = num_operator_tokens
        refs.append(get_all_TERNARY_actions(
            start_index, end_index, operator_tks, operand_tks, action_shift))
    action_lists = refs

    ternary_action_list, ternary_token_action_lists = [], []
    for action_list, token_action_list in action_lists:
        ternary_action_list.extend(action_list)
        ternary_token_action_lists.extend(token_action_list)
    return ternary_action_list, ternary_token_action_lists
