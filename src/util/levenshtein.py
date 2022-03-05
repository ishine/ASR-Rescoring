import sys
from typing import List
import numpy as np

def levenshtein_distance_alignment(hypthesis: List[str] or str, reference: List[str] or str):
    '''
    What kind of operation that make hypothesis sentence become reference sentence.

    "U": unchange
    "S": substitution
    "I": Insertion
    "D": Deletion

    '''
    if isinstance(hypthesis, str):
        hypthesis = list(hypthesis)
    if isinstance(reference, str):
        reference = list(reference)

    hypthesis = ["[start]"] + hypthesis
    reference = ["[start]"] + reference

    len_hyp = len(hypthesis)
    len_ref = len(reference)

    cost_matrix = np.zeros((len_hyp, len_ref))
    operation_matrix = np.full((len_hyp, len_ref), "U")

    for i in range(1, len_hyp):
        cost_matrix[i][0] = i
        operation_matrix[i][0] = "D"
    for j in range(1, len_ref):
        cost_matrix[0][j] = j
        operation_matrix[0][j] = "I"

    for i in range(1, len_hyp):
        for j in range(1, len_ref):
            if hypthesis[i] == reference[j]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution_cost = cost_matrix[i-1][j-1] + 1
                insertion_cost = cost_matrix[i][j-1] + 1
                deletion_cost = cost_matrix[i-1][j] + 1

                priority = {"S": substitution_cost, "I": insertion_cost, "D": deletion_cost}

                min_cost = priority["S"]
                op_id = "S"
                for operation_id, operation_cost in priority.items():
                    if operation_cost < min_cost:
                        min_cost = operation_cost
                        op_id = operation_id

                cost_matrix[i][j] = min_cost
                operation_matrix[i][j] = op_id


    output = []
    i = len_hyp - 1
    j = len_ref - 1
    while i >= 1 or j >= 1:
        if operation_matrix[i][j] == "U":
            output.append((reference[j], hypthesis[i], "U"))
            i -= 1
            j -= 1
        
        elif operation_matrix[i][j] == "S":
            output.append((reference[j], hypthesis[i], "S"))
            i -= 1
            j -= 1
        
        elif operation_matrix[i][j] == "D":
            output.append(("*", hypthesis[i], "D"))
            i -= 1
        
        elif operation_matrix[i][j] == "I":
            output.append((reference[j], "*", "I"))
            j -= 1

    output.reverse()
    
    return output