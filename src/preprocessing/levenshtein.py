import sys
from typing import List
import numpy as np

def levenshtein_distance_alignment(reference: List[str] or str, hypthesis: List[str] or str):
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


    aligned_ref = ""
    aligned_hyp = ""
    aligned_op = ""

    i = len_hyp - 1
    j = len_ref - 1
    while i >= 1 or j >= 1:
        if operation_matrix[i][j] == "U":
            aligned_ref += reference[j]
            aligned_hyp += hypthesis[i]
            aligned_op += "U"
            i -= 1
            j -= 1
        
        elif operation_matrix[i][j] == "S":
            aligned_ref += reference[j]
            aligned_hyp += hypthesis[i]
            aligned_op += "S"
            i -= 1
            j -= 1
        
        elif operation_matrix[i][j] == "D":
            aligned_ref += "*"
            aligned_hyp += hypthesis[i]
            aligned_op += "D"
            i -= 1
        
        elif operation_matrix[i][j] == "I":
            aligned_ref += reference[j]
            aligned_hyp += "*"
            aligned_op += "I"
            j -= 1

    output = []
    for string in [aligned_ref, aligned_hyp, aligned_op]:
        reversed_string=''.join(reversed(string))
        output.append(reversed_string)

    return output