from typing import List
import torch
import utility_functions as utility_fn

def mbr_decode(n_best: int, all_hyps: List[List[str]], utility_function: utility_fn.BaseFunction):
    cands = []
    refs = []
    for utt_hyps in all_hyps:
        for hyp_i_pos in range(n_best):
            hyp_i = [utt_hyps[hyp_i_pos]] * (n_best - 1)
            other_hyps = utt_hyps[:hyp_i_pos] + utt_hyps[hyp_i_pos+1:n_best]
            cands += hyp_i
            refs += other_hyps

    scores = utility_function.score(cands, refs)
    # score size: utt_num * n_best * (n_best - 1)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float)
    scores = scores.reshape(len(all_hyps), n_best, n_best-1)

    scores = scores.sum(dim=-1)
    max_value_index = scores.argmax(dim=-1).cpu()

    predictions = [
        all_hyps[utt_id][v_index] 
        for utt_id, v_index in enumerate(max_value_index)
    ]
    return predictions, scores