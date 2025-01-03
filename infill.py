
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


# The get_motif function of this code is highly motivated by EvoDiff:
# https://github.com/microsoft/evodiff

import argparse
import os
import random
from pathlib import Path
from pprint import pprint
import numpy as np
import torch
from byprot import utils
from byprot.models.lm.dplm import DiffusionProteinLanguageModel
from byprot.datamodules.dataset.data_utils import PDBDataProcessor
from copy import deepcopy
import esm
# import esm.inverse_folding
import pandas as pd

single_res = ['1qjg']

start_idx_dict = {
    '1prw': [15, 51],
    # '1bcf': [17, 46, 90, 122],
    # '5tpn': [108],
    # '3ixt': [0],
    # '4jhw': [37, 144],
    # '4zyp': [357],
    # '5wn9': [1],
    # '5ius': [34, 88],
    # '5yui': [89, 114, 194],
    # '6vw1': [5, 45],
    # '1qjg': [13, 37, 98],
    # '1ycr': [2],
    # '2kl8': [0, 27],
    # '7mrx': [25],
    # '5trv': [45],
    # '6e6r': [22],
    # '6exz': [25],
}

end_idx_dict = {
    '1prw': [34, 70],
    # '1bcf': [24, 53, 98, 129],
    # '5tpn': [126],
    # '3ixt': [23],
    # '4jhw': [43, 159],
    # '4zyp': [371],
    # '5wn9': [20],
    # '5ius': [53, 109],
    # '5yui': [93, 116, 196],
    # '6vw1': [23, 63],
    # '1qjg': [13, 37, 98],
    # '1ycr': [10],
    # '2kl8': [6, 78],
    # '7mrx': [46],
    # '5trv': [69],
    # '6e6r': [34],
    # '6exz': [39],
}

chain_dict = {
    '1prw': 'A',
    # '1bcf': 'A',
    # '5tpn': 'A',
    # '3ixt': 'P',
    # '4jhw': 'F',
    # '4zyp': 'A',
    # '5wn9': 'A',
    # '5ius': 'A',
    # '5yui': 'A',
    # '6vw1': 'A',
    # '1qjg': 'A',
    # '1ycr': 'B',
    # '2kl8': 'A',
    # '7mrx': 'B',
    # '5trv': 'A',
    # '6e6r': 'A',
    # '6exz': 'A',
}

    
def prepare_data(pdb_path, alphabet, collator, num_seqs, device):
    def _full_mask(target_tokens, coord_mask, alphabet):
        target_mask = (
            target_tokens.ne(alphabet.padding_idx)  # & mask
            & target_tokens.ne(alphabet.cls_idx)
            & target_tokens.ne(alphabet.eos_idx)
        )
        _tokens = target_tokens.masked_fill(
            target_mask, alphabet.mask_idx
        )
        _mask = _tokens.eq(alphabet.mask_idx) & coord_mask
        return _tokens, _mask
    
    pdb_id = Path(pdb_path).stem
    structure = PDBDataProcessor().parse_PDB(pdb_path)
    batch = collator(
        [
            deepcopy(structure) for idx in range(num_seqs)
        ]
    )
    prev_tokens, prev_token_mask = _full_mask(
        batch['tokens'], batch['coord_mask'], alphabet
    )
    batch['prev_tokens'] = prev_tokens
    batch['prev_token_mask'] = prev_tokens.eq(alphabet.mask_idx)
    batch = utils.recursive_to(batch, device=device)
    return batch, structure['seq']

def get_intervals(list, single_res_domain=False):
    "Given a list (Tensor) of non-masked residues get new start and end index for motif placed in scaffold"
    if single_res_domain:
        start = [l.item() for l in list]
        stop = start
    else:
        start = []
        stop = []
        for i, item in enumerate(list):
            if i == 0:
                start.append(item.item())
            elif i == (len(list)-1):
                stop.append(item.item())
            elif i != len(list) and (item+1) != list[i+1]:
                stop.append(item.item())
                start.append(list[i+1].item())
    return start, stop


def get_motif(PDB_ID):
    # Get motif of sequence from PDB code
    start_idxs = start_idx_dict[PDB_ID]
    end_idxs = end_idx_dict[PDB_ID]
    pdb_clean_path = os.path.join('data-bin/scaffolding-pdbs/' + str(PDB_ID) + '_clean.pdb')

    chain = chain_dict[PDB_ID]
    chain_ids = [chain]
    print("WARNING: USING CHAIN", chain, "FROM PDB FILE")
    structure = esm.inverse_folding.util.load_structure(pdb_clean_path, chain_ids)
    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
    sequence = native_seqs[chain_ids[0]]
    print("sequence extracted from pdb", sequence)
    if not os.path.isfile('data-bin/scaffolding-pdbs/'+ PDB_ID +'.fasta'):
        with open('data-bin/scaffolding-pdbs/'+ PDB_ID +'.fasta', 'a') as f:
            f.write('>' + PDB_ID+'\n'+sequence)
    print("sequence length", len(sequence))
    assert len(start_idxs) == len(end_idxs)

    end_idxs = [i+1 for i in end_idxs] # inclusive of final residue
    if len(start_idxs) > 1:
        motif = ''
        spacers = []
        
        for i in range(len(start_idxs)):
            motif += sequence[start_idxs[i]:end_idxs[i]]
            if i < (len(start_idxs)-1):
                spacer = start_idxs[i+1] - end_idxs[i]
                motif += '<mask>' * spacer
                spacers.append(spacer)
    else:
        motif = sequence[start_idxs[0]: end_idxs[0]]
        spacers=[0]
    print("motif extracted from indexes supplied:", motif)
    
    # pdb_motif_path = os.path.join('data-bin/scaffolding-pdbs/' + str(PDB_ID) + '_motif.pdb')
    # structure_motif = esm.inverse_folding.util.load_structure(pdb_motif_path, chain_ids)
    # _, native_seqs_motif = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure_motif)
    # motif_pdb = native_seqs_motif[chain_ids[0]]
    # print("motif extracted from motif.pdb:", motif_pdb)
    return motif


def get_initial(args, tokenizer, linker_length, device):
    num = args.num_seqs
    
    # endos2
    # left = 'MDKHLLVKRTLGCVCAATLMGAALATHHDSLNTVKAEEKTVQTGKTDQQVGAKLVQEIREGKRGPLYAGYFRTWHDRASTGIDGKQQHPENTMAEVPKEVDILFVFHDHTASDSPFWSELKDSYVHKLHQQGTALVQTIGVNELNGRTGLSKDYPDTPEGNKALAAAIVKAFVTDRGVDGLDIDIEHEFTNKRTPEEDARALNVFKEIAQLIGKNGSDKSKLLIMDTTLSVENNPIFKGIAEDLDYLLRQYYGSQGGEAEVDTINSDWNQYQNYIDASQFMIGFSFFEESASKGNLWFDVNEYDPNNPEKGKDIEGTRAKKYAEWQPSTGGLKAGIFSYAIDRDGVAHVPSTYKNRTSTNLQRHEVDNISHTDYTVSRKLKTLMTE'
    # right = 'LAKGAKVIGTSGDFEQAKKIFDGEKSDRFFTWGQTNWIAFDLGEINLAKEWRLFNAETNTEIKTDSSLNVAKGRLQILKDTTIDLEKMDIKNRKEYLSNDENWTDVAQMDDAKAIFNSKLSNVLSRYWRFCVDGGASSYYPQYTELQILGQRLSNDVANTLKD'
    
    # trastuzumab
    left = 'EVQLVESGGGLVQPGGSLRLSCAASGFNIKEYYMHWVRQAPGKGLEWVGLIDPEQGNTIYDPKFQDRATISADNSKNTAYLQMNSLRAEDTAVYYCAR'
    right = 'WGQGTLVTVS'
    
    mask = tokenizer.mask_token_id
    bos = tokenizer.cls_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    
    init_seq = []
    scaffold_length_list = []
    for i in range(num):
        scaffold_length_list.append(linker_length)
        seq = list(left) + ['<mask>'] * linker_length + list(right)
        seq = ''.join(seq)
        init_seq.append(seq)
    
    batch = tokenizer.batch_encode_plus(init_seq,
                                add_special_tokens=True,
                                padding="longest",
                                return_tensors='pt')
    batch = {
        'input_ids':  batch['input_ids'],
        'input_mask': batch['attention_mask'].bool(),
    }
    batch = utils.recursive_to(batch, device)
    
    start_idxs_list = []
    end_idxs_list = []
    for seq in batch['input_ids']:
        nonmask_locations = ((seq != mask) & (seq != bos) & (seq != eos) & (seq != pad)).nonzero().flatten() - 1 
        new_start_idxs, new_end_idxs = get_intervals(nonmask_locations, False)
        start_idxs_list.append(new_start_idxs)
        end_idxs_list.append(new_end_idxs)
    pprint(batch)
    # print(start_idxs_list)
    # print(end_idxs_list)
    return batch, start_idxs_list, end_idxs_list, scaffold_length_list


def generate(args, saveto):
    model = DiffusionProteinLanguageModel.from_pretrained(args.model_name, from_huggingface=True)
    tokenizer = model.tokenizer
    model = model.eval()
    model = model.cuda(); 
    device = next(model.parameters()).device

    # Generate
    max_iter = args.max_iter
    batch, start_idxs_list, end_idxs_list, scaffold_lengths_list = get_initial(args, tokenizer, args.linker_length, device)
    partial_mask = (batch['input_ids'].ne(tokenizer.mask_token_id) & \
                batch['input_ids'].ne(tokenizer.pad_token_id)).type_as(batch['input_mask'])
    
    with torch.cuda.amp.autocast():
        outputs = model.generate(
            batch=batch, 
            temperature=args.temperature,
            max_iter=max_iter,
            sampling_strategy=args.sampling_strategy,
            partial_masks=partial_mask
        )
    output_tokens = outputs[0]
    
    print('final:')
    output_results = [''.join(seq.split(' ')) for seq in tokenizer.batch_decode(output_tokens, skip_special_tokens=True)]
    pprint(output_results)
    
    # save output
    out_path = os.path.join(saveto, 'endo_s2')
    os.makedirs(out_path, exist_ok=True)
    saveto_name = os.path.join(out_path, f'{args.linker_length}.fasta')
    fp_save = open(saveto_name, 'w')
    for idx, seq in enumerate( 
        output_results
    ):
        fp_save.write(f">seq_{idx}\n")
        fp_save.write(f"{seq}\n")
    fp_save.close()
    
    # scaffold_info_path = os.path.join(saveto, 'scaffold_info')
    # os.makedirs(scaffold_info_path, exist_ok=True)
    # strings = output_results
    # save_df = pd.DataFrame(list(zip(strings, start_idxs_list, end_idxs_list, scaffold_lengths_list)), columns=['seqs', 'start_idxs', 'end_idxs', 'scaffold_lengths'])
    # save_df.to_csv(os.path.join(scaffold_info_path, f'{pdb}.csv'), index=True)

    
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='airkingbd/dplm_150m')
    parser.add_argument('--num_seqs', type=int, default=40)
    parser.add_argument('--linker_length', type=int, default=8)
    parser.add_argument('--saveto', type=str, default='gen.fasta')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sampling_strategy', type=str, default='gumbel_argmax')
    parser.add_argument('--max_iter', type=int, default=500)

    
    args = parser.parse_args()
    pprint(args)

    generate(args, args.saveto)
    

if __name__ == '__main__':
    main()
