from typing import List, Union, Any, Dict

import einops
import torch
import torch.nn.functional as F
import transformers as tfs
from transformers.tokenization_utils_base import BatchEncoding
from transformers.data.data_collator import _torch_collate_batch, tolist

import tensor_utils


class ConversationDataCollator:
    def __init__(
        self,
        #tokenizer,
        label_name):

        #self.tokenizer = tokenizer
        self.label_name = label_name

    def __call__(self, features):
        first = features[0]
        keys = set(first.keys())

        batch = {}
        if 'example_id' in first:
            example_ids = [f['example_id'] for f in features]
            batch['example_ids'] = torch.stack(example_ids)

        if self.label_name in keys:
            keys.remove(self.label_name)
            labels = [f[self.label_name] for f in features]
            batch['labels'] = torch.stack(labels)

        max_seq_len = max([len(f['input_ids']) for f in features])
        for k in keys:
            if k in ('example_id', 'labels'):
                continue

            to_batch = []
            for f in features:
                t = torch.stack(f[k])
                seq_len = t.size(0)
                pad_len = max_seq_len - seq_len
                t = F.pad(t, (0, 0, 0, pad_len))
                to_batch.append(t)
            batch[k] = torch.stack(to_batch)
        return batch

        
class DataCollatorForWholeWordMaskAndWholeSentenceMask(tfs.DataCollatorForWholeWordMask):
    def __init__(
        self,
        tokenizer,
        mlm_probability,
        msm_probability,
        max_num_turns,
        min_num_valid_tokens=None,
        mask_whole_sentence=True,
        same_mask=False):
        super().__init__(tokenizer, mlm_probability)
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.msm_probability = msm_probability
        self.max_num_turns = max_num_turns
        self.min_num_valid_tokens = min_num_valid_tokens
        self.mask_whole_sentence = mask_whole_sentence
        self.same_mask = same_mask

    
    def mlm_torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        if self.same_mask and hasattr(self, "batch_mask"):
            batch_mask = self.batch_mask 
        else:
            mask_labels = []
            for e in examples:
                ref_tokens = []
                for id in tolist(e["input_ids"]):
                    token = self.tokenizer._convert_id_to_token(id)
                    ref_tokens.append(token)

                # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
                if "chinese_ref" in e:
                    ref_pos = tolist(e["chinese_ref"])
                    len_seq = len(e["input_ids"])
                    for i in range(len_seq):
                        if i in ref_pos:
                            ref_tokens[i] = "##" + ref_tokens[i]
                mask_labels.append(self._whole_word_mask(ref_tokens))
            batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            self.batch_mask = batch_mask
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)

        return {"input_ids": inputs, "labels": labels}
    

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        #dialogue_idxs = [ex['dialogue_idx'] for ex in examples]
        #dialogue_idx_to_len = collections.Counter(dialogue_idxs)
        #max_num_turns = max(dialogue_idx_to_len.values())
        #num_dialogues = len(dialogue_idx_to_len)

        #dialogue_lens = torch.tensor(list(
        #    map(dialogue_idx_to_len.get, sorted(dialogue_idx_to_len))))
        #turn_mask = torch.arange(max_num_turns) < dialogue_lens.view(-1, 1)

        dialogue_lens = torch.tensor(
            [len(ex['input_ids']) for ex in examples], dtype=torch.long)
        #max_num_turns = dialogue_lens.max()
        num_dialogues = len(examples)
        turn_mask = torch.arange(self.max_num_turns) < dialogue_lens.view(-1, 1)

        # target_sentence_idxs shape = (nd, mnt).
        # sentence_masked_indices shape = (nd, mnt).
        target_sentence_idxs, sentence_masked_indices = \
            self._mask_sentences(num_dialogues, self.max_num_turns, ~turn_mask)

        def pad_turns(tensor_list, pad_value):
            #tensor_list = torch.tensor_split(tensor, dialogue_lens[:-1])
            tensor_list = [
                tensor_utils.pad_tensor(
                    torch.stack(x), self.max_num_turns, pad_value, dim=0)
                        for x in tensor_list]
            return torch.cat(tensor_list)

        input_ids_list = [ex['input_ids'] for ex in examples]
        attention_mask_list = [ex['attention_mask'] for ex in examples]
        input_ids = pad_turns(input_ids_list, self.tokenizer.pad_token_id)
        attention_mask = pad_turns(attention_mask_list, 0)

        # Makes sure we get decoder_labels from input_ids before MLM.
        decoder_labels = input_ids[target_sentence_idxs.masked_fill(
                target_sentence_idxs == -100, 0)]

        input_ids = [x.view(-1) for x in torch.split(input_ids, 1)]
        attention_mask = [x.view(-1) for x in torch.split(attention_mask, 1)]
        examples = []
        for inp_ids, attn_mask in zip(input_ids, attention_mask):
            examples.append({'input_ids': inp_ids, 'attention_mask': attn_mask})

        # Applies WWM.
        #ret = super().torch_call(examples)
        ret = self.mlm_torch_call(examples)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True) for val in input_ids
        ]
        special_tokens_mask = torch.tensor(
            special_tokens_mask, dtype=torch.bool)
        special_tokens_mask = einops.rearrange(
            special_tokens_mask,
            '(bs mnt) nt -> bs mnt nt',
            mnt=self.max_num_turns)

        input_ids = ret['input_ids']
        labels = ret.pop('labels')
        attention_mask = input_ids != self.tokenizer.pad_token_id

        attention_mask = einops.rearrange(
            attention_mask, '(bs mnt) nt -> bs mnt nt', mnt=self.max_num_turns)
        labels = einops.rearrange(
            labels, '(bs mnt) nt -> bs mnt nt', mnt=self.max_num_turns)
        #input_ids = pad_turns(input_ids, self.tokenizer.pad_token_id)
        #labels = pad_turns(labels, -100)
        #attention_mask = pad_turns(attention_mask, 0)
        #special_tokens_mask = pad_turns(special_tokens_mask, True)

        #print(sentence_masked_indices)

        if self.min_num_valid_tokens:
            # Only makes long sentences as targets.
            # num_valid_tokens shape = (nd, mnt).
            num_valid_tokens = attention_mask.sum(2)
            short_sentence_mask = num_valid_tokens < self.min_num_valid_tokens
            target_sentence_idxs = target_sentence_idxs.masked_fill(
                short_sentence_mask, -100)
            sentence_masked_indices = sentence_masked_indices.masked_fill(
                short_sentence_mask, 0)

            #short_sentence_mask = short_sentence_mask.unsqueeze(-1)
            #labels = labels.masked_fill(short_sentence_mask, -100)

        #print('@'*50)
        #print(sentence_masked_indices)

        #input_ids = torch.stack(sum([ex['input_ids'] for ex in examples], []))
        num_tokens = input_ids.size(-1)
        #decoder_labels = input_ids[target_sentence_idxs.masked_fill(
        #        target_sentence_idxs == -100, 0)]
        helper = einops.repeat(
            target_sentence_idxs, 'nd mnt -> nd mnt nt', nt=num_tokens)
        #print(target_sentence_idxs)
        #a = target_sentence_idxs[target_sentence_idxs != -100]
        #print(a)
        #print(f"{self.tokenizer.decode(decoder_labels[0][a[0]]) = }")

        # decoder_input_ids and decoder_labels are almost the same except 
        # decoder_labels uses -100 for padding.
        decoder_input_ids = decoder_labels.masked_fill(helper == -100, 0)
        #print(f"{self.tokenizer.decode(decoder_input_ids[0][a[0]]) = }")
        decoder_attention_mask = (decoder_input_ids != 0).long()
        decoder_labels = decoder_labels.masked_fill(helper == -100, -100) 
        decoder_labels = decoder_labels.masked_fill(
            decoder_labels == self.tokenizer.pad_token_id, -100)
        
        input_ids = einops.rearrange(
            input_ids, '(bs mnt) nt -> bs mnt nt', mnt=self.max_num_turns)

        if self.mask_whole_sentence:
            # Expands to token dim.
            expanded_sentence_masked_indices = einops.repeat(
                sentence_masked_indices, 'nd mnt -> nd mnt nt', nt=num_tokens)

            # Masks out entire sentences but keeping [CLS] and [SEP].
            expanded_sentence_masked_indices = \
                expanded_sentence_masked_indices & (~special_tokens_mask)
            input_ids = input_ids.masked_fill(
                expanded_sentence_masked_indices, self.tokenizer.mask_token_id)

        ret['input_ids'] = input_ids
        ret['mlm_labels'] = labels
        ret['attention_mask'] = attention_mask
        #ret['sentence_labels'] = sentence_labels
        #ret['decoder_input_ids'] = decoder_input_ids
        #ret['decoder_attention_mask'] = decoder_attention_mask
        #ret['labels'] = decoder_labels
        #ret['sentence_masked_indices'] = sentence_masked_indices
        return ret

    def _mask_sentences(self, num_dialogues, max_num_sentences, special_tokens_mask):
        target_sentence_indices = torch.arange(num_dialogues * max_num_sentences)
        target_sentence_indices = target_sentence_indices.view(num_dialogues, max_num_sentences)
        shape = target_sentence_indices.shape
        
        probability_matrix = torch.full(shape, self.msm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 80% of the time, we keep generating the original masked sentence.
        indices_replaced = torch.bernoulli(torch.full(shape, 0.8)).bool() & masked_indices
        
        # 10% of the time, we generate a random sentence in the batch for the masked sentence.
        indices_random = torch.bernoulli(
            torch.full(shape, 0.5)).bool() & masked_indices & ~indices_replaced
        valid_target_sentence_indices = \
            torch.where(special_tokens_mask.view(-1) == 0)[0]
        # Random samples sentences in the batch.
        p = torch.ones(
            len(valid_target_sentence_indices)) / len(valid_target_sentence_indices)
        num_samples = indices_random.sum().tolist()
        if num_samples > 0:
            random_target_sentence_indices = torch.multinomial(p, num_samples=num_samples)
            target_sentence_indices[indices_random] = random_target_sentence_indices
        
        target_sentence_indices.masked_fill_(~masked_indices, -100)
        masked_indices = masked_indices & (indices_replaced | indices_random)
        return target_sentence_indices, masked_indices