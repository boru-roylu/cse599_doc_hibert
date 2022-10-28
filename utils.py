import collections
import einops
import numpy as np
import os
import torch
import transformers as tfs
import tqdm

import comm
import modeling_bert
import tensor_utils


def obj_to_device(obj, device):
    # only put tensor into GPU
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    elif isinstance(obj, (list, tuple)):
        obj = list(obj)
        for v_i, v in enumerate(obj):
            obj[v_i] = obj_to_device(v, device)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = obj_to_device(v, device)
    return obj


def get_preprocess_fn(tokenizer, max_seq_length=512, max_num_sections=32):
    def preprocess(examples):
        docs = examples['document']

        num_sections = [len(d[:max_num_sections]) for d in docs]

        sentences = [
            f"{section['title']} {section['text']}"
            for doc in docs for section in doc[:max_num_sections]]

        inputs = tokenizer(sentences,
                           max_length=max_seq_length,
                           truncation=True,
                           return_token_type_ids=False,
                           padding='max_length',
                           return_tensors='pt')

        examples['input_ids'] = list(
            torch.split(inputs['input_ids'], num_sections))
        examples['attention_mask'] = list(
            torch.split(inputs['attention_mask'], num_sections))
        examples['num_sections'] = num_sections

        return examples
    return preprocess


def ctx_hibert_model_init(tokenizer, model_path, coordinator_config_path):
    coordinator_config = tfs.AutoConfig.from_pretrained(
        coordinator_config_path, use_cache=False)

    encoder_config = tfs.AutoConfig.from_pretrained(
        model_path,
        gradient_checkpointing=True,
        use_cache=False)
        #add_pooling_layer=True,
        #tie_word_embeddings=False)

    encoder = modeling_bert.CustomBertForMaskedLM.from_pretrained(
        model_path, config=encoder_config)

    if encoder_config.vocab_size < len(tokenizer):
        if comm.is_local_master():
            print(
                f'Resize vocab: {encoder_config.vocab_size}->{len(tokenizer)}')
        encoder.resize_token_embeddings(len(tokenizer))

    encoder_config.vocab_size = len(tokenizer)
    model = modeling_bert.HierarchicalBertForMaskedLM(
        encoder_config, coordinator_config)

    model.hibert.bert = encoder.bert
    model.cls = encoder.cls
    return model


def get_turn_embeddings(
    model, ds, dc, batch_size=4, slide=None, cuda_device=0):

    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, collate_fn=dc)

    tensor_dict = collections.defaultdict(list)
    for batch in tqdm.tqdm(data_loader):
        batch_size = batch['input_ids'].size(0)

        batch.pop('mlm_labels')

        obj_to_device(batch, device=cuda_device)

        with torch.no_grad():
            output = model(**batch)

        #hidden_states = output[embedding_name].detach().cpu()
        bert_pooler_output = output['bert_pooler_output'].detach().cpu()
        pooler_output = output['pooler_output'].detach().cpu()
        ctx_pooler_output = output['ctx_pooler_output'].detach().cpu()
        attention_mask = batch['attention_mask'].detach().cpu()

        bert_last_hidden_state = output['last_hidden_state'].detach().cpu()
        conversation_pooler_output = \
            output['conversation_pooler_output'].detach().cpu()

        bert_last_hidden_state = einops.rearrange(
            bert_last_hidden_state,
            '(bs nt) mnt hs -> bs nt mnt hs',
            bs=batch_size)

        bert_mean_pooler_output = tensor_utils.masked_mean(
            bert_last_hidden_state, attention_mask.unsqueeze(-1), 2)

        # attention_mask shape = (bt, nt).
        attention_mask = torch.clamp(attention_mask.sum(2), max=1).bool()

        bert_pooler_output = einops.rearrange(
            bert_pooler_output, '(bs nt) hs -> bs nt hs', bs=batch_size)
        pooler_output = einops.rearrange(
            pooler_output, '(bs nt) hs -> bs nt hs', bs=batch_size)
        ctx_pooler_output = einops.rearrange(
            ctx_pooler_output, '(bs nt) 1 hs -> bs nt hs', bs=batch_size)

        if slide is not None:
            #hidden_states = hidden_states[:, :slide, :]
            bert_pooler_output = bert_pooler_output[:, :slide, :]
            pooler_output = pooler_output[:, :slide, :]
            ctx_pooler_output = ctx_pooler_output[:, :slide, :]
            attention_mask = attention_mask[:, :slide]
            bert_mean_pooler_output = bert_mean_pooler_output[:, :slide, :]

        tensor_dict['bert_pooler_output'].append(bert_pooler_output)
        tensor_dict['pooler_output'].append(pooler_output)
        tensor_dict['ctx_pooler_output'].append(ctx_pooler_output)
        tensor_dict['attention_mask'].append(attention_mask)
        tensor_dict['bert_mean_pooler_output'].append(bert_mean_pooler_output)
        tensor_dict['conversation_pooler_output'].append(
            conversation_pooler_output)

        # Releases GPU memory to avoid OOM.
        del output

    attention_mask = torch.cat(tensor_dict.pop('attention_mask'), dim=0)
    for k, v in tensor_dict.items():
        if k == 'conversation_pooler_output':
            continue
        tensor = torch.cat(v, dim=0)
        tensor = tensor[attention_mask].numpy()
        tensor_dict[k] = tensor
    tensor_dict['conversation_pooler_output'] = torch.cat(
        tensor_dict['conversation_pooler_output'], dim=0)
    tensor_dict['dialog_lens'] = attention_mask.sum(1).numpy()
    return tensor_dict


def save_embeddings(emb_dir, split, output):
    split_emb_dir = os.path.join(emb_dir, split)
    os.makedirs(split_emb_dir, exist_ok=True)
    for k in ['bert_mean_pooler_output', 'bert_pooler_output', 'pooler_output', 'ctx_pooler_output', 'dialog_lens', 'conversation_pooler_output']:
        with open(os.path.join(split_emb_dir, f'{k}.npy'), 'wb') as f:
            np.save(f, output[k])


def load_embeddings(emb_dir, split):
    split_emb_dir = os.path.join(emb_dir, split)
    ret = {}
    for k in ['bert_mean_pooler_output', 'bert_pooler_output', 'pooler_output', 'ctx_pooler_output', 'dialog_lens', 'conversation_pooler_output']:
        path = os.path.join(split_emb_dir, f'{k}.npy')
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            ret[k] = np.load(f)
    return ret