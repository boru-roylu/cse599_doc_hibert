from typing import Optional, Tuple, Union

import einops
import torch
import transformers as tfs
from transformers.utils import logging

import modeling_outputs

logger = logging.get_logger(__name__)


def safe_cross_entropy_loss(logits, labels, ignore_index=-100, weight=None):
    loss_fct = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
    mask = (labels != ignore_index).type_as(logits)
    loss = loss_fct(logits, labels)
    if mask.sum() == 0:
        loss = loss.mean()
    else:
        loss = loss.sum() / mask.sum()
    return loss


class CustomBertForMaskedLM(tfs.BertForMaskedLM):
    #_keys_to_ignore_on_load_unexpected = [r"pooler"]
    #_keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = tfs.models.bert.modeling_bert.BertModel(config, add_pooling_layer=True)
        self.cls = tfs.models.bert.modeling_bert.BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()


class HierarchicalBertModel(tfs.models.bert.modeling_bert.BertPreTrainedModel):
    def __init__(self, config, coordinator_config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        #assert config.hidden_size == coordinator_config.hidden_size

        self.bert = tfs.BertModel(config)#, add_pooling_layer=False)

        #self.coordinator = tfs.BertModel(coordinator_config)
        self.coordinator = tfs.models.bert.modeling_bert.BertEncoder(
            coordinator_config)#, add_pooling_layer=False)

        self.cls_token_emb = torch.nn.Parameter(
            torch.randn(1, 1, coordinator_config.hidden_size))

        self.coordinator_type_embeddings = torch.nn.Embedding(
            3, coordinator_config.hidden_size)

        #if coordinator_config.hidden_size != config.hidden_size:
        self.proj_for_coordinator = torch.nn.Linear(
            config.hidden_size, coordinator_config.hidden_size)
        if coordinator_config.add_ctx_pooled_output_to_tokens:
            self.output_proj_for_coordinator = torch.nn.Linear(
                coordinator_config.hidden_size, config.hidden_size)

        #position_embedding_type = getattr(
        #    coordinator_config, 'position_embedding_type', 'absolute')
        # Explicit add one more setting becasuse we might use absolute and 
        # relative positional embeddings at the same time.
        if (coordinator_config.add_absolute_position_embeddings
            and coordinator_config.max_position_embeddings > 0):
            self.turn_position_embeddings = torch.nn.Embedding(
                coordinator_config.max_position_embeddings,
                coordinator_config.hidden_size)
            self.register_buffer(
                'turn_position_ids',
                torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.LayerNorm = torch.nn.LayerNorm(
                coordinator_config.hidden_size,
                eps=coordinator_config.layer_norm_eps)
            self.dropout = torch.nn.Dropout(coordinator_config.hidden_dropout_prob)

        if coordinator_config.add_ctx_pooled_output_to_tokens:
            self.ctx_LayerNorm = torch.nn.LayerNorm(
                config.hidden_size,
                eps=coordinator_config.layer_norm_eps)
            self.ctx_dropout = torch.nn.Dropout(coordinator_config.hidden_dropout_prob)
        self.coordinator_config = coordinator_config


        # Initialize weights and apply final processing
        self.post_init()
        #self.init_weights()

    #def post_init(self):
    #    """
    #    A method executed at the end of each Transformer model initialization, to execute code that needs the model's
    #    modules properly initialized (such as weight initialization).
    #    """
    #    self.init_weights()
    #    self._backward_compatibility_gradient_checkpointing()

    #def _backward_compatibility_gradient_checkpointing(self):
    #    if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
    #        self.gradient_checkpointing_enable()
    #        # Remove the attribute now that is has been consumed, so it's no saved in the config.
    #        delattr(self.config, "gradient_checkpointing")

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bert.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.Seq2SeqLMOutput]:
        r"""
        mlm_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        if input_ids != None:
            assert input_ids.ndim == 3
            batch_size, num_turns, num_tokens = input_ids.size()
            input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask != None:
            coordinator_attention_mask = torch.clamp(attention_mask.sum(2), max=1)
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        device = input_ids.device

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        bert_pooled_output = encoder_outputs[1]

        # Sentence-level encoding.
        pooled_output = einops.rearrange(
            bert_pooled_output, '(bs nt) hs-> bs nt hs', bs=batch_size)

        if hasattr(self, 'proj_for_coordinator'):
            pooled_output = self.proj_for_coordinator(pooled_output)
        cls_token_embs = einops.repeat(
            self.cls_token_emb, '() nt hs -> bs nt hs', bs=batch_size)
        cls_attention_mask = torch.ones(
            (batch_size, 1), dtype=attention_mask.dtype).to(device)
        coordinator_attention_mask = torch.cat(
            (cls_attention_mask, coordinator_attention_mask), dim=1)

        # num_turns + 1: +1 is for the utterance-level CLS.
        coordinator_type_ids = torch.zeros(
            (batch_size, num_turns+1), dtype=input_ids.dtype).to(device)
        coordinator_type_embs = self.coordinator_type_embeddings(
            coordinator_type_ids)

        pooled_output = torch.cat(
            (cls_token_embs, pooled_output), dim=1)
        pooled_output = pooled_output + coordinator_type_embs

        if hasattr(self, 'turn_position_ids'):
            turn_position_ids = self.turn_position_ids[:, :num_turns+1]
            turn_position_embeddings = \
                self.turn_position_embeddings(turn_position_ids)
            pooled_output = pooled_output + turn_position_embeddings
            pooled_output = self.LayerNorm(pooled_output)
            pooled_output = self.dropout(pooled_output)

        extended_coordinator_attention_mask = \
            self.get_extended_attention_mask(
                coordinator_attention_mask,
                pooled_output.size()[:2], device)
                #coordinator_inputs_embs.size()[:2], device)
        coordinator_outputs = self.coordinator(
            #inputs_embeds=pooled_output,
            #attention_mask=coordinator_attention_mask)
            pooled_output,
            attention_mask=extended_coordinator_attention_mask)
        ctx_pooled_output = coordinator_outputs[0]

        # The first index is for the utterance-level CLS token.
        conversation_pooled_output = ctx_pooled_output[:, 0, :]
        ctx_pooled_output = ctx_pooled_output[:, 1:, :]
        coordinator_attention_mask = coordinator_attention_mask[:, 1:]
        ctx_pooled_output = einops.rearrange(
            ctx_pooled_output, 'bs nt hs -> (bs nt) 1 hs')
        coordinator_attention_mask = einops.rearrange(
            coordinator_attention_mask, 'bs nt -> (bs nt) 1')

        if self.coordinator_config.add_ctx_pooled_output_to_tokens:
            sequence_output = sequence_output + \
                self.output_proj_for_coordinator(ctx_pooled_output)
            sequence_output = self.ctx_LayerNorm(sequence_output)
            sequence_output = self.ctx_dropout(sequence_output)

        if not return_dict:
            return (
                sequence_output,
                bert_pooled_output,
                pooled_output,
                ctx_pooled_output) + encoder_outputs[1:]

        # Ignores the utterance-level CLS token.
        pooled_output = pooled_output[:, 1:, :]
        pooled_output = einops.rearrange(pooled_output, 'bs nt hs -> (bs nt) hs')

        return modeling_outputs.HierarchicalBertModelOutput(
            last_hidden_state=sequence_output,
            bert_pooler_output=bert_pooled_output,
            pooler_output=pooled_output,
            ctx_pooler_output=ctx_pooled_output,
            ctx_attentions=coordinator_attention_mask,
            conversation_pooler_output=conversation_pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class HierarchicalBertForMaskedLM(tfs.models.bert.modeling_bert.BertPreTrainedModel):
    def __init__(self, config, coordinator_config):
        super().__init__(config)

        self.hibert = HierarchicalBertModel(config, coordinator_config)
        self.cls = tfs.models.bert.modeling_bert.BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.hibert.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.hibert.bert.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.Seq2SeqLMOutput]:

        if mlm_labels != None:
            assert mlm_labels.ndim == 3
            mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))

        outputs = self.hibert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        prediction_scores = self.cls(sequence_output).to(dtype=torch.float32)
        masked_lm_loss = None
        if mlm_labels is not None:
            #loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
            #masked_lm_loss = loss_fct(
            #    prediction_scores.view(-1, self.config.vocab_size),
            #    mlm_labels.view(-1))
            masked_lm_loss = safe_cross_entropy_loss(
                prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

        #loss = masked_lm_loss + decoder_loss

        #if not return_dict:
        #    output = (prediction_scores,) + outputs[2:]
        #    return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return tfs.modeling_outputs.MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HierarchicalBertForMaskedLMandSentenceInfilling(tfs.models.bert.modeling_bert.BertPreTrainedModel):
    def __init__(self, config, coordinator_config, decoder_config):
        super().__init__(config)

        assert config.is_encoder_decoder

        self.hibert = HierarchicalBertModel(config, coordinator_config)
        self.decoder = tfs.BertLMHeadModel(decoder_config)
        self.cls = tfs.models.bert.modeling_bert.BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    #def gradient_checkpointing_enable(self):
    #    self.hibert.apply(
    #        functools.partial(
    #            self.hibert._set_gradient_checkpointing, value=True))

    #    self.decoder.apply(
    #        functools.partial(
    #            self.decoder._set_gradient_checkpointing, value=True))

    def get_encoder(self):
        return self.hibert

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.hibert.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.hibert.bert.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        #past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs):
        # cut decoder_input_ids if past is used
        #if past is not None:
        #    decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            #"past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            #"cross_attn_head_mask": cross_attn_head_mask,
            #"use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        if input_ids != None:
            batch_size, num_turns, num_tokens = input_ids.size()
        if decoder_input_ids != None:
            #assert decoder_input_ids.ndim == 3
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size(-1))
        if labels != None:
            #assert decoder_labels.ndim == 3
            labels = labels.view(-1, labels.size(-1))
        if decoder_attention_mask != None:
            decoder_attention_mask = decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.hibert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = encoder_outputs['last_hidden_state']

        mlm_logits = self.cls(sequence_output).to(dtype=torch.float32)
        masked_lm_loss = None
        if mlm_labels is not None:
            #loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
            #masked_lm_loss = loss_fct(
            #    mlm_logits.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            masked_lm_loss = safe_cross_entropy_loss(
                mlm_logits.view(-1, self.config.vocab_size), mlm_labels.view(-1))

        hidden_states = encoder_outputs['ctx_pooler_output']
        attention_mask = encoder_outputs['ctx_attentions']

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            head_mask=decoder_head_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask)
            #labels=labels)

        loss = masked_lm_loss
        prediction_scores = decoder_outputs['logits']
        decoder_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            shifted_prediction_scores = \
                shifted_prediction_scores.to(dtype=torch.float32)

            loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
            decoder_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1))
            #decoder_loss = modeling_utils.safe_cross_entropy_loss(
            #    shifted_prediction_scores.view(-1, self.config.vocab_size),
            #    labels.view(-1))
            loss = masked_lm_loss + decoder_loss

        decoder_logits = decoder_outputs['logits']

        #if not return_dict:
        #    output = (prediction_scores,) + outputs[2:]
        #    return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        #mlm_preds = encoder_outputs['logits'].argmax(-1)
        #mlm_logits = encoder_outputs['logits']

        return modeling_outputs.Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_logits,
            decoder_loss=decoder_loss,
            mlm_logits=mlm_logits,
            mlm_loss=masked_lm_loss,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.hidden_states,
            #encoder_hidden_states=outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class HierarchicalBertForConversationClassification(
    tfs.models.bert.BertPreTrainedModel):
    def __init__(self, config, coordinator_config):
        super().__init__(config)

        self.hibert = HierarchicalBertModel(config, coordinator_config)
        self.classifier = torch.nn.Linear(
            coordinator_config.hidden_size, coordinator_config.num_labels)
        self.coordinator_config = coordinator_config

        # Initialize weights and apply final processing
        self.post_init()
        #self.init_weights()

    def get_input_embeddings(self):
        return self.hibert.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.hibert.bert.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], modeling_outputs.Seq2SeqLMOutput]:

        outputs = self.hibert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        cls_pooled_output = outputs['conversation_pooler_output']
        logits = self.classifier(cls_pooled_output).to(dtype=torch.float32)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            #if self.class_weight is None:
            #    loss_fct = torch.nn.CrossEntropyLoss()
            #else:
            #    loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weight)
            loss = loss_fct(
                logits.view(-1, self.coordinator_config.num_labels),
                labels.view(-1))

        return tfs.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=cls_pooled_output,
            attentions=outputs.ctx_attentions)