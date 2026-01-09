"""
Focus integration for Qwen3 models.

This module provides modified forward functions for Qwen3 model components
that integrate the Focus algorithm for visual token compression.

Based on the Qwen2 Focus implementation pattern.
"""

from typing import List, Optional, Tuple, Union, Callable
from functools import partial

import torch
import torch.utils.checkpoint
import torch.nn as nn
import math
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
    repeat_kv,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def Qwen3DecoderLayer_focus_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Forward pass for Qwen3DecoderLayer with Focus integration.
    
    Args:
        hidden_states: input to the layer of shape (batch, seq_len, embed_dim)
        attention_mask: attention mask of size (batch, sequence_length)
        position_ids: position indices
        past_key_value: cached past key and value projection states
        output_attentions: whether to return attention weights
        use_cache: whether to use key/value cache
        cache_position: indices depicting position in sequence
        position_embeddings: cosine and sine positional embeddings
        kwargs: additional arguments
    """
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    ### FOCUS SEC MODIFICATION START ###
    if self.self_attn.layer_idx in self.focus.selected_layer and hidden_states.shape[1] > 1:
        self.focus.update_alpha(layer_idx=self.self_attn.layer_idx)
        if self.focus.start_drop:
            hidden_states = self.focus.recover_tokens(hidden_states)
        position_embeddings, attention_mask = self.focus.semantic_concentration(position_embeddings, attention_mask)
        hidden_states = self.focus.drop_tokens(hidden_states)
    ### FOCUS SEC MODIFICATION END ###

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    outputs += (position_embeddings, attention_mask)
    return outputs


def Qwen3Model_focus_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                partial(decoder_layer.__call__, **flash_attn_kwargs),
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )
            
        ### FOCUS: Update position embeddings and attention mask modified by Focus
        position_embeddings = layer_outputs[-2]
        causal_mask = layer_outputs[-1]
        ### End Focus modification
        
        hidden_states = layer_outputs[0]
        
        if output_attentions:
            all_self_attns += (layer_outputs[1],)
    
    hidden_states = self.norm(hidden_states)
    
    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    
    output = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    
    # Focus post-processing
    if inputs_embeds.shape[1] > 1:
        self.focus.post_process()

    # Always return object (not tuple) since Qwen3ForCausalLM expects BaseModelOutputWithPast
    return output


def Qwen3Attention_focus_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Forward pass for Qwen3Attention with Focus integration.
    Identical to original Qwen3Attention.forward except for Focus SIC modifications.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    seq_len = hidden_states.shape[1]

    ### FOCUS SIC: Compress before QKV projections ###
    hidden_states = self.focus(hidden_states, name="q_proj")

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    ### FOCUS SIC: Compress query for attention ###
    query_states = self.focus(query_states, is_attention=True, name='query')

    # Calculate attention weights for token importance (only in selected layers)
    if (self.layer_idx in self.focus.selected_layer and seq_len > 1) or \
       (self.layer_idx in self.focus.extract_attention_layer and seq_len > 1):
        attn_weights_for_focus = calc_attn_weights_qwen3(self, query_states, key_states, attention_mask=attention_mask)
        self.focus.set_token_importance(attn_weights_for_focus)

    # Attention interface selection - identical to original
    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    ### FOCUS SIC: Compress before o_proj ###
    attn_output = self.focus(attn_output, name="o_proj")

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def Qwen3MLP_focus_forward(self, x):
    """
    Forward pass for Qwen3MLP with Focus integration.
    """
    ### FOCUS SIC MODIFICATION START ###
    x = self.focus(x, name="gate_proj")
    ### FOCUS SIC MODIFICATION END ###

    gate_output = self.act_fn(self.gate_proj(x))
    up_output = self.up_proj(x)
    down_input = gate_output * up_output

    ### FOCUS SIC MODIFICATION START ###
    down_input = self.focus(down_input, name="down_proj")
    ### FOCUS SIC MODIFICATION END ###
    
    down_proj = self.down_proj(down_input)
    return down_proj


def calc_attn_weights_qwen3(
    self,
    query_states,
    key_states,
    attention_mask=None,
):
    """
    Calculate attention weights for Focus token importance estimation.
    """
    is_causal = None
    scale = self.scaling
     
    query = query_states.clone()
    key = key_states.clone()

    dropout_p = 0.0
    if hasattr(self, "num_key_value_groups"):
        key = repeat_kv(key, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]
    
    query = query.contiguous()
    key = key.contiguous()

    if is_causal is None:
        is_causal = causal_mask is None and query.shape[2] > 1

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_mask = causal_mask
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    if attn_mask is not None:
        attn_bias = torch.zeros_like(attn_mask, dtype=query.dtype)
    else:
        attn_bias = torch.zeros(L, S, dtype=query.dtype)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), -1e4)
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), -1e4)
        else:
            attn_bias += attn_mask
    
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(query.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight



