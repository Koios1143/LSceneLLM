"""KV-cached greedy decoding for LSceneLLM (``--fast`` in ``infer.py``).

The upstream decoder (``tools/generation_utils.greedy_decode``) re-runs the whole sequence for
every new token, ``max_length`` times. LSceneLLM's adaptive dense-token attention
(``models/modeling_llama.LlamaAttention``) is causal row by row: the dense (hd) tokens a query
row attends to in layer ``l`` are chosen from that same row's attention to the sparse scene
tokens in layer ``l - 1`` (min-max normalised per row, threshold), and its output mixes the
two attention results for rows in ``[start_learnable_seq_id, end_learnable_seq_id]``. A row
therefore depends only on itself and the earlier rows' keys / values, so caching the keys /
values of earlier positions gives the same result as recomputing them.

``flex_attention_forward`` is ``LlamaAttention.forward`` with one change: the query rows are
the positions ``past_len .. past_len + q_len - 1`` (``past_len`` = cached length) instead of
``0 .. q_len - 1``, so the row slices ``[start_learnable_seq_id:]`` and the mixing range use
absolute positions. With an empty cache it computes exactly what the original computes.
``kv_greedy`` prefills the prompt once through the model's own forward (which places the scene
and visual-prompt embeddings and sets the learnable range), then feeds one token per step,
moving ``end_learnable_seq_id`` to the newest position as the upstream re-run does
(``_replace_placehold_to_embeddings`` sets it to the last position every step). It stops at
EOS; the upstream loop continues to ``max_length`` but writes only EOS after it.
"""

import math
from collections import OrderedDict

import torch
from torch import nn

from models import modeling_llama as ml


def flex_attention_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                           output_attentions=False, use_cache=False, last_select_mask_list=[], **kwargs):
    bsz, q_len, _ = hidden_states.size()

    has_hd_features = self.config.FLEX and hasattr(self, 'hd_features')
    need_to_concat_hd_feature = self.layer_idx > self.config.START_FLEX_LAYER and has_hd_features
    need_to_get_hd_feature_mask = self.layer_idx >= self.config.START_FLEX_LAYER and has_hd_features
    scene_token_start_index = self.config.scene_token_start_index
    start_learnable_seq_id = [int(i.item()) for i in self.config.start_learnable_seq_id]
    end_learnable_seq_id = [int(i.item()) for i in self.config.end_learnable_seq_id]

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    past_len = kv_seq_len - q_len  # absolute position of query row 0
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = ml.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx,
                                                         {"sin": sin, "cos": cos})
    key_states = ml.repeat_kv(key_states, self.num_key_value_groups)
    value_states = ml.repeat_kv(value_states, self.num_key_value_groups)

    if need_to_concat_hd_feature:
        if not self.hd_features.shape[0] == bsz:
            self.hd_features = self.hd_features.repeat(bsz, 1, 1, 1)
        hd_features = self.hd_features.view(bsz, -1, self.hidden_size)
        hd_cos = cos[scene_token_start_index: scene_token_start_index + self.config.VISION_TOKEN_NUM] \
            .repeat_interleave(self.config.DENSE_TOKEN_NUM, dim=0)
        hd_sin = sin[scene_token_start_index: scene_token_start_index + self.config.VISION_TOKEN_NUM] \
            .repeat_interleave(self.config.DENSE_TOKEN_NUM, dim=0)
        key_hd = self.k_proj_hd(hd_features).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_hd = self.v_proj_hd(hd_features).view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key_hd = (key_hd * hd_cos) + (ml.rotate_half(key_hd) * hd_sin)
        key_states = torch.cat([key_states, key_hd], dim=2)
        value_states = torch.cat([value_states, value_hd], dim=2)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                             f"but is {attention_mask.size()}")
        if need_to_concat_hd_feature:
            empty = torch.ones(attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[2],
                               hd_features.shape[1], device=attention_mask.device,
                               dtype=attention_mask.dtype) * torch.finfo(attn_weights.dtype).min
            select = empty.clone()
            for ib in range(hd_features.shape[0]):
                select[ib, 0][last_select_mask_list[ib]] = 0.0
            attn_weights_select = attn_weights + torch.cat([attention_mask, select], dim=3)
            attn_weights = attn_weights + torch.cat([attention_mask, empty], dim=3)
        else:
            attn_weights = attn_weights + attention_mask

    rows = torch.arange(past_len, past_len + q_len, device=hidden_states.device)  # absolute positions
    if need_to_concat_hd_feature:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_select = nn.functional.softmax(attn_weights_select, dim=-1,
                                                    dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output_select = torch.matmul(attn_weights_select, value_states)
        alpha = 1.0  # eval (upstream: warm-up factor only while training)
        masks = [((rows >= start_learnable_seq_id[ib]) & (rows <= end_learnable_seq_id[ib]))[None, None]
                 for ib in range(attn_weights.shape[0])]
        masks = torch.cat(masks, dim=0).expand(-1, attn_output.shape[1], -1)
        attn_output[masks] *= (1.0 - alpha)
        attn_output[masks] += attn_output_select[masks] * alpha
    else:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    select_mask_list = []
    if need_to_get_hd_feature_mask:
        global_w = torch.sum(attn_weights, dim=1).detach()
        first = max(start_learnable_seq_id[0] - past_len, 0)  # first local row with position >= start
        scene_w = global_w[:, first:, :][:, :, scene_token_start_index: scene_token_start_index
                                          + self.config.VISION_TOKEN_NUM]
        threshold = self.config.DENSE_TOKEN_SELECT_THRESHOLD
        for bi in range(attn_weights.shape[0]):
            w = scene_w[bi]
            select_mask = torch.zeros(attn_weights.shape[-2], self.config.VISION_TOKEN_NUM
                                      * self.config.DENSE_TOKEN_NUM, device=attn_output.device,
                                      dtype=attn_output.dtype)
            mn = w.min(dim=1, keepdim=True).values
            mx = w.max(dim=1, keepdim=True).values
            w = (((w - mn) / (mx - mn)) * 255 > threshold).bool()
            select_mask[first:, :] = w.repeat_interleave(self.config.DENSE_TOKEN_NUM, dim=-1)
            select_mask_list.append(select_mask.bool())

    return attn_output, None, past_key_value, select_mask_list


def install():
    """Use the cache-aware forward for every LlamaAttention (eager) layer."""
    ml.LlamaAttention.forward = flex_attention_forward


@torch.no_grad()
def kv_greedy(transformer, **kwargs):
    """Same interface / output as ``tools.generation_utils.generation`` with ``num_beams=None``."""
    max_length = kwargs["max_length"]
    eos = kwargs["eos_token_id"]
    input_ids = kwargs["input_ids"]
    assert input_ids.shape[0] == 1
    out = transformer(input_ids=input_ids, attention_mask=kwargs["attention_mask"],
                      vision_embeds=kwargs["vision_embeds"], vision_mask=kwargs["vision_mask"],
                      visual_prompt_embeds=kwargs["visual_prompt_embeds"],
                      visual_prompt_mask=kwargs["visual_prompt_mask"], tokenizer=kwargs["tokenizer"],
                      use_cache=True, output_attentions=False)
    cfg = transformer.config
    past = out.past_key_values
    # the prefill's 2-D mask as _replace_placehold_to_embeddings builds it (an unused visual prompt
    # has mask 0 and stays masked), then one 1 per generated token, as the upstream re-run does
    tok = kwargs["tokenizer"]
    p0 = int(torch.where(input_ids == tok.convert_tokens_to_ids("<scene_placehold>"))[1][0])
    p1 = int(torch.where(input_ids == tok.convert_tokens_to_ids("<vp_placehold>"))[1][0])
    m = kwargs["attention_mask"]
    mask = torch.cat((m[:, :p0], kwargs["vision_mask"], m[:, p0 + 1:p1], kwargs["visual_prompt_mask"],
                      m[:, p1 + 1:]), dim=1).float()
    seq_len = int(cfg.end_learnable_seq_id[0].item()) + 1  # prefill length (scene + prompt tokens)
    assert mask.shape[1] == seq_len, (mask.shape, seq_len)
    output_ids = torch.full((1, max_length), eos, dtype=torch.long, device=input_ids.device)
    embed = transformer.get_input_embeddings()
    logits = out.logits[:, -1, :]
    for step in range(max_length):
        nxt = logits.argmax(-1)
        output_ids[:, step] = nxt
        if int(nxt.item()) == eos or step == max_length - 1:
            break
        cfg.end_learnable_seq_id = cfg.end_learnable_seq_id.clone().fill_(seq_len)  # new row's position
        seq_len += 1
        mask = torch.cat((mask, mask.new_ones(1, 1)), dim=1)
        o = transformer.model(inputs_embeds=embed(nxt[:, None]), past_key_values=past, use_cache=True,
                              attention_mask=mask.to(m.dtype), output_attentions=False)
        past = o.past_key_values
        logits = transformer.lm_head(o[0][:, -1:, :]).float()[:, -1, :]
    return OrderedDict({"output_ids": output_ids, "attentions": None})
