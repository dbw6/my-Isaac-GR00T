"""
Focus integration for Eagle3_VL models.

This module provides modified forward functions for Eagle3_VLForConditionalGeneration
that integrate the Focus algorithm for visual token compression.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

# TEXT_TOKEN constant for Focus
TEXT_TOKEN = -1


def Eagle3_VLForConditionalGeneration_focus_forward(
    self,
    pixel_values: List[torch.FloatTensor],
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    image_flags: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """
    Forward pass for Eagle3_VLForConditionalGeneration with Focus integration.
    
    This is identical to the original forward function except for Focus metadata
    preparation before calling the language model.
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    input_embeds = self.language_model.get_input_embeddings()(input_ids)

    num_images = len(pixel_values)
    
    if image_flags is not None:
        image_flags = image_flags.view(-1)

    vit_embeds = self.extract_feature(pixel_values, image_flags)

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids_flat = input_ids.reshape(B * N)
    selected = (input_ids_flat == self.image_token_index)
    
    try:
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
    except Exception as e:
        print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
              f'vit_embeds.shape={vit_embeds.shape}')
        n_token = selected.sum()
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

    input_embeds = input_embeds.reshape(B, N, C)

    ### FOCUS METADATA PREPARATION START ###
    if hasattr(self, 'focus') and input_ids is not None:
        image_token_mask = (input_ids == self.image_token_index)
        
        if image_token_mask.any():
            # Get all image token positions for the first batch
            image_positions = torch.where(image_token_mask[0])[0]
            
            # Group image positions by contiguous segments (one per image)
            image_positions_list = image_positions.tolist()
            image_groups = []
            if image_positions_list:
                group_start = 0
                for idx in range(1, len(image_positions_list)):
                    if image_positions_list[idx] != image_positions_list[idx - 1] + 1:
                        image_groups.append(image_positions_list[group_start:idx])
                        group_start = idx
                image_groups.append(image_positions_list[group_start:])
            
            # Find start and end positions of image tokens only (ignore wrapper tokens)
            # Create lists to store start and end positions for each image
            image_token_start_indices = []
            image_token_end_indices = []
            
            for group in image_groups:
                if not group:
                    continue
                # Use the actual first and last image token positions for this image
                image_token_start_indices.append(group[0])
                image_token_end_indices.append(group[-1])
            
            image_token_length = image_positions.numel()
            original_length = N
            
            # Calculate query_token_start_index and query_token_length
            # For gr00t model, query tokens ("open the left drawer") are BEFORE the first image
            # Format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\nopen the left drawer<image 1><img><IMG_CONTEXT>...
            # <image 1> = 5 tokens, <img> = 1 token, so 6 tokens before first IMG_CONTEXT
            first_image_start = image_token_start_indices[0] if image_token_start_indices else image_positions[0].item()
            # Query text ends right before <image 1> wrapper
            query_token_end_index = first_image_start - 6
            
            # Find where user query starts (after <|im_start|>user\n, exclude system prompt)
            # Look for the second <|im_start|> token (first is system, second is user)
            im_start_token_id = 151644  # <|im_start|> token ID
            im_start_positions = torch.where(input_ids[0] == im_start_token_id)[0]
            if len(im_start_positions) >= 2:
                # User turn starts at second <|im_start|>
                # Skip <|im_start|>user\n = 3 tokens (im_start + user + \n)
                query_token_start_index = im_start_positions[1].item() + 3
            else:
                # Fallback: use position 0
                query_token_start_index = 0
            
            query_token_length = query_token_end_index - query_token_start_index
            
            # Calculate patch dimensions from visual tokens
            num_visual_tokens = vit_embeds.shape[0]
            tokens_per_image = num_visual_tokens // max(num_images, 1)
            if tokens_per_image == 0:
                tokens_per_image = num_visual_tokens
            
            # Estimate patch dimensions (assuming roughly square images after downsampling)
            import math
            patch_size = int(math.sqrt(tokens_per_image))
            if patch_size * patch_size != tokens_per_image:
                # Find closest factors
                for h in range(int(math.sqrt(tokens_per_image)) + 1, 0, -1):
                    if tokens_per_image % h == 0:
                        patch_height = h
                        patch_width = tokens_per_image // h
                        break
                else:
                    patch_height = patch_size
                    patch_width = patch_size
            else:
                patch_height = patch_size
                patch_width = patch_size
            
            patch_num = patch_height * patch_width
            n_frames = num_images
            
            # Calculate strides
            frame_stride = patch_height * patch_width
            height_stride = patch_width
            width_stride = 1
            
            # Create patch_type tensor: TEXT_TOKEN for text, patch indices for image tokens
            patch_type_list = [TEXT_TOKEN] * original_length
            
            # Assign patch indices to each image group (skip wrapper tokens between images)
            for group in image_groups:
                for patch_idx, pos in enumerate(group[:patch_num]):
                    patch_type_list[pos] = patch_idx
            
            patch_type = torch.tensor([patch_type_list], device=input_ids.device)
            
            # Prepare Focus with metadata
            # Pass lists of start and end indices for each image
            self.focus.prepare(
                patch_type, 
                n_frames, 
                patch_height, 
                patch_width, 
                frame_stride, 
                height_stride, 
                width_stride, 
                image_token_start_indices,  # List of start indices for each image
                image_token_end_indices,     # List of end indices for each image
                image_token_length, 
                original_length,
                query_token_start_index,
                query_token_length
            )
    ### FOCUS METADATA PREPARATION END ###

    outputs = self.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )
    logits = outputs.logits

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
