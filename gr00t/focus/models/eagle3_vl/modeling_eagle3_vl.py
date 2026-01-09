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
        # Get image token positions for each batch
        # Note: Eagle3_VL may have multiple images with variable spatial shapes
        # For simplicity, we handle the common case of uniform image sizes
        
        image_token_mask = (input_ids == self.image_token_index)
        
        # Only prepare Focus if there are image tokens
        if image_token_mask.any():
            # Get the total number of image tokens per batch
            # Assuming all batches have the same structure
            batch_image_tokens = image_token_mask[0].sum().item()
            
            if batch_image_tokens > 0:
                # Find the start and end positions of image tokens in the first batch
                image_positions = torch.where(image_token_mask[0])[0]
                image_token_start_index = image_positions[0].item()
                image_token_end_index = image_positions[-1].item()
                image_token_length = batch_image_tokens
                original_length = N
                
                # Calculate patch dimensions
                # Eagle3_VL uses pixel_shuffle_back with downsample_ratio (typically 0.5)
                # The visual tokens are organized based on the downsampled spatial shapes
                # For now, we estimate patch dimensions from the number of visual tokens
                
                # Try to get spatial information from the model's last extraction
                # This is a simplified calculation - for variable image sizes,
                # this would need to be more sophisticated
                
                # Estimate square patch size from number of tokens
                # Each image contributes tokens based on its downsampled spatial shape
                num_visual_tokens = vit_embeds.shape[0]
                tokens_per_image = num_visual_tokens // max(num_images, 1)
                
                # Estimate patch dimensions (assuming roughly square images after downsampling)
                import math
                patch_size = int(math.sqrt(tokens_per_image))
                if patch_size * patch_size != tokens_per_image:
                    # Not a perfect square, try to find closest factors
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
                n_frames = num_images  # Each image is treated as a "frame"
                
                # Calculate strides
                frame_stride = patch_height * patch_width
                height_stride = patch_width
                width_stride = 1
                
                # Create patch_type tensor
                # TEXT_TOKEN for text positions, patch indices for image positions
                patch_type_list = []
                patch_type_list.extend([TEXT_TOKEN] * image_token_start_index)
                
                # Add patch indices for each frame/image
                for frame_idx in range(n_frames):
                    tokens_this_frame = min(patch_num, image_token_length - frame_idx * patch_num)
                    patch_type_list.extend(list(range(tokens_this_frame)))
                
                # Fill remaining with TEXT_TOKEN
                patch_type_list.extend([TEXT_TOKEN] * (original_length - len(patch_type_list)))
                
                # Ensure patch_type has correct length
                if len(patch_type_list) > original_length:
                    patch_type_list = patch_type_list[:original_length]
                elif len(patch_type_list) < original_length:
                    patch_type_list.extend([TEXT_TOKEN] * (original_length - len(patch_type_list)))
                
                patch_type = torch.tensor([patch_type_list], device=input_ids.device)
                
                # Prepare Focus with metadata
                self.focus.prepare(
                    patch_type, 
                    n_frames, 
                    patch_height, 
                    patch_width, 
                    frame_stride, 
                    height_stride, 
                    width_stride, 
                    image_token_start_index, 
                    image_token_end_index, 
                    image_token_length, 
                    original_length
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

