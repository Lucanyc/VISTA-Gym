
#!/usr/bin/env python3
"""VLM Agent implementation"""

import torch
from typing import Dict, Any, Tuple, Optional, List
import logging
from PIL import Image

from .base import BaseAgent
from .utils import (
    load_image_safely,
    parse_vlm_response,
    format_prompt_with_choices,
    validate_observation,
    calculate_confidence_score
)

logger = logging.getLogger(__name__)


class VLMAgent(BaseAgent):
    """Vision-Language Model Agent for VLM Gym
    
    This agent handles pure inference for vision-language tasks.
    All RL/training logic is handled externally.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize VLM Agent
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Model-specific attributes
        self.device = None
        self._model_loaded = False
        
    def load_model(self):
        """Load the VLM model and processor"""
        if self._model_loaded:
            return
            
        logger.info(f"Loading model: {self.config.model_name}")
        
        try:
            # Try loading with Qwen2.5 specific class first
            if "Qwen2.5" in self.config.model_name or "Qwen2_5" in self.config.model_name:
                self._load_qwen25_model()
            else:
                self._load_auto_model()
                
            self._model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_qwen25_model(self):
        """Load Qwen2.5-VL model specifically"""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            # Determine dtype
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32
            }
            dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
            
            # Load model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                device_map=self.config.device_map,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Get device
            self.device = next(self.model.parameters()).device
            
        except ImportError:
            logger.warning("Qwen2_5_VLForConditionalGeneration not available, falling back to AutoModel")
            self._load_auto_model()
    
    def _load_auto_model(self):
        """Load model using AutoModel classes"""
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
        
        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Get device
        self.device = next(self.model.parameters()).device
    
    def _prepare_input(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for the VLM model
        
        Args:
            observation: Dictionary with image_path, question, choices, etc.
            
        Returns:
            Dictionary with prepared inputs for model
        """
        # Validate observation
        is_valid, error_msg = validate_observation(observation)
        if not is_valid:
            raise ValueError(f"Invalid observation: {error_msg}")
        
        # Extract components
        image_path = observation['image_path']
        question = observation['question']
        choices = observation.get('choices', None)
        
        # Load image
        image = load_image_safely(image_path)
        
        # Format prompt based on task type
        if choices:
            prompt = format_prompt_with_choices(question, choices)
        else:
            prompt = question
        
        # Build messages in the format expected by the model
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                 for k, v in inputs.items()}
        
        # Store additional info for response parsing
        inputs['_observation'] = observation
        inputs['_messages'] = messages
        
        return inputs
    
    def _generate_response(self, model_input: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate response using the model
        
        Args:
            model_input: Prepared input from _prepare_input
            
        Returns:
            response: Generated text
            extra_info: Additional information
        """
        # Extract observation info
        observation = model_input.pop('_observation')
        messages = model_input.pop('_messages')
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        # Generate with optional logprobs
        extra_info = {}
        
        try:
            with torch.no_grad():
                if self.config.return_logprobs:
                    # Generate with scores for logprob calculation
                    gen_kwargs["output_scores"] = True
                    gen_kwargs["return_dict_in_generate"] = True
                    
                    outputs = self.model.generate(**model_input, **gen_kwargs)
                    generated_ids = outputs.sequences
                    scores = outputs.scores
                    
                    # Calculate logprobs
                    logprobs = self._calculate_logprobs(
                        generated_ids, scores, model_input['input_ids']
                    )
                    extra_info["logprobs"] = logprobs
                    extra_info["confidence"] = calculate_confidence_score(logprobs)
                    
                else:
                    # Standard generation
                    generated_ids = self.model.generate(**model_input, **gen_kwargs)
                
                # Decode response
                # Only decode the generated tokens (not the input)
                input_length = model_input['input_ids'].shape[1]
                generated_tokens = generated_ids[0][input_length:]
                response = self.processor.decode(
                    generated_tokens,
                    skip_special_tokens=True
                )
                
                # Add token count
                extra_info["tokens_generated"] = len(generated_tokens)
                extra_info["tokens_used"] = len(generated_tokens) + input_length
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
        
        return response.strip(), extra_info
    
    def _parse_response(self, raw_response: str, observation: Dict[str, Any]) -> str:
        """Parse the model response
        
        Args:
            raw_response: Raw text from model
            observation: Original observation for context
            
        Returns:
            Parsed response
        """
        # Use utility function to parse based on task type
        choices = observation.get('choices', None)
        parsed = parse_vlm_response(raw_response, choices)
        
        logger.debug(f"Parsed response: '{raw_response}' -> '{parsed}'")
        
        return parsed
    
    def _calculate_logprobs(self, generated_ids: torch.Tensor, 
                           scores: tuple, 
                           input_ids: torch.Tensor) -> List[float]:
        """Calculate log probabilities for generated tokens
        
        Args:
            generated_ids: Generated token IDs
            scores: Tuple of score tensors from generation
            input_ids: Input token IDs
            
        Returns:
            List of log probabilities
        """
        logprobs = []
        input_length = input_ids.shape[1]
        
        for i, score_tensor in enumerate(scores):
            # Get the token that was actually generated
            token_id = generated_ids[0, input_length + i]
            
            # Calculate log softmax
            log_probs = torch.log_softmax(score_tensor[0], dim=-1)
            
            # Get log prob for the generated token
            token_logprob = log_probs[token_id].item()
            logprobs.append(token_logprob)
        
        return logprobs
    
    def batch_act(self, observations: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        """Process multiple observations in a batch
        
        Args:
            observations: List of observation dictionaries
            
        Returns:
            List of (action, extra_info) tuples
        """
        # For now, process sequentially
        # TODO: Implement true batch processing for efficiency
        results = []
        
        for obs in observations:
            action, extra_info = self.act(obs)
            results.append((action, extra_info))
        
        return results
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint
        
        Args:
            path: Path to save checkpoint
        """
        if self.model is not None:
            logger.info(f"Saving checkpoint to {path}")
            # Save only the model state dict
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__,
                'step_count': self.step_count,
            }, path)
        else:
            logger.warning("No model loaded, cannot save checkpoint")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint
        
        Args:
            path: Path to load checkpoint from
        """
        logger.info(f"Loading checkpoint from {path}")
        
        # Ensure model is loaded first
        if self.model is None:
            self.load_model()
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.step_count = checkpoint.get('step_count', 0)
        
        logger.info(f"Checkpoint loaded, step_count: {self.step_count}")
