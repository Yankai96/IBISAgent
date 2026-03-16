import torch
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info

class MLLMEngine:
    def __init__(self, model_path="qwen2_5vl-7b-RL"):
        print(f"[MLLMEngine] Loading Qwen2.5-VL model from {model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.gen_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            num_return_sequences=1,
            pad_token_id=151643,
        )
        self.messages = []
        self.system_prompt = ""
        print("[MLLMEngine] Model loaded successfully.")

    def init_session(self, system_prompt):
        """初始化一个多轮对话会话"""
        self.system_prompt = system_prompt
        self.messages = []
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

    def chat(self, user_prompt=None, image_path=None, is_first_turn=True, use_history=True):
        """
        进行一轮对话。
        如果是第一轮，传入 user_prompt 和 初始 image_path。
        如果是后续轮次，仅传入更新后的 image_path（包含掩码），并将图片作为 observation 传入。
        use_history: 是否使用历史对话（默认为 True）。如果为 False，则每轮仅保留 System Prompt 和当前的 User Prompt。
        """
        if not use_history:
            self.messages = []
            if self.system_prompt:
                self.messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}]
                })

        if is_first_turn:
            content = []
            if image_path:
                content.append({"type": "image", "image": image_path})
            if user_prompt:
                content.append({"type": "text", "text": user_prompt})
            self.messages.append({"role": "user", "content": content})
        else:
            content = []
            if image_path:
                content.append({"type": "image", "image": image_path})
            
            refinement_prompt = f"<image>This is a refinement step. User's original query: \"{user_prompt}\". Do you think the image with semi-transparent green mask need further segmentation? If so, please use point action; otherwise, use Terminate action."
            content.append({"type": "text", "text": refinement_prompt})
            self.messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(self.messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            generation_config=self.gen_config
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
        
        return self.parse_output(output_text)

    def parse_output(self, text):
        """解析模型输出中的 <think> 和 <action> 标签"""
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        action_match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        
        think_content = think_match.group(1).strip() if think_match else ""
        action_content = action_match.group(1).strip() if action_match else ""
        
        action_type = None
        coords = None
        
        if "Terminate" in action_content:
            action_type = "Terminate"
        elif "Positive Point" in action_content:
            action_type = "Positive"
            coords_match = re.search(r'\(([\d.]+),\s*([\d.]+)\)', action_content)
            if coords_match:
                coords = (float(coords_match.group(1)), float(coords_match.group(2)))
        elif "Negative Point" in action_content:
            action_type = "Negative"
            coords_match = re.search(r'\(([\d.]+),\s*([\d.]+)\)', action_content)
            if coords_match:
                coords = (float(coords_match.group(1)), float(coords_match.group(2)))
                
        return {
            "raw": text,
            "think": think_content,
            "action_raw": action_content,
            "action_type": action_type,
            "coords": coords
        }
