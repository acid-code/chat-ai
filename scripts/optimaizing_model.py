from optimum.intel import IncQuantizer
from transformers import AutoModelForCausalLM
from huggingface_hub import login

hugging_face_token = "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"


login(hugging_face_token)

model = AutoModelForCausalLM.from_pretrained("your_model_name")
quantizer = IncQuantizer(model)
optimized_model = quantizer.quantize()
optimized_model.push_to_hub(
    "finetuned_tinyllama", use_auth_token="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"
)
