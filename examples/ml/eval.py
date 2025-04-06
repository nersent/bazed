import torch
from transformers import AutoTokenizer

from ml.model import Transformer, TransformerConfig

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def decoder_only_generate(
    prompt,
    model,
    tokenizer,
    sample=False,
    top_p=1.0,
    top_k=50,
    temperature=1.0,
    max_len=128,
    repetition_penalty=1.0,
    use_kv_cache=True,
    stream=True,
):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)[
        :, :
    ][:, :]

    print(input_ids.shape[1])
    
    past = None

    kwargs = {}

    with torch.inference_mode():
        for i in range(max_len):
            input = (
                input_ids[:, -1].unsqueeze(1) if use_kv_cache and i > 0 else input_ids
            )

            logits = model(input, past_key_values=past, **kwargs)

            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[1]):
                    if input_ids[0][i] in set(input_ids[0][:i]):
                        logits[0][i] /= repetition_penalty

            if not sample:
                probabilities = torch.nn.functional.softmax(logits[0, -1, :], dim=-1)
                
                next_token = logits[:, -1, :].argmax(dim=1)
            else:
                logits = logits[0, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(
                    logits, top_k=top_k, top_p=top_p
                )

                # Sample from the filtered distribution
                probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
              

            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            decoded = tokenizer.decode(next_token, skip_special_tokens=False)
           
            if stream:
                print(decoded, end="", flush=True)
          

            if next_token == tokenizer.eos_token_id:
                break

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

model = Transformer(
  TransformerConfig(
    embed_dim=256,
    num_decoder_layers=4,
    initializer_range=0.02,
    context_size=512,
    vocab_size=len(tokenizer),
    num_heads=4,
    num_kv_heads=4,
    grouped_heads=False,
    rotary=True,
    intermediate_size=None,
    rope_theta=10_000,
  )
)
model.eval()

model.load_state_dict(torch.load("ml/model.pt", map_location="cpu"))

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

message = """Transformers are"""

with torch.autocast(device_type=device, dtype=torch.float16):
  decoder_only_generate(
          message,
          model,
          tokenizer,
          sample=True,
          top_p=1.0,
          top_k=50,
          temperature=0.4,
          max_len=512,
          repetition_penalty=1.0,
          use_kv_cache=False,
          stream=True,
      )
  
print("")