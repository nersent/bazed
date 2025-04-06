import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from ml.dataset import SimpleDataset
from ml.model import Transformer, TransformerConfig

torch.set_float32_matmul_precision('high')

batch_size = 32
ctx_len = 512

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

model = Transformer(
  TransformerConfig(
    embed_dim=256,
    num_decoder_layers=4,
    initializer_range=0.02,
    context_size=ctx_len,
    vocab_size=len(tokenizer),
    num_heads=4,
    num_kv_heads=4,
    grouped_heads=False,
    rotary=True,
    intermediate_size=None,
    rope_theta=10_000,
  )
)

model.train()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# model = torch.compile(model, fullgraph=True)

dataset = SimpleDataset(path="ml/dataset", window_len=ctx_len)

dataloader = torch.utils.data.DataLoader(
  dataset,
  batch_size=batch_size,
  shuffle=True,
  num_workers=4,
  pin_memory=True,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def shift_logits_and_labels(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

    return shift_logits, shift_labels

def calc_loss(logits, labels):
    x = logits.view(-1, logits.size(-1))
    y = labels.view(-1)

    return torch.nn.functional.cross_entropy(x, y)

for batch in tqdm(dataloader):
  optimizer.zero_grad()

  input_ids = batch["input_ids"].to(device)
  labels = batch["labels"].to(device)

  with torch.autocast(device_type=device, dtype=torch.float16):
    x = model(input_ids)
    
    shift_logits, shift_labels = shift_logits_and_labels(x, labels)
    loss = calc_loss(shift_logits, shift_labels)
  
  loss.backward()
  optimizer.step()
  
print(f"Loss: {loss.item()}")
  
print("Saving model...")
torch.save(model.state_dict(), "ml/model.pt")
