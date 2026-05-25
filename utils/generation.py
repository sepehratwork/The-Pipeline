import torch
import torch.nn.functional as F

def generate_completions(model, input_ids, attention_mask, max_new_tokens, pad_token_id, eos_token_id):
    model.eval()
    batch_size = input_ids.size(0)
    device = input_ids.device

    current_input_ids = input_ids
    current_attention_mask = attention_mask
    past_key_values = None

    generated_tokens = []
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )

            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            probs = F.softmax(next_token_logits, dim=-1)
            if torch.isnan(probs).any():
                probs = torch.where(torch.isnan(probs), torch.ones_like(probs) / probs.size(-1), probs)

            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_tokens = torch.where(finished, torch.tensor(pad_token_id, device=device), next_tokens)
            finished = finished | (next_tokens == eos_token_id)

            generated_tokens.append(next_tokens.unsqueeze(-1))
            if finished.all(): break

            current_input_ids = next_tokens.unsqueeze(-1)
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((batch_size, 1), dtype=torch.long, device=device)
            ], dim=1)

    model.train()
    if len(generated_tokens) == 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=device)
    return torch.cat(generated_tokens, dim=1)
