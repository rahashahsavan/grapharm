# ریشه مشکل Indexing Error

## خلاصه:
`node_order` = ترتیب FORWARD masking ✅ درست است

## مشکل:
در `prepare_denoising_input`، خط 514 و 517:
```python
previous_mask[previous_nodes_original] = True
previous_x = graph_t_plus_1.x[previous_nodes_original]
```

`previous_nodes_original` یک **list Python** است، نه tensor!

## چرا مشکل ایجاد می‌کند؟
- `previous_nodes_original = node_order[t+1:]` → list Python
- PyTorch وقتی list می‌بیند، باید آن را به tensor تبدیل کند
- اگر اندیس‌ها خارج از محدوده باشند → CUDA assertion error

## راه‌حل:
تبدیل به tensor قبل از indexing:
```python
previous_nodes_tensor = torch.tensor(previous_nodes_original, dtype=torch.long, device=device)
previous_mask[previous_nodes_tensor] = True
previous_x = graph_t_plus_1.x[previous_nodes_tensor]
```

## Device Error:
همچنین مشکل device در خط 513:
```python
previous_mask = torch.zeros(..., device=device)
```
اگر `previous_nodes_original` روی device دیگری باشد، مشکل ایجاد می‌شود.


