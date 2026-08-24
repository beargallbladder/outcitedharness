Extract the UART configuration from the datasheet excerpt.

Return JSON only, with exactly these keys:

```json
{
  "peripheral": "USART1",
  "baud": 115200,
  "data_bits": 8,
  "parity": "none",
  "stop_bits": 1,
  "frame_bits": 10
}
```

Rules:

- `parity` must be one of: none, even, odd
- `frame_bits` is the number of bits on the wire for one character:
  1 start bit + data_bits + 1 if parity is not none + stop_bits
- Do not include markdown, commentary, or extra keys
