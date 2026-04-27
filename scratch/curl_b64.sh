B64=$(base64 -i assets/images/mannequin_front.png | tr -d '\n')
curl -v -X POST "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev" \
  -H "Authorization: Bearer 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" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d "{
    \"prompt\": \"Dress the mannequin in a red jacket.\",
    \"image\": \"data:image/png;base64,${B64}\",
    \"width\": 1024,
    \"height\": 1024,
    \"steps\": 28,
    \"cfg_scale\": 3.5,
    \"seed\": 42
  }"
