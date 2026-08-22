# NVIDIA NIM Support Ticket — Draft

**Submit at:** https://forums.developer.nvidia.com/c/ai-data-science/cloud-functions/

**Subject:** `flux.1-kontext-dev` hosted endpoint returns 500 Internal Server Error for valid requests

## Summary
The hosted NIM endpoint
`POST https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev`
returns `500 Internal Server Error` (empty body) for every valid request shape
I have tried, after a successful NVCF asset upload.

## API key
- Format: `nvapi-…` (32-byte personal API key created in build.nvidia.com)
- Account email: *(your build.nvidia.com email)*

## Reproduction (Node 20)

```js
// 1. Upload asset
const r1 = await fetch('https://api.nvcf.nvidia.com/v2/nvcf/assets', {
  method: 'POST',
  headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' },
  body: JSON.stringify({ contentType: 'image/png', description: 'test' }),
});
const { assetId, uploadUrl } = await r1.json();   // 200 OK

// 2. PUT bytes
await fetch(uploadUrl, {
  method: 'PUT',
  headers: { 'Content-Type': 'image/png', 'x-amz-meta-nvcf-asset-description': 'test' },
  body: fs.readFileSync('mannequin.png'),         // 200 OK, 15 KB
});

// 3. Invoke
const res = await fetch(
  'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev',
  {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${KEY}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
      'NVCF-INPUT-ASSET-REFERENCES': assetId,
    },
    body: JSON.stringify({
      prompt: 'Add a small red baseball cap.',
      image: `data:image/png;example_id,${assetId}`,
    }),
  },
);
// → 500 Internal Server Error (empty body)
```

## What I verified is correct
- POSTing without `prompt` returns `422 missing field "prompt"`.
- POSTing without `image` returns `422 missing field "image"`.
- POSTing `image: data:image/png;base64,…` returns `422 Expected: example_id, got: base64`.
- POSTing `image: data:image/png;asset_id,…` returns `422 Expected: example_id, got: asset_id`.
- POSTing `image: data:image/png;example_id,<assetId>` is **accepted** by validation
  and proceeds to inference, where it then **always 500s**.
- `num_inference_steps`, `seed`, `width`, `height`, `guidance_scale`, `cfg_scale` are all
  rejected with `422 extra_forbidden` — so the schema is just `{ prompt, image, aspect_ratio? }`.
- Multipart/form-data also returns 500.
- Variants `flux.1-kontext-pro` and `flux.1-kontext-max` return 404 (not hosted).

## Ask
1. Is the hosted `flux.1-kontext-dev` NIM currently healthy on the API catalog?
2. If yes, what additional header / payload field am I missing that the validator
   accepts but the inference layer needs?
3. Could you check the inference logs around the 500 for assetId
   `<paste-an-assetId-here>` made with API key prefix `nvapi-IWanUC…`?

Thanks!
