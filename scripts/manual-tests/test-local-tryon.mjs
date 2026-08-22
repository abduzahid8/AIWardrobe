import fs from 'node:fs';
import path from 'node:path';

const MANNEQUIN_PATH = path.join(process.cwd(), 'assets', 'images', 'mannequin_front.png');
const OUT_PATH = path.join(process.cwd(), 'scripts', 'out', 'local-tryon-result.json');
const TOKEN = process.env.TEST_AUTH_TOKEN || 'dev-test-token';

const garments = [
  {
    label: 'top',
    type: 'upper_body',
    garment_image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg',
    name: 'Oxford Slim-Fit Shirt',
    description: 'Light blue button-down collar 100% cotton oxford long-sleeve shirt, slim fit',
  },
  {
    label: 'layer',
    type: 'upper_body',
    garment_image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/448034/item/usgoods_09_448034_3x4.jpg',
    name: 'AirSense Blazer',
    description: 'Navy blue ultra-light wool-like two-button single-breasted blazer jacket',
  },
  {
    label: 'pants',
    type: 'lower_body',
    garment_image: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/450251/item/usgoods_35_450251_3x4.jpg',
    name: 'Slim-Fit Chino Pants',
    description: 'Brown beige slim-fit Supima cotton stretch chino trousers, full-length',
  },
  {
    label: 'shoes',
    type: 'shoes',
    garment_image: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/484330/item/goods_32_484330_3x4.jpg',
    name: 'Combination Sneaker',
    description: 'White leather low-top sneakers with beige suede side panels and gum rubber sole',
  },
];

const mannequinB64 = fs.readFileSync(MANNEQUIN_PATH).toString('base64');
const mannequin_image = `data:image/png;base64,${mannequinB64}`;

const res = await fetch('http://127.0.0.1:3000/api/tryon/render', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${TOKEN}`,
  },
  body: JSON.stringify({ mannequin_image, garments, total: garments.length }),
});

const text = await res.text();
let data;
try {
  data = JSON.parse(text);
} catch {
  data = { raw: text };
}

fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
fs.writeFileSync(OUT_PATH, JSON.stringify({ status: res.status, data }, null, 2));
console.log(JSON.stringify({ status: res.status, outfile: OUT_PATH, success: data?.success, error: data?.error, methodUsed: data?.methodUsed }, null, 2));
