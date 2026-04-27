// End-to-end smoke test for the BFL-powered mannequin-tryon edge function.
// Sends mannequin + a real garment image with descriptive metadata and
// writes the dressed result to disk.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const SUPABASE_URL = 'https://fyqpifmrsftsfqibhwhy.supabase.co';
const ANON = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5cXBpZm1yc2Z0c2ZxaWJod2h5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQzMDYyOTgsImV4cCI6MjA3OTg4MjI5OH0.dydnFn3lqub7qMo9uFfn5yUyY4Wr_eQPnsbvHWHwMTk';

function imgToDataUri(p) {
    const buf = fs.readFileSync(p);
    const ext = path.extname(p).slice(1).toLowerCase();
    const mime = ext === 'jpg' || ext === 'jpeg' ? 'image/jpeg' : 'image/png';
    return `data:${mime};base64,${buf.toString('base64')}`;
}

async function call(body) {
    const res = await fetch(`${SUPABASE_URL}/functions/v1/mannequin-tryon`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${ANON}`,
            'apikey': ANON,
        },
        body: JSON.stringify(body),
    });
    const text = await text_or_json(res);
    return { status: res.status, body: text };
}

async function text_or_json(res) {
    const t = await res.text();
    try { return JSON.parse(t); } catch { return t; }
}

async function dressOne({ mannequin, item, step, total, alreadyWearing }) {
    console.log(`\n→ Step ${step}/${total} ${item.label} "${item.name}"`);
    const t0 = Date.now();
    const submit = await call({
        action: 'submit',
        mannequin_image: mannequin,
        garment_image: item.image,
        garment: { type: item.type, label: item.label, name: item.name, description: item.description },
        already_wearing: alreadyWearing,
        step,
        total,
    });
    if (submit.status !== 200 || !submit.body?.success) {
        throw new Error(`submit failed: ${submit.status} ${JSON.stringify(submit.body).slice(0, 300)}`);
    }
    // NVIDIA NIM is synchronous — result is in the submit response.
    const resultUrl = submit.body.resultUrl;
    if (!resultUrl) throw new Error('No resultUrl in submit response');
    console.log(`  ✓ done in ${((Date.now() - t0) / 1000).toFixed(1)}s, method=${submit.body.methodUsed}`);
    return resultUrl;
}

async function main() {
    const mannequinPath = path.join(ROOT, 'assets/images/mannequin_front.png');
    let current = imgToDataUri(mannequinPath);
    console.log('Mannequin loaded:', mannequinPath, `(${(current.length / 1024) | 0} KB data URI)`);

    // Use bundled assets we know exist for stable testing.
    const items = [
        {
            label: 'top',
            type: 'upper_body',
            image: imgToDataUri(path.join(ROOT, 'assets/images/basic_white_tshirt.png')),
            name: 'Slim white crew-neck t-shirt',
            description: 'Slim-fit short sleeve cotton t-shirt, crew neck, regular hip-length, plain white.',
        },
        {
            label: 'pants',
            type: 'lower_body',
            image: imgToDataUri(path.join(ROOT, 'assets/images/basic_brown_pants.png')),
            name: 'Brown straight-leg trousers',
            description: 'Mid-rise straight-leg cotton trousers in brown, full-length to the ankle, regular fit.',
        },
        {
            label: 'shoes',
            type: 'shoes',
            image: imgToDataUri(path.join(ROOT, 'assets/images/basic_brown_loafers.png')),
            name: 'Brown leather loafers',
            description: 'Classic brown leather loafers, low-cut, slip-on dress shoes.',
        },
    ];

    const wearing = [];
    for (let i = 0; i < items.length; i++) {
        const it = items[i];
        current = await dressOne({
            mannequin: current,
            item: it,
            step: i + 1,
            total: items.length,
            alreadyWearing: [...wearing],
        });
        const out = path.join(__dirname, `tryon_bodyfit_step${i + 1}_${it.label}.png`);
        fs.writeFileSync(out, Buffer.from(current.split(',')[1], 'base64'));
        console.log(`  saved → ${out}`);
        wearing.push(it.description);
    }

    const final = path.join(__dirname, 'tryon_bodyfit_final.png');
    fs.writeFileSync(final, Buffer.from(current.split(',')[1], 'base64'));
    console.log(`\n✅ Final dressed mannequin saved → ${final}`);
}

main().catch((e) => {
    console.error('❌', e);
    process.exit(1);
});
