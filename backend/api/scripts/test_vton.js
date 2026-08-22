import "dotenv/config";
import fetch from 'node-fetch';
import Replicate from 'replicate';

const replicate = new Replicate({
    auth: process.env.REPLICATE_API_TOKEN
});

async function run() {
    try {
        console.log("Starting IDM-VTON test for ghost mannequin...");

        // Let's use a dummy garment URL and a dummy mannequin URL for a quick test
        const output = await replicate.run(
            "cuuupid/idm-vton:0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985",
            {
                input: {
                    garm_img: "https://replicate.delivery/pbxt/Kj1e6A7A6K5XJj9eP2NMBG5g9E2sE6kGZ2L2z2C/garm1.jpg",
                    human_img: "https://replicate.delivery/pbxt/Kj1e4m5H6j2XJj9eP2NMBG5g9E2sE6kGZ2L2z2C/human1.jpg", // We will replace this with a blank mannequin image in prod
                    garment_des: "professional studio photo of a jacket",
                    category: "upper_body",
                    crop: false,
                    steps: 20
                }
            }
        );
        console.log("SUCCESS! OUTPUT:", output);
    } catch (e) {
        console.log("ERROR:", e.message);
    }
}
run();
