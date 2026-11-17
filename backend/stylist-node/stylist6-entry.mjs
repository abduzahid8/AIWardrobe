/**
 * Stylist 6 cloud staging entry for the Vercel Node bundle.
 *
 * Everything the isolated `stylist6-generate-outfits` route needs is
 * re-exported here so esbuild can trace the full execution graph
 * (fashion ranking, visual hydration, NVIDIA critic, transport) into one
 * artifact. No Deno APIs, no dotenv, no local media: configuration arrives
 * exclusively through the route's environment.
 */
export {
  handleStylist6CloudGenerate,
  createCloudIo,
  mapCloudVisualStatus,
  projectWardrobeItem,
  fitVisualBytesCloud,
  STYLIST6_CLOUD_VERSION,
} from '../stylist6/cloud_handler.mjs';
