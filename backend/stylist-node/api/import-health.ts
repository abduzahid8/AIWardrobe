import { STYLIST_TRANSPORT_MAX_BODY_BYTES } from '../transport/signature.mjs';
export default {
  fetch() {
    return Response.json({ status: 'ok', maxRequestBytes: STYLIST_TRANSPORT_MAX_BODY_BYTES });
  }
};
