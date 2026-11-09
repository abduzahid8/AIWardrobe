import { STYLIST_TRANSPORT_MAX_BODY_BYTES } from '../../stylist-transport/signature.ts';
export default {
  fetch() {
    return Response.json({ status: 'ok', maxRequestBytes: STYLIST_TRANSPORT_MAX_BODY_BYTES });
  }
};
