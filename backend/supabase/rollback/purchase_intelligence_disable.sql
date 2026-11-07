-- Approval-gated rollback: retain all new personal/history data; stop exposure/writes.
-- Clients currently use local evidence and fallbacks, so existing app contracts remain valid.
begin;
revoke insert,update,delete on public.purchase_fit_evidence,public.purchase_fit_profiles,public.purchase_price_watches from authenticated;
revoke all on public.shop_merchants,public.shop_offers,public.shop_price_history from anon,authenticated;
alter table public.shop_offers disable trigger record_shop_offer_price;
commit;
