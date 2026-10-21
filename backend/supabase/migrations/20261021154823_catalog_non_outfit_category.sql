-- Keep imported non-outfit products out of clothing slots. Name-only garment
-- exceptions match the shared client/Edge classifier; no prices or images change.
begin;
set local lock_timeout = '5s';
set local statement_timeout = '10s';
create or replace function app_security.normalize_catalog_non_outfit_category()
returns trigger language plpgsql security invoker set search_path = ''
as $$
begin
    if lower(coalesce(new.name,'') || ' ' || coalesce(new.garment_type,'')) ~ '\m(watch(?:es)?|towels?|(?:suit|garment)[ -]+carriers?|key[ -]?rings?|keychains?|(?:hand|tote)?bags?|backpacks?|wallets?|card[ -]?holders?|purses?|belts?|scar(?:f|ves)|hats?|caps?|beanies?|jewell?ery|sunglasses|gloves?|cufflinks?|pocket[ -]?squares?|necklaces?|earrings?|bracelets?|rings?|umbrellas?|socks?|cases?|pouches?|perfumes?|fragrances?)\M'
       and lower(coalesce(new.name,'')) !~ '\m(t-?shirts?|tees?|shirts?|polos?|blouses?|dress(?:es)?|pants?|trousers?|jeans|shorts?|skirts?|chinos?|jackets?|coats?|blazers?|cardigans?|sweaters?|hoodies?|vests?|shoes?|loafers?|boots?|sneakers?|sandals?)\M' then
        new.category := 'accessories';
        new.garment_type := 'accessory';
    end if;
    return new;
end;
$$;
revoke all on function app_security.normalize_catalog_non_outfit_category() from public, anon, authenticated, service_role;
create trigger shop_catalog_normalize_non_outfit
before insert or update of name, category, garment_type on public.shop_catalog
for each row execute function app_security.normalize_catalog_non_outfit_category();
update public.shop_catalog
set category = 'accessories', garment_type = 'accessory'
where lower(coalesce(name,'') || ' ' || coalesce(garment_type,'')) ~ '\m(watch(?:es)?|towels?|(?:suit|garment)[ -]+carriers?|key[ -]?rings?|keychains?|(?:hand|tote)?bags?|backpacks?|wallets?|card[ -]?holders?|purses?|belts?|scar(?:f|ves)|hats?|caps?|beanies?|jewell?ery|sunglasses|gloves?|cufflinks?|pocket[ -]?squares?|necklaces?|earrings?|bracelets?|rings?|umbrellas?|socks?|cases?|pouches?|perfumes?|fragrances?)\M'
  and lower(coalesce(name,'')) !~ '\m(t-?shirts?|tees?|shirts?|polos?|blouses?|dress(?:es)?|pants?|trousers?|jeans|shorts?|skirts?|chinos?|jackets?|coats?|blazers?|cardigans?|sweaters?|hoodies?|vests?|shoes?|loafers?|boots?|sneakers?|sandals?)\M'
  and (category <> 'accessories' or garment_type <> 'accessory');
commit;
