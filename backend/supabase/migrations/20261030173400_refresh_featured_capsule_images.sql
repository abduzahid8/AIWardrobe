-- Replace the reused flat sweater product photos with distinct men's
-- collection imagery for each recommended capsule.
update public.featured_capsules
set image_url = case id
    when '66da23a0-ab32-43e7-8c61-7208a0d6caf2' then
        'https://canali.vtexassets.com/assets/vtex.file-manager-graphql/images/d3f94624-638e-4535-b4b7-dfd5acf8bbc8___55fcdc9efef8053172d5934e355f9aff.jpg'
    when '71038c57-5e1c-48b7-acd8-5d6f1f9e689b' then
        'https://media.cos.com/sb/251958/1920x1200/4c5ff47968/1920x1200.jpg?imwidth=2160'
    when '07fa22e6-756e-4b16-af85-2dc5e141b517' then
        'https://static.massimodutti.net/assets/public/87ff/2deb/2ca64fb4ac7d/8de668c5be64/02600356710-o6/02600356710-o6.jpg?ts=1773330968888&f=auto&w=2560'
    else image_url
end,
updated_at = now()
where id in (
    '66da23a0-ab32-43e7-8c61-7208a0d6caf2',
    '71038c57-5e1c-48b7-acd8-5d6f1f9e689b',
    '07fa22e6-756e-4b16-af85-2dc5e141b517'
);
