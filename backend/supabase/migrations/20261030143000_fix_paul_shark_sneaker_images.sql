-- Paul & Shark publishes the Club Riviera sneaker's front image as f3.
update public.inspo_look_products
set image_url = 'https://media-cdn.paulandshark.com/dam/ecomm/M/15318007-050-f3.jpg'
where id in ('paul-shark-fw26-01-shoes', 'paul-shark-fw26-02-shoes');
