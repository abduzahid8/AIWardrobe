#!/bin/bash
# Fetch Zara men's product data and download images
# Uses Jina reader to bypass bot protection

mkdir -p /Users/zohidvohidjonov/Desktop/AIWardrobe/assets/images/zara
mkdir -p /Users/zohidvohidjonov/Desktop/AIWardrobe/assets/images/massimo-dutti

fetch_product() {
  local url="$1"
  local sku=$(echo "$url" | grep -oP 'p[0-9]{8}')
  echo "Fetching: $sku"
  
  local content=$(curl -s "https://r.jina.ai/${url}" \
    -H "User-Agent: Mozilla/5.0" \
    --connect-timeout 20 2>&1)
  
  # Extract product name (first h1 or title)
  local name=$(echo "$content" | grep -m1 -oP '(?<=# )[^|]+' | head -1 | sed 's/ *$//')
  # Extract main product image URL
  local img_url=$(echo "$content" | grep -oP 'https://static\.zara\.net/assets/public/[^"'"'"' )]+-p/[^"'"'"' )]+\.jpg' | head -1)
  
  if [ -n "$img_url" ]; then
    local filename="${sku}.jpg"
    local dest="/Users/zohidvohidjonov/Desktop/AIWardrobe/assets/images/zara/${filename}"
    curl -s -o "$dest" \
      -H "User-Agent: Mozilla/5.0" \
      -H "Referer: https://www.zara.com/" \
      "$img_url" --connect-timeout 10
    echo "Downloaded: $filename ($name)"
    echo "$sku|$name|$img_url" >> /tmp/zara_products.txt
  fi
}

# Process products from each category
# Linen Shirts
fetch_product "https://www.zara.com/us/en/100-linen-polo-shirt-p02634252.html"
fetch_product "https://www.zara.com/us/en/100-linen-regular-fit-shirt-p03090110.html"
fetch_product "https://www.zara.com/us/en/100-linen-relaxed-fit-shirt-p05070904.html"
fetch_product "https://www.zara.com/us/en/regular-fit-100-linen-shirt-p01063410.html"
fetch_product "https://www.zara.com/us/en/linen---cotton-shirt-p01063412.html"
fetch_product "https://www.zara.com/us/en/relaxed-fit-100-linen-shirt-with-pleated-pockets-p01195264.html"

# Polos & T-shirts
fetch_product "https://www.zara.com/us/en/knit-cotton-linen-blend-polo-p03920678.html"
fetch_product "https://www.zara.com/us/en/knit-textured-polo-shirt-p03332410.html"

# Trousers & Pants
fetch_product "https://www.zara.com/us/en/100-linen-relaxed-fit-pants-p02634254.html"
fetch_product "https://www.zara.com/us/en/100-linen-relaxed-fit-pants-p05070902.html"
fetch_product "https://www.zara.com/us/en/linen-cotton-blend-pleated-suit-pants-p04553594.html"
fetch_product "https://www.zara.com/us/en/flowy-pleated-pants-p04512653.html"

# Blazers
fetch_product "https://www.zara.com/us/en/100-linen-double-breasted-blazer-p04632333.html"
fetch_product "https://www.zara.com/us/en/100-linen-double-breasted-suit-blazer-p04286333.html"

# Shoes
fetch_product "https://www.zara.com/us/en/casual-leather-loafers-p12613720.html"
fetch_product "https://www.zara.com/us/en/barefoot-leather-sneaker-p12242720.html"

# Jeans
fetch_product "https://www.zara.com/us/en/basic-slim-fit-jeans-p00774454.html"
fetch_product "https://www.zara.com/us/en/baggy-fit-jeans-p04048407.html"

# Shorts
fetch_product "https://www.zara.com/us/en/100-linen-relaxed-fit-shorts-p05070903.html"

echo "Done! Products fetched:"
cat /tmp/zara_products.txt 2>/dev/null
