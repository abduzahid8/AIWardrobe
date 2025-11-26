export const pants = [
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753853814/image6_erkdps.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859286/image8_s1e9g8.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859286/image7_wc3dxq.png",
  ].map((img, idx) => ({ id: idx + 1, image: img, type: "pants", gender: "f" }));
  
  export const tops = [
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859286/top1_lb1wb4.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859286/top2_lti11g.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859286/top5_kg0u6k.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859287/top4_xtudow.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859287/top7_fvxx5u.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859288/top9_sxatmr.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859287/top8_de4sz3.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859288/top6_tfdc0u.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753977657/Untitled_design_16_ppxbya.png"
  ].map((img, idx) => ({ id: idx + 1, image: img, type: "shirt", gender: "f" }));
  
  export const skirts = [
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859289/skirt3_oanqxj.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859290/skirt2_kdsmmj.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753859290/skirt1_wog7t9.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753975648/Untitled_design_7_zres8n.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753975914/Untitled_design_14_r1gzzf.png",
  ].map((img, idx) => ({ id: idx + 1, image: img, type: "skirts", gender: "f" }));
  
  export const mpants = [
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753975629/Untitled_design_3_syip4x.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753975630/Untitled_design_6_mt7zou.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753975765/Untitled_design_10_bq0etd.png",
  ].map((img, idx) => ({ id: idx + 1, image: img, type: "pants", gender: "m" }));
  
  export const mshirts = [
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753975660/Untitled_design_8_vmlkmo.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753975660/Untitled_design_9_pfvmv4.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753975885/Untitled_design_13_mfmdxd.png",
  ].map((img, idx) => ({ id: idx + 1, image: img, type: "shirt", gender: "m" }));
  
  export const shoes = [
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753975802/Untitled_design_11_p7t2us.png",
    "https://res.cloudinary.com/db1ccefar/image/upload/v1753975973/Untitled_design_15_kyl9nd.png",
  ].map((img, idx) => ({ id: idx + 1, image: img, type: "shoes", gender: "unisex" }));



  const features = [
    {
      title: "AI Suggestions",
      image:
        "https://i.pinimg.com/736x/2e/3d/d1/2e3dd14ac81b207ee6d86bc99ef576eb.jpg",
      screen: "AIChat",
    },
    {
      title: "AI Outfit Maker",
      image:
        "https://i.pinimg.com/736x/50/83/0e/50830e372ee844c1f429b8ef89e26fd1.jpg",
      screen: "AIOutfit",
    },
    {
      title: "AI Try On",
      image:
        "https://i.pinimg.com/736x/c2/78/95/c2789530a2dc8c9dbfd4aa5e2e70d608.jpg",
      screen: "AITryOn",
    },
    {
      title: "Color Analysis",
      image:
        "https://i.pinimg.com/736x/84/bf/ce/84bfce1e46977d50631c4ef2f72f83b1.jpg",
      screen: "ColorAnalysis",
    },
  ];
  
  const popularItems = [
    {
      username: "Trisha Wushres",
      profile: "https://randomuser.me/api/portraits/women/1.jpg",
      image:
        "https://res.cloudinary.com/db1ccefar/image/upload/v1753859289/skirt3_oanqxj.png",
      itemName: "Floral Skirt",
    },
    {
      username: "Anna Cris",
      profile: "https://randomuser.me/api/portraits/women/2.jpg",
      image:
        "https://res.cloudinary.com/db1ccefar/image/upload/v1753975629/Untitled_design_3_syip4x.png",
      itemName: "Mens Jeans",
    },
    {
      username: "Isabella",
      profile: "https://randomuser.me/api/portraits/women/3.jpg",
      image:
        "https://res.cloudinary.com/db1ccefar/image/upload/v1753975802/Untitled_design_11_p7t2us.png",
      itemName: "Shoes",
    },
  ];
  
  const initialStories = [
    {
      username: "Your OOTD",
      avatar: "https://picsum.photos/100/100?random=8",
      isOwn: true,
      viewed: false,
    },
    {
      username: "_trishwushres",
      avatar: "https://picsum.photos/100/100?random=10",
      isOwn: false,
      viewed: false,
    },
    {
      username: "myglam",
      avatar: "https://picsum.photos/100/100?random=11",
      isOwn: false,
      viewed: false,
    },
    {
      username: "stylist",
      avatar: "https://picsum.photos/100/100?random=12",
      isOwn: false,
      viewed: false,
    },
  ];
  