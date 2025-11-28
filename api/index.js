import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import jwt from "jsonwebtoken";
import { audioToAudio, HfInference } from "@huggingface/inference";
import User from "./models/user.js";
import SavedOutfit from "./models/savedoutfit.js";
import Outfit from "./models/outfit.js";
import cosineSimilarity from "compute-cosine-similarity";
import { scrapeProduct } from './scraper.js'; // <-- Добавь эту строку
import Replicate from "replicate";
import multer from "multer";
import { GoogleAIFileManager } from "@google/generative-ai/server";
import { GoogleGenerativeAI } from "@google/generative-ai";
import fs from "fs"; // Для работы с файловой системой
import path from "path"; // Для путей
import ClothingItem from "./models/ClothingItem.js";
import "dotenv/config";
import { createClient } from '@supabase/supabase-js';
import axios from 'axios';





const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

const app = express();
const port = 3000;
const JWT_SECRET =
  "965de78b929b09f4693a231ab5934a910ea823d96d6ff5e33a4b18ed2c9c1f09";

// Настройка для сохранения временных видео
const upload = multer({ dest: "uploads/" });

// Настройка Google AI (Вставь свой новый ключ!)
const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);


app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));




const hf = new HfInference(process.env.HF_TOKEN);

const authenticateToken = (req, res, next) => {
  const token = req.headers["authorization"]?.split(" ")[1];
  if (!token) return res.status(401).json({ error: "No token provided" });

  jwt.verify(token, JWT_SECRET, (err, decoded) => {
    if (err) return res.status(403).json({ error: "Invalid token" });
    req.user = decoded;
    next();
  });
};

mongoose
  .connect("mongodb+srv://karimdzanovzoha:Abduzahid8@aiwardrobe.fah7ml3.mongodb.net/?appName=AIWardrobe")
  .then(() => console.log("Connected to MongoDB"))
  .catch((err) => console.log("Error connecting to MongoDb", err));

app.post("/register", async (req, res) => {
  try {
    const { email, password, username, gender, profileImage } = req.body;
    console.log("email", email);
    const existingUser = await User.findOne({ email });
    if (existingUser)
      return res.status(400).json({ error: "Email already exists" });
    const existingUsername = await User.findOne({ username });
    if (existingUsername)
      return res.status(400).json({ error: "Username already exists" });
    const user = new User({
      email,
      password,
      username,
      gender,
      profileImage,
      outfits: [],
    });

    console.log("user", user);

    await user.save();
    const token = jwt.sign({ id: user._id }, JWT_SECRET);
    res.status(201).json({ token });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    console.log("email", email);
    const user = await User.findOne({ email });
    if (!user || !(await user.comparePassword(password))) {
      return res.status(401).json({ error: "Invalid credentials" });
    }
    const token = jwt.sign({ id: user._id }, JWT_SECRET);
    res.json({ token });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/me", authenticateToken, async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select("-password");
    if (!user) return res.status(404).json({ error: "User not found" });
    res.json(user);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/save-outfit", authenticateToken, async (req, res) => {
  try {
    const { date, items, caption, occasion, visibility, isOotd } = req.body;
    const userId = req.user.id;

    let user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    const itemsWithImages = items?.map((item) => {
      if (!item || typeof item !== "object") {
        console.warn("Invalid item skipped", item);
        return null;
      }
      let imageUrl = item?.image;
      if (!imageUrl || !imageUrl.match(/^https?:\/\/res\.cloudinary\.com/)) {
        console.warn("Invalid or non-Cloudinary image URL:", imageUrl);
        return null; // Skip invalid URLs
      }
      return {
        id: item.id !== undefined || "null",
        type: item.type || "Unknown",
        image: imageUrl,
        x: item.x !== undefined ? item?.x : 0,
        y: item.y !== undefined ? item?.y : 0,
      };
    });

    const validItems = itemsWithImages.filter((item) => item !== null);

    if (validItems.length == 0) {
      return res.status(400).json({ error: "No valid items provided" });
    }

    const newOutfit = new SavedOutfit({
      userId: user._id,
      date,
      items: validItems,
      caption: caption || "",
      occasion: occasion || "",
      visibilty: visibility || "Everyone",
      isOotd: isOotd || false,
    });

    await newOutfit.save();

    user.outfits.push(newOutfit._id);
    await user.save();

    res.status(201).json({ outfit: newOutfit });
  } catch (err) {
    console.log("Error in save-outfit", err.message);
    res
      .status(500)
      .json({ error: "Internal server error", details: err.message });
  }
});





// Добавьте эти маршруты в ваш api/index.js 
// (вставьте ПЕРЕД app.listen)

// ========== WEATHER API ==========

const WEATHER_API_KEY = "0b0b523ea5bec9aef3883b17a3dbec98";


app.get("/weather", async (req, res) => {
  const { city, lat, lon } = req.query;

  try {
    let query;

    // Если переданы координаты
    if (lat && lon) {
      query = `${lat},${lon}`;
    }
    // Если передан город
    else if (city) {
      query = city;
    }
    // Если ничего не передано - используем Ташкент по умолчанию
    else {
      query = "Tashkent";
    }

    // WeatherAPI.com - бесплатный и работает сразу без задержки активации
    const url = `https://api.weatherapi.com/v1/current.json?key=${WEATHER_API_KEY}&q=${query}&aqi=no`;

    console.log("🌤️ Запрашиваю погоду для:", query);

    const response = await fetch(url);

    if (!response.ok) {
      const errorData = await response.json();
      console.error("❌ Weather API Error:", errorData);
      return res.status(response.status).json({
        error: errorData.error?.message || "Failed to fetch weather"
      });
    }

    const data = await response.json();

    console.log("✅ Weather fetched:", data.location.name, data.current.temp_c + "°C");

    // Форматируем ответ под ваш фронтенд
    res.json({
      temp: Math.round(data.current.temp_c),
      feels_like: Math.round(data.current.feelslike_c),
      description: data.current.condition.text,
      icon: data.current.condition.icon,
      city: data.location.name,
      humidity: data.current.humidity,
      wind_speed: data.current.wind_kph
    });

  } catch (error) {
    console.error("❌ Weather fetch error:", error.message);
    res.status(500).json({
      error: "Failed to fetch weather data",
      details: error.message
    });
  }
});

// Маршрут для координат
app.post("/weather/coords", async (req, res) => {
  const { latitude, longitude } = req.body;

  if (!latitude || !longitude) {
    return res.status(400).json({
      error: "Latitude and longitude are required"
    });
  }

  try {
    const query = `${latitude},${longitude}`;
    const url = `https://api.weatherapi.com/v1/current.json?key=${WEATHER_API_KEY}&q=${query}&aqi=no`;

    console.log("🌤️ Запрашиваю погоду по координатам:", query);

    const response = await fetch(url);

    if (!response.ok) {
      const errorData = await response.json();
      console.error("❌ Weather API Error:", errorData);
      return res.status(response.status).json({
        error: errorData.error?.message || "Failed to fetch weather"
      });
    }

    const data = await response.json();

    res.json({
      temp: Math.round(data.current.temp_c),
      feels_like: Math.round(data.current.feelslike_c),
      description: data.current.condition.text,
      icon: data.current.condition.icon,
      city: data.location.name,
      humidity: data.current.humidity,
      wind_speed: data.current.wind_kph
    });

  } catch (error) {
    console.error("❌ Weather fetch error:", error.message);
    res.status(500).json({
      error: "Failed to fetch weather data",
      details: error.message
    });
  }
});

// Дополнительный маршрут для получения погоды по координатам
app.post("/weather/coords", async (req, res) => {
  const { latitude, longitude } = req.body;

  if (!latitude || !longitude) {
    return res.status(400).json({
      error: "Latitude and longitude are required"
    });
  }

  try {
    const url = `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=${WEATHER_API_KEY}&units=metric`;

    const response = await fetch(url);

    if (!response.ok) {
      const errorData = await response.json();
      console.error("❌ Weather API Error:", errorData);
      return res.status(response.status).json({
        error: errorData.message || "Failed to fetch weather"
      });
    }

    const data = await response.json();

    res.json({
      temp: Math.round(data.main.temp),
      feels_like: Math.round(data.main.feels_like),
      description: data.weather[0].description,
      icon: data.weather[0].icon,
      city: data.name,
      humidity: data.main.humidity,
      wind_speed: data.wind.speed
    });

  } catch (error) {
    console.error("❌ Weather fetch error:", error.message);
    res.status(500).json({
      error: "Failed to fetch weather data",
      details: error.message
    });
  }
});









app.get("/save-outfit/user/:userId", authenticateToken, async (req, res) => {
  try {
    const userId = req.params.userId;
    if (req.user.id !== userId) {
      return res.status(403).json({ error: "Unauthorized access" });
    }
    const user = await User.findById(userId).populate("outfits");
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    res.status(200).json(user.outfits);
  } catch (error) {
    console.error("Error fetching outfits", error);
    res
      .status(500)
      .json({ error: "Internal server error", details: err.message });
  }
});



// Сохранение списка вещей (Batch Save)
// Роут для сохранения вещей с генерацией картинок
app.post("/wardrobe/add-batch", authenticateToken, async (req, res) => {
  try {
    const { items } = req.body;
    const userId = req.user.id;

    if (!items || !Array.isArray(items) || items.length === 0) {
      return res.status(400).json({ error: "No items provided" });
    }

    console.log(`🎨 Начинаю обработку ${items.length} вещей через Supabase...`);

    const itemsWithImages = await Promise.all(items.map(async (item) => {
      let finalImageUrl = "https://via.placeholder.com/300?text=No+Image";

      try {
        // А. Генерируем промпт
        const prompt = `A professional studio photography of a ${item.color} ${item.style} ${item.itemType} (${item.description}), isolated on clean white background, flat lay, fashion catalog style, high quality, realistic, no shadows`;

        // Б. Просим Replicate создать картинку
        const output = await replicate.run(
          "black-forest-labs/flux-schnell",
          {
            input: {
              prompt: prompt,
              aspect_ratio: "1:1",
              output_format: "jpg",
              output_quality: 80
            }
          }
        );

        // В. Если картинка есть -> Скачиваем и заливаем в Supabase
        if (output && output[0]) {
          const replicateUrl = output[0];

          // 1. Скачиваем картинку как ArrayBuffer
          const imageResponse = await axios.get(replicateUrl, { responseType: 'arraybuffer' });
          const buffer = Buffer.from(imageResponse.data, 'binary');

          // 2. Генерируем уникальное имя файла
          const fileName = `${userId}/${Date.now()}_${Math.random().toString(36).substring(7)}.jpg`;

          // 3. Загружаем в Supabase Storage
          const { data, error } = await supabase
            .storage
            .from('wardrobe_images') // Имя твоего бакета
            .upload(fileName, buffer, {
              contentType: 'image/jpeg',
              upsert: false
            });

          if (error) {
            console.error("Supabase error:", error);
            throw error;
          }

          // 4. Получаем публичную ссылку
          const { data: publicUrlData } = supabase
            .storage
            .from('wardrobe_images')
            .getPublicUrl(fileName);

          finalImageUrl = publicUrlData.publicUrl;
        }

      } catch (genError) {
        console.error(`Ошибка с вещью ${item.itemType}:`, genError.message);
      }

      // Возвращаем объект для MongoDB
      return {
        userId: userId,
        type: item.itemType,
        color: item.color,
        season: item.season,
        style: item.style,
        description: item.description,
        imageUrl: finalImageUrl
      };
    }));

    // Сохраняем в MongoDB
    const savedItems = await ClothingItem.insertMany(itemsWithImages);

    // Обновляем юзера
    await User.findByIdAndUpdate(userId, {
      $push: { outfits: { $each: savedItems.map(i => i._id) } }
    });

    console.log(`✅ Успешно сохранено: ${savedItems.length} шт.`);
    res.status(201).json({ success: true, count: savedItems.length });

  } catch (err) {
    console.error("Critical Error:", err);
    res.status(500).json({ error: err.message });
  }
});




const generateEmbedding = async (text) => {
  const response = await hf.featureExtraction({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    inputs: text,
  });
  return response;
};
const seedData = async () => {
  try {
    const count = await Outfit.countDocuments();
    if (count === 0) {
      const outfits = [
        {
          occasion: "date",
          style: "casual",
          items: ["White linen shirt", "Dark jeans", "Loafers"],
          image: "https://i.pinimg.com/736x/b2/6e/c7/b26ec7bc30ca9459b918ae8f7bf66305.jpg",
        },
        {
          occasion: "date",
          style: "elegant",
          items: ["White flared pants", "sandals", "sunglasses"],
          image: "https://i.pinimg.com/736x/8c/61/12/8c6112457ae46fa1e0aea8b8f5ed18ec.jpg",
        },
        {
          occasion: "coffee",
          style: "casual",
          items: ["cropped t-shirt", "wide-leg beige trousers", "Samba sneakers"],
          image: "https://i.pinimg.com/736x/d7/2d/26/d72d268ca4ff150db1db560b25afb843.jpg",
        },
        {
          occasion: "interview",
          style: "formal",
          items: ["Light blue shirt", "wide-leg jeans", "Silver wristwatch"],
          image: "https://i.pinimg.com/736x/1c/50/bc/1c50bcef1b46efe5db4008252ea8cfa5.jpg",
        },
        {
          occasion: "beach",
          style: "beach",
          items: ["brown T shirt", "beige shorts", "Sunglasses"],
          image: "https://i.pinimg.com/1200x/86/57/59/8657592bd659335ffd081fdab10b87a4.jpg",
        },
      ];

      for (const outfit of outfits) {
        const text = `${outfit.occasion} ${outfit.style} ${outfit.items.join(", ")}`;
        const embedding = await generateEmbedding(text);
        await new Outfit({ ...outfit, embedding }).save();
      }
      console.log("✅ Database seeded with", outfits.length, "outfits");
    } else {
      console.log("✅ Database already has", count, "outfits");
    }
  } catch (err) {
    console.error("❌ Seeding failed:", err.message);
  }
}

seedData();

const normalizeQuery = (query) => {
  const synonyms = {
    "coffee date": "coffee date",
    "dinner date": "date",
    "job interview": "interview",
    work: "interview",
    casual: "casual",
    formal: "formal",
    outfit: "",
    "give me": "",
    a: "",
    an: "",
    for: "",
  };

  let normalized = query.toLowerCase();
  Object.keys(synonyms).forEach((key) => {
    normalized = normalized.replace(
      new RegExp(`\\b${key}\\b`, "gi"),
      synonyms[key]
    );
  });
  return [...new Set(normalized.trim().split(/\s+/).filter(Boolean))].join(" ");
};

app.get("/smart-search", async (req, res) => {
  const { query } = req.query;
  if (!query) return res.status(400).json({ error: "Query required" });

  try {
    const normalizedQuery = normalizeQuery(query);
    const queryEmbedding = await generateEmbedding(normalizedQuery);
    const outfits = await Outfit.find();

    const MIN_SIMILARITY = query.length > 20 ? 0.3 : 0.4;

    let scored = outfits
      .map((o) => {
        const score = cosineSimilarity(queryEmbedding, o.embedding);
        return { ...o.toObject(), score };
      })
      .filter((o) => o.score >= MIN_SIMILARITY)
      .sort((a, b) => b.score - a.score);

    if (scored.length === 0) {
      const queryTerms = normalizedQuery.split(" ");
      scored = outfits
        .filter((o) =>
          queryTerms.some(
            (term) =>
              // 👇 ДОБАВИЛИ ЗАЩИТУ: ( ... || "")
              (o.occasion || "").toLowerCase().includes(term) ||
              (o.style || "").toLowerCase().includes(term) ||
              (o.items || []).some((item) => (item || "").toLowerCase().includes(term))
          )
        )
        .map((o) => ({ ...o.toObject(), score: 0.1 }));
    }

    res.json(scored.slice(0, 5));
  } catch (err) {
    console.error("🔴 ОШИБКА ИИ:", err); // <--- ДОБАВИТЬ ЭТУ СТРОКУ
    res.status(500).json({ error: err.message });
  }
});

// 👇 ВСТАВИТЬ ЭТО ПЕРЕД app.listen
app.post("/ai-chat", async (req, res) => {
  const { query } = req.body;
  console.log("📨 Запрос:", query);

  try {
    const result = await hf.chatCompletion({
      // 👇 МЕНЯЕМ МОДЕЛЬ ЗДЕСЬ. 72B слишком тяжелая, ставим 7B или Llama 3
      model: "meta-llama/Meta-Llama-3-8B-Instruct",
      messages: [
        { role: "system", content: "You are a helpful fashion stylist. Keep answers short and fun with emojis." },
        { role: "user", content: query }
      ],
      max_tokens: 500, // Чуть увеличим токены
      temperature: 0.7 // Креативность
    });

    // Проверка, есть ли ответ
    if (result && result.choices && result.choices.length > 0) {
      console.log("🤖 Ответ:", result.choices[0].message.content);
      res.json({ text: result.choices[0].message.content });
    } else {
      throw new Error("AI вернул пустой ответ");
    }

  } catch (err) {
    console.error("❌ Ошибка HF:", err.message);
    // Возвращаем понятную ошибку на телефон, а не просто 500
    res.status(500).json({ error: "AI model is busy, try again later." });
  }
});









// 👇 ВСТАВЬТЕ СЮДА ВАШ ТОКЕН ОТ REPLICATE (начинается на r8_...)
const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

app.post("/try-on", async (req, res) => {
  const { human_image, garment_image, description } = req.body;

  console.log("🎨 Начинаю примерку...");
  console.log("Человек:", human_image);
  console.log("Одежда:", garment_image);

  try {
    // Используем модель IDM-VTON (она очень качественная)
    const output = await replicate.run(
      "cuuupid/idm-vton:906425dbca90663ff54276248397db52027860a241f03fad3e5a04127a7570c8",
      {
        input: {
          human_img: human_image, // Ссылка на фото человека
          garm_img: garment_image, // Ссылка на фото одежды
          garment_des: description || "clothing",
          crop: false,
          seed: 42,
          crop: false,
          steps: 30
        }
      }
    );

    console.log("✅ Готово:", output);
    res.json({ image: output }); // Replicate возвращает ссылку на результат

  } catch (error) {
    console.error("Ошибка Replicate:", error);
    res.status(500).json({ error: error.message });
  }
});


// 👇 РОУТ ДЛЯ ВИДЕО-СКАНИРОВАНИЯ
app.post("/scan-wardrobe", upload.single("video"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No video file uploaded" });
    }

    console.log("🎥 Видео получено:", req.file.path);
    console.log("⏳ Начинаю обработку ИИ...");

    // 1. Загружаем видео в Google AI
    const uploadResult = await fileManager.uploadFile(req.file.path, {
      mimeType: req.file.mimetype,
      displayName: "Wardrobe Scan",
    });

    console.log(`✅ Видео загружено в облако: ${uploadResult.file.uri}`);

    // 2. Ждем, пока видео обработается (Google требует пару секунд)
    let file = await fileManager.getFile(uploadResult.file.name);
    let pollCount = 0;
    const maxPolls = 90; // Max 3 minutes (90 * 2 seconds)

    while (file.state === "PROCESSING" && pollCount < maxPolls) {
      console.log(`...обработка видео (${pollCount * 2}s)...`);
      await new Promise((resolve) => setTimeout(resolve, 2000));
      file = await fileManager.getFile(uploadResult.file.name);
      pollCount++;
    }

    if (pollCount >= maxPolls) {
      throw new Error("Video processing timeout - try shorter video");
    }

    if (file.state === "FAILED") {
      throw new Error("Google failed to process video.");
    }

    // 3. Спрашиваем Gemini
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    const result = await model.generateContent([
      {
        fileData: {
          mimeType: uploadResult.file.mimeType,
          fileUri: uploadResult.file.uri,
        },
      },
      {
        text: `Analyze this video of a wardrobe. Detect each distinct clothing item shown.
        Ignore background, furniture, or hands.
        Return a JSON ARRAY where each object has:
        - itemType (e.g. Shirt, Pants, Dress)
        - color (dominant color)
        - season (Summer, Winter, All)
        - style (Casual, Formal, Sport)
        - description (short description)
        
        OUTPUT RAW JSON ONLY. NO MARKDOWN. NO \`\`\`.`,
      },
    ]);

    // 4. Чистим ответ и сохраняем
    const responseText = result.response.text();
    console.log("🤖 Ответ ИИ:", responseText);

    const cleanJson = responseText.replace(/```json|```/g, "").trim();
    const items = JSON.parse(cleanJson);

    // 5. (Опционально) Тут можно сразу сохранить их в MongoDB
    // Но пока просто вернем на клиент, чтобы показать пользователю

    // Удаляем временный файл с сервера
    fs.unlinkSync(req.file.path);

    res.json({ detectedItems: items });

  } catch (error) {
    console.error("Video Scan Error:", error);
    res.status(500).json({ error: error.message });
  }
});





app.listen(port, '0.0.0.0', () => {
  console.log(`Server running on port ${port}`);
});
