// Comprehensive locale fix: correct all structural issues and placeholder values
const fs = require('fs');
const path = require('path');

const enPath = path.join(__dirname, 'i18n', 'locales', 'en.json');
const ruPath = path.join(__dirname, 'i18n', 'locales', 'ru.json');
const uzPath = path.join(__dirname, 'i18n', 'locales', 'uz.json');

const en = JSON.parse(fs.readFileSync(enPath, 'utf8'));
const ru = JSON.parse(fs.readFileSync(ruPath, 'utf8'));
const uz = JSON.parse(fs.readFileSync(uzPath, 'utf8'));

function removePlaceholders(obj) {
  if (Array.isArray(obj)) {
    return obj.map(item => removePlaceholders(item));
  }
  if (obj && typeof obj === 'object') {
    const result = {};
    for (const [k, v] of Object.entries(obj)) {
      const cleaned = removePlaceholders(v);
      if (typeof cleaned === 'string' && cleaned.startsWith('[EN] ') && !cleaned.includes('[object Object]')) {
        result[k] = cleaned.slice(5).trim();
      } else if (typeof cleaned === 'string' && cleaned.startsWith('[EN]')) {
        result[k] = 'TODO';
      } else {
        result[k] = cleaned;
      }
    }
    return result;
  }
  return obj;
}

// 1. Fix en.json: restore styleGoals to original, remove stray [EN] placeholders
en.styleGoals = {
  alreadyActive: "Already Active",
  goalAlreadyProgress: "This goal is already in progress!",
  goalAchieved: "🎉 Goal Achieved!",
  congratulationsCompleted: "Congratulations! You've completed",
  challengeComplete: "🏆 Challenge Complete!",
  amazingCompleted: "Amazing! You've completed",
  title: "Style Goals",
  subtitle: "Track your fashion journey",
  activeGoals: "Active Goals",
  completed: "Completed"
};

// 2. Add proper English values for keys that were synced with [EN] placeholders
// Style goals extra keys
const styleGoalsExtra = {
  challenges: "Challenges",
  yourStyleGoals: "Your Style Goals",
  weeklyChallenges: "Weekly Challenges",
  completedText: "Completed!",
  startGoal: "Start Goal",
  logToday: "Log Today",
  challengeCompleteText: "Challenge Complete!",
  acceptChallenge: "Accept Challenge",
  available: [
    { title: "Build a Capsule Wardrobe", unit: "versatile items", description: "Curate 30 versatile pieces that mix and match" },
    { title: "Shop More Sustainably", unit: "sustainable purchases", description: "Make 10 conscious fashion choices this month" },
    { title: "Add More Color", unit: "colorful items", description: "Step out of your comfort zone with 5 colorful pieces" },
    { title: "Embrace Minimalism", unit: "items decluttered", description: "Declutter 20 items you no longer wear" },
    { title: "Elevate Work Style", unit: "work outfits", description: "Create 7 polished work outfit combinations" },
    { title: "Plan Complete Outfits", unit: "outfits planned", description: "Plan 14 complete outfits for the next 2 weeks" }
  ],
  weeklyChallengesList: [
    { title: "No Repeat Week", description: "Wear different outfits every day this week" },
    { title: "Monochrome Monday", description: "Create a single-color outfit on Monday" },
    { title: "Accessory Focus", description: "Add a new accessory to each outfit this week" },
    { title: "Wardrobe Rediscovery", description: "Wear 3 items you haven't worn in months" }
  ]
};

// Add to en.json if missing
Object.entries(styleGoalsExtra).forEach(([k, v]) => {
  if (!en.styleGoals[k]) en.styleGoals[k] = v;
});

// Auth section
if (en.auth && !en.auth.createAccount) en.auth.createAccount = "Create Account";
if (en.auth && !en.auth.welcome) en.auth.welcome = "Welcome";
if (en.auth && !en.auth.getStarted) en.auth.getStarted = "Get Started";

// Auth signUp subsection
if (en.auth && en.auth.signUp) {
  if (!en.auth.signUp.pleaseEnterValidEmail) en.auth.signUp.pleaseEnterValidEmail = "Please enter a valid email";
  if (!en.auth.signUp.passwordRequirements) en.auth.signUp.passwordRequirements = "Password Requirements";
  if (!en.auth.signUp.passwordRequirementsText) en.auth.signUp.passwordRequirementsText = "Your password must be at least 8 characters with uppercase, lowercase, and a number.";
  if (!en.auth.signUp.invalidUsername) en.auth.signUp.invalidUsername = "Invalid Username";
  if (!en.auth.signUp.usernameRequirements) en.auth.signUp.usernameRequirements = "3-30 chars, letters/numbers/underscores only";
  if (!en.auth.signUp.invalidGender) en.auth.signUp.invalidGender = "Invalid Gender";
  if (!en.auth.signUp.genderOptions) en.auth.signUp.genderOptions = "Please select a valid gender option";
  if (!en.auth.signUp.gender) en.auth.signUp.gender = "Gender";
  if (!en.auth.signUp.preferNotToSay) en.auth.signUp.preferNotToSay = "Prefer not to say";
  if (!en.auth.signUp.skip) en.auth.signUp.skip = "Skip";
  if (!en.auth.signUp.profileImageUrlOptional) en.auth.signUp.profileImageUrlOptional = "Profile Image URL (optional)";
  if (!en.auth.signUp.creatingAccount) en.auth.signUp.creatingAccount = "Creating Account...";
}

// designRoom extra keys
if (en.designRoom && !en.designRoom.scanVideo) en.designRoom.scanVideo = "Scan Video";
if (en.designRoom && !en.designRoom.stickers) en.designRoom.stickers = "Stickers";
if (en.designRoom && !en.designRoom.background) en.designRoom.background = "Background";
if (en.designRoom && !en.designRoom.emptyState) en.designRoom.emptyState = "Add clothing to start designing";
if (en.designRoom && !en.designRoom.foundItems) en.designRoom.foundItems = "✅ Found {{count}} clothing items!";
if (en.designRoom && !en.designRoom.failedSaveWardrobe) en.designRoom.failedSaveWardrobe = "Failed to save item to wardrobe";
if (en.designRoom && !en.designRoom.saved) en.designRoom.saved = "Saved! 🎉";
if (en.designRoom && !en.designRoom.itemsSaved) en.designRoom.itemsSaved = "items saved to your wardrobe!";
if (en.designRoom && !en.designRoom.next) en.designRoom.next = "Next";

// newOutfit
if (en.newOutfit && !en.newOutfit.failedToSaveOutfit) en.newOutfit.failedToSaveOutfit = "Failed to save outfit:";

// emailOnboarding
if (en.emailOnboarding && !en.emailOnboarding.gmailConnected) en.emailOnboarding.gmailConnected = "Gmail connected successfully (Simulation)";
if (en.emailOnboarding && !en.emailOnboarding.failedConnectEmail) en.emailOnboarding.failedConnectEmail = "Failed to connect email. Please try again.";
if (en.emailOnboarding && !en.emailOnboarding.notConnected) en.emailOnboarding.notConnected = "Not Connected";
if (en.emailOnboarding && !en.emailOnboarding.connectEmailFirst) en.emailOnboarding.connectEmailFirst = "Please connect your email first.";
if (en.emailOnboarding && !en.emailOnboarding.scanCompleteMessage) en.emailOnboarding.scanCompleteMessage = "Found {{itemsDetected}} clothing items from {{receiptsFound}} receipts.";

// myCloset
if (en.myCloset && !en.myCloset.failedDeleteItem) en.myCloset.failedDeleteItem = "Failed to delete item from wardrobe";

// wardrobe extras for uz
if (en.wardrobe && !en.wardrobe.tryAgain) en.wardrobe.tryAgain = "Try Again";
if (en.wardrobe && !en.wardrobe.addItem) en.wardrobe.addItem = "Add Item";

// wardrobeVideo extras
if (en.wardrobeVideo && !en.wardrobeVideo.outfit) en.wardrobeVideo.outfit = "Outfit {{index}}";
if (en.wardrobeVideo && !en.wardrobeVideo.itemsSaved) en.wardrobeVideo.itemsSaved = "{{count}} item(s) saved to your wardrobe. They will sync automatically.";
if (en.wardrobeVideo && !en.wardrobeVideo.foundItems) en.wardrobeVideo.foundItems = "Found {{count}} items";
if (en.wardrobeVideo && !en.wardrobeVideo.itemsCount) en.wardrobeVideo.itemsCount = "{{count}} items";

// aiOutfitmaker (was in uz.json but not en.json)
if (!en.aiOutfitmaker) {
  en.aiOutfitmaker = {
    cannotSave: "Cannot Save",
    noValidItems: "This outfit has no valid items to save.",
    saved: "Saved",
    outfitSavedCloset: "Outfit saved to your closet."
  };
}

// review extras
if (en.review && !en.review.itemsAdded) en.review.itemsAdded = "Added {{count}} items. Now generating photos for them.";
if (en.review && !en.review.saveFailed) en.review.saveFailed = "Failed to save items: {{error}}";
if (en.review && !en.review.itemsFound) en.review.itemsFound = "Found {{count}} items";
if (en.review && !en.review.addToWardrobe) en.review.addToWardrobe = "Add to Wardrobe ({{count}})";

// extras missing in en.json
if (en.trialExpired && !en.trialExpired.havePromoCode) en.trialExpired.havePromoCode = "I have a promo code";

// tabs extras
if (en.tabs && !en.tabs.discover) en.tabs.discover = "Discover";
if (en.tabs && !en.tabs.add) en.tabs.add = "Add";
if (en.tabs && !en.tabs.design) en.tabs.design = "Design";
if (en.tabs && !en.tabs.home) en.tabs.home = "Home";

// styles extras
if (!en.styles) en.styles = {};
if (!en.styles.dior) en.styles.dior = { name: "Christian Dior", description: "Elegant, feminine, high fashion" };
if (!en.styles.armani) en.styles.armani = { name: "Giorgio Armani", description: "Minimalist, refined, relaxed cut" };
if (!en.styles.lauren) en.styles.lauren = { name: "Ralph Lauren", description: "Preppy, American classic, sporty elegance" };

// admin.categories
if (en.admin && !en.admin.categories) {
  en.admin.categories = {
    tops: "Tops",
    bottoms: "Bottoms",
    shoes: "Shoes",
    dresses: "Dresses",
    outerwear: "Outerwear"
  };
}
if (en.admin && en.admin.garmentTypes && !en.admin.garmentTypes.outfit) {
  en.admin.garmentTypes.outfit = "Outfit";
}
if (en.admin && en.admin.guide && !en.admin.guide.saveFailed) {
  en.admin.guide.saveFailed = "Failed to save guide content";
}

// Add missing language names
if (en.language && !en.language.english) en.language.english = "English";
if (en.language && !en.language.russian) en.language.russian = "Russian";
if (en.language && !en.language.uzbek) en.language.uzbek = "Uzbek";

// Add missing wearLog.dayStreak
if (en.wearLog && !en.wearLog.dayStreak) en.wearLog.dayStreak = "{{count}} day streak";

// Add outfitGeneration.outfitLoggedFor
if (en.outfitGeneration && !en.outfitGeneration.outfitLoggedFor) {
  en.outfitGeneration.outfitLoggedFor = "Outfit logged for {{dateKey}}.";
}

// home extras
if (!en.home.title) en.home.title = "Looks";
if (!en.home.dailyBrief) en.home.dailyBrief = "Daily Brief";
if (!en.home.stylistInsight) en.home.stylistInsight = "Stylist Insight";
if (!en.home.askStylist) en.home.askStylist = "Ask Stylist";
if (!en.home.todaysSuggestion) en.home.todaysSuggestion = "Today's Suggestion";
if (!en.home.askAI) en.home.askAI = "Ask AI";
if (!en.home.yourWeek) en.home.yourWeek = "Your Week";
if (!en.home.planner) en.home.planner = "Planner";
if (!en.home.noEssentials) en.home.noEssentials = "No Essentials Available";
if (!en.home.businessCasual) en.home.businessCasual = "Business Casual";
if (!en.home.dinner) en.home.dinner = "Dinner Outfit";
if (!en.home.itemAddedToWardrobe) en.home.itemAddedToWardrobe = "{{itemName}} added to wardrobe";
if (!en.home.addItemToWardrobe) en.home.addItemToWardrobe = "Add {{itemName}} to wardrobe";

// quickActions extras
if (!en.quickActions.menu) en.quickActions.menu = "Menu";
if (!en.quickActions.profile) en.quickActions.profile = "Profile";

// profile extras
if (!en.profile.shareProfile) en.profile.shareProfile = "Share Profile";
if (!en.profile.followers) en.profile.followers = "Followers";
if (!en.profile.following) en.profile.following = "Following";
if (!en.profile.tabs) en.profile.tabs = { clothes: "Clothes", outfits: "Outfits", collections: "Collections" };
if (!en.profile.categories) en.profile.categories = { all: "All", tops: "Tops", bottoms: "Bottoms", shoes: "Shoes", outerwear: "Outerwear" };
if (!en.profile.noClothes) en.profile.noClothes = "No clothes in this category";

// aiChat extras
if (!en.aiChat.greeting) en.aiChat.greeting = "Hi! I'm your AI stylist. How can I help you today?";

// aiTryOn extras
const aiTryOnExtra = {
  heroTitle: "See how an outfit looks on you",
  heroSubtitle: "Upload a full-length photo. AI will dress you in any clothes to see how it looks.",
  step1Label: "1. Your full-length photo",
  step1Hint: "Stand so your full height is visible in the frame for best results.",
  step2Label: "2. Clothes to try on",
  step3Label: "3. Preview",
  fullLengthPhoto: "Add full-length photo",
  fullLengthHint: "Camera or gallery — full body is best",
  takePhoto: "Take Photo",
  choosePhoto: "Choose from Gallery",
  cancel: "Cancel",
  permissionTitle: "Need Access",
  cameraPermission: "Allow camera access for full-length photos.",
  photoPermission: "Allow photo access to choose a full-length photo.",
  upload: "Upload",
  myWardrobe: "My Wardrobe",
  selectItem: "Select Item",
  loadingWardrobe: "Loading wardrobe...",
  noWardrobeItems: "No items in wardrobe yet",
  scanWardrobe: "Scan Wardrobe",
  selected: "selected",
  generating: "Generating new look...",
  takesTime: "(Takes ~20 seconds)",
  resultHere: "Result will appear here",
  generate: "Generate Try-On",
  processing: "Processing...",
  savedTitle: "Saved!",
  savedMessage: "Try-on result saved to your wardrobe!",
  viewWardrobe: "View Wardrobe",
  saveToWardrobe: "Save to Wardrobe",
  saving: "Saving...",
  saveFailed: "Failed to save. Try again.",
  demoTitle: "Demo",
  demoMessage: "AI service not configured. Showing demo result.",
  errorTitle: "Error",
  errorMessage: "Failed to generate try-on.",
  loginRequired: "Login required for AI try-on.",
  model: "Model",
  digitalModelTitle: "Your Digital Model",
  digitalModelPro: "PRO",
  digitalModelDescription: "Create images and try on clothes on your avatar.",
  upgradeToPro: "Upgrade to Pro",
  errors: {
    missingPhotos: "Add a full-length photo and clothes to try on.",
    tryOnFailed: "Try-on failed. Check server console."
  }
};
Object.entries(aiTryOnExtra).forEach(([k, v]) => {
  if (en.aiTryOn && en.aiTryOn[k] === undefined) en.aiTryOn[k] = v;
});
// Special case for the "try your self" key
if (en.aiTryOn && !en.aiTryOn["try your self"]) en.aiTryOn["try your self"] = "Try it yourself";

// outfitMaker extras
const outfitMakerExtra = {
  styleChat: "Style Chat",
  outfitAI: "Outfit AI",
  cannotSave: "Cannot Save",
  noValidItems: "No valid items to save",
  saved: "Saved",
  outfitSavedCloset: "Outfit saved to your closet",
  outfitExists: "Outfit already exists",
  replaceOutfit: "You already have an outfit for this date. Replace it?",
  replace: "Replace"
};
Object.entries(outfitMakerExtra).forEach(([k, v]) => {
  if (en.outfitMaker && en.outfitMaker[k] === undefined) en.outfitMaker[k] = v;
});

// paywall extras
if (en.paywall && !en.paywall.restoreSuccessful) en.paywall.restoreSuccessful = "Restore Successful";
if (en.paywall && !en.paywall.premiumActivated) en.paywall.premiumActivated = "Your Pro subscription has been restored.";
if (en.paywall && !en.paywall.subscriptionUnavailable) en.paywall.subscriptionUnavailable = "Subscriptions are currently unavailable. Please try again later.";

// trialCountdown extras
if (en.trialCountdown && !en.trialCountdown.hoursLeft) en.trialCountdown.hoursLeft = "{{hours}}h left in your free trial — Upgrade now";
if (en.trialCountdown && !en.trialCountdown.daysLeft) en.trialCountdown.daysLeft = "{{days}} day{{suffix}} left in your free trial";

// promo extras
if (en.promo && !en.promo.enterCodeLabel) en.promo.enterCodeLabel = "PROMO CODE";
if (en.promo && !en.promo.trialActivated) en.promo.trialActivated = "Trial Activated!";
if (en.promo && !en.promo.trialActivatedMessage) en.promo.trialActivatedMessage = "Your {{days}}-day free trial has been activated. Enjoy full Pro access!";

// featureLock extras
if (en.featureLock && !en.featureLock.unlockWith) en.featureLock.unlockWith = "Unlock with {{tier}}";

// weeklyInsights extras
if (en.weeklyInsights && !en.weeklyInsights.utilization50) en.weeklyInsights.utilization50 = "You wore {{percent}}% of your wardrobe this month. Try adding forgotten items.";
if (en.weeklyInsights && !en.weeklyInsights.utilizationLow) en.weeklyInsights.utilizationLow = "Only {{percent}}% of your wardrobe was worn this month. There's lots to rediscover!";
if (en.weeklyInsights && !en.weeklyInsights.unwornItems) en.weeklyInsights.unwornItems = "{{count}} unworn items";
if (en.weeklyInsights && !en.weeklyInsights.unwornItemsSubtext) en.weeklyInsights.unwornItemsSubtext = "These items haven't been worn in 30 days. Try one this week!";
if (en.weeklyInsights && !en.weeklyInsights.emptySubtext) en.weeklyInsights.emptySubtext = "Start scanning and logging outfits to see style insights";

// flashSales extras
if (en.flashSales && !en.flashSales.upToOff) en.flashSales.upToOff = "Up to {{discount}}% OFF";

// completeYourLook extras
if (en.completeYourLook && !en.completeYourLook.addCategory) en.completeYourLook.addCategory = "Add {{category}}";

// aiOutfitCreator extras
if (en.aiOutfitCreator && !en.aiOutfitCreator.selectEvent) en.aiOutfitCreator.selectEvent = "Select {{event}}";

// clothingEditor extras
if (en.clothingEditor && !en.clothingEditor.freePlanLimit) en.clothingEditor.freePlanLimit = "Free plan stores {{limit}} items. Upgrade to Pro for unlimited wardrobe.";

// outfitLogForm extras
if (en.outfitLogForm && !en.outfitLogForm.top) en.outfitLogForm.top = "Top";
if (en.outfitLogForm && !en.outfitLogForm.bottoms) en.outfitLogForm.bottoms = "Bottoms";
if (en.outfitLogForm && !en.outfitLogForm.shoes) en.outfitLogForm.shoes = "Shoes";

// Add layers outfit extras  
if (en.layeredOutfit && !en.layeredOutfit.noLayerItems) en.layeredOutfit.noLayerItems = "No {{layer}} items";

// aiThinking extras
if (en.aiThinking && !en.aiThinking.synthesizingDna) en.aiThinking.synthesizingDna = "Synthesizing {{style}} DNA";

// signIn extras
if (en.signIn && !en.signIn.title) en.signIn.title = "Sign In";
if (en.signIn && !en.signIn.email) en.signIn.email = "Email";
if (en.signIn && !en.signIn.password) en.signIn.password = "Password";
if (en.signIn && !en.signIn.signIn) en.signIn.signIn = "Sign In";

// admin.guide.saveFailed
if (en.admin && en.admin.guide && !en.admin.guide.saveFailed) en.admin.guide.saveFailed = "Failed to save guide content";

// admin manage extras
if (en.admin && en.admin.manage && !en.admin.manage.deleteConfirm) en.admin.manage.deleteConfirm = 'Delete "{{name}}" from shop?';

// wardrobe extras
if (en.wardrobe && !en.wardrobe.addItem) en.wardrobe.addItem = "Add Item";

// 3. Fix ru.json: remove [EN] placeholders and add tripPlanner arrays
ru.tripPlanner.available = [
  { title: "Создать капсульный гардероб", unit: "универсальные вещи", description: "Подберите 30 универсальных вещей, которые сочетаются друг с другом" },
  { title: "Покупать более устойчиво", unit: "устойчивые покупки", description: "Сделайте 10 осознанных модных выборов в этом месяце" },
  { title: "Добавить больше цвета", unit: "цветные вещи", description: "Выйдите из зоны комфорта с 5 цветными вещами" },
  { title: "Принять минимализм", unit: "убранные вещи", description: "Уберите 20 вещей, которые вы больше не носите" },
  { title: "Поднять рабочий стиль", unit: "рабочие образы", description: "Создайте 7 элегантных рабочих комбинаций образов" },
  { title: "Планировать полные образы", unit: "запланированные образы", description: "Запланируйте 14 полных образов на следующие 2 недели" }
];
ru.tripPlanner.weeklyChallengesList = [
  { title: "Неделя без повторов", description: "Носите разные образы каждый день на этой неделе" },
  { title: "Монохромный понедельник", description: "Создайте одноцветный образ в понедельник" },
  { title: "Фокус на аксессуарах", description: "Добавьте новый аксессуар к каждому образу на этой неделе" },
  { title: "Переоткрытие гардероба", description: "Носите 3 вещи, которые вы не носили месяцами" }
];

// 4. Fix uz.json
uz.tripPlanner.available = [
  { title: "Kapsula shkaf yaratish", unit: "ko'p qirrali narsalar", description: "Bir-biri bilan mos keladigan 30 ko'p qirrali narsani tanlang" },
  { title: "Barqarorroq xarid qiling", unit: "barqaror xaridlar", description: "Bu oy 10 ta ongli moda tanlovini qiling" },
  { title: "Ko'proq rang qo'shing", unit: "rangli narsalar", description: "5 ta rangli narsa bilan qulay zonadan chiqing" },
  { title: "Minimalizmni qabul qiling", unit: "o'chirilgan narsalar", description: "Kiymaydigan 20 ta narsani o'chiring" },
  { title: "Ish uslubini ko'taring", unit: "ish liboslari", description: "7 ta sayqallangan ish libosi kombinatsiyasini yarating" },
  { title: "To'liq liboslarni rejalashtiring", unit: "rejalashtirilgan liboslar", description: "Keyingi 2 hafta uchun 14 ta to'liq libosni rejalashtiring" }
];
uz.tripPlanner.weeklyChallengesList = [
  { title: "Takrorlanmas hafta", description: "Bu hafta har kuni boshqa libos kiying" },
  { title: "Monokrom dushanba", description: "Dushanba bir rangli libos yarating" },
  { title: "Aksessuar fokus", description: "Bu hafta har bir libosga yangi aksessuar qo'shing" },
  { title: "Shkafni qayta kashf qilish", description: "Oylardan kiyilmagan 3 ta narsani kiying" }
];

// 5. Fix uz.json: add wardrobe.tryAgain
if (!uz.wardrobe) uz.wardrobe = {};
uz.wardrobe.tryAgain = "Qayta urinib ko'ring";

// 6. Add ru translations for styleGoals.available and styleGoals.weeklyChallengesList (already there from original)

// 7. Remove leftover [EN] placeholders from ru with proper translations
// These are keys from uz.json that got synced to en.json and then to ru.json
ru.auth = ru.auth || {};
ru.auth.createAccount = "Создать аккаунт";
ru.auth.welcome = "Добро пожаловать";
ru.auth.getStarted = "Начать";
ru.auth.signUp = ru.auth.signUp || {};
ru.auth.signUp.pleaseEnterValidEmail = "Пожалуйста, введите действительный email";
ru.auth.signUp.passwordRequirements = "Требования к паролю";
ru.auth.signUp.passwordRequirementsText = "Ваш пароль должен содержать не менее 8 символов с заглавной, строчной буквой и цифрой.";
ru.auth.signUp.invalidUsername = "Неверное имя пользователя";
ru.auth.signUp.usernameRequirements = "3-30 символов, только буквы/цифры/подчеркивания";
ru.auth.signUp.invalidGender = "Неверный пол";
ru.auth.signUp.genderOptions = "Пожалуйста, выберите вариант пола";
ru.auth.signUp.gender = "Пол";
ru.auth.signUp.preferNotToSay = "Предпочитаю не указывать";
ru.auth.signUp.skip = "Пропустить";
ru.auth.signUp.profileImageUrlOptional = "URL изображения профиля (необязательно)";
ru.auth.signUp.creatingAccount = "Создание аккаунта...";

ru.designRoom = ru.designRoom || {};
ru.designRoom.failedSaveWardrobe = "Не удалось сохранить вещь в гардероб";
ru.designRoom.saved = "Сохранено! 🎉";
ru.designRoom.itemsSaved = "вещей сохранено в ваш гардероб!";

ru.emailOnboarding = ru.emailOnboarding || {};
ru.emailOnboarding.gmailConnected = "Gmail успешно подключен (Симуляция)";
ru.emailOnboarding.failedConnectEmail = "Не удалось подключить почту. Попробуйте снова.";
ru.emailOnboarding.notConnected = "Не подключено";
ru.emailOnboarding.connectEmailFirst = "Пожалуйста, сначала подключите почту.";

ru.newOutfit = ru.newOutfit || {};
ru.newOutfit.failedToSaveOutfit = "Не удалось сохранить образ:";

ru.myCloset = ru.myCloset || {};
ru.myCloset.failedDeleteItem = "Не удалось удалить предмет из гардероба";

ru.aiOutfitmaker = ru.aiOutfitmaker || {};
ru.aiOutfitmaker.cannotSave = "Невозможно сохранить";
ru.aiOutfitmaker.noValidItems = "В этом образе нет действительных предметов для сохранения";
ru.aiOutfitmaker.saved = "Сохранено";
ru.aiOutfitmaker.outfitSavedCloset = "Образ сохранен в гардероб";

// Add missing wardrobe.tryAgain for ru
if (!ru.wardrobe) ru.wardrobe = {};
ru.wardrobe.tryAgain = "Попробовать снова";

// 8. Also add the missing styleGoals arrays to ru + uz json
// (but the check script treats them at a flat level vs array level)

// 9. Fix the specific array key names ru.json lacks
// styleGoals.available and styleGoals.weeklyChallengesList as full arrays
if (!ru.styleGoals.available) {
  ru.styleGoals.available = [
    { title: "Создать капсульный гардероб", unit: "универсальные вещи", description: "Подберите 30 универсальных вещей, которые сочетаются друг с другом" },
    { title: "Покупать более устойчиво", unit: "устойчивые покупки", description: "Сделайте 10 осознанных модных выборов в этом месяце" },
    { title: "Добавить больше цвета", unit: "цветные вещи", description: "Выйдите из зоны комфорта с 5 цветными вещами" },
    { title: "Принять минимализм", unit: "убранные вещи", description: "Уберите 20 вещей, которые вы больше не носите" },
    { title: "Поднять рабочий стиль", unit: "рабочие образы", description: "Создайте 7 элегантных рабочих комбинаций образов" },
    { title: "Планировать полные образы", unit: "запланированные образы", description: "Запланируйте 14 полных образов на следующие 2 недели" }
  ];
}
if (!ru.styleGoals.weeklyChallengesList) {
  ru.styleGoals.weeklyChallengesList = [
    { title: "Неделя без повторов", description: "Носите разные образы каждый день на этой неделе" },
    { title: "Монохромный понедельник", description: "Создайте одноцветный образ в понедельник" },
    { title: "Фокус на аксессуарах", description: "Добавьте новый аксессуар к каждому образу на этой неделе" },
    { title: "Переоткрытие гардероба", description: "Носите 3 вещи, которые вы не носили месяцами" }
  ];
}
if (!uz.styleGoals.available) {
  uz.styleGoals.available = [
    { title: "Kapsula shkaf yaratish", unit: "ko'p qirrali narsalar", description: "Bir-biri bilan mos keladigan 30 ko'p qirrali narsani tanlang" },
    { title: "Barqarorroq xarid qiling", unit: "barqaror xaridlar", description: "Bu oy 10 ta ongli moda tanlovini qiling" },
    { title: "Ko'proq rang qo'shing", unit: "rangli narsalar", description: "5 ta rangli narsa bilan qulay zonadan chiqing" },
    { title: "Minimalizmni qabul qiling", unit: "o'chirilgan narsalar", description: "Kiymaydigan 20 ta narsani o'chiring" },
    { title: "Ish uslubini ko'taring", unit: "ish liboslari", description: "7 ta sayqallangan ish libosi kombinatsiyasini yarating" },
    { title: "To'liq liboslarni rejalashtiring", unit: "rejalashtirilgan liboslar", description: "Keyingi 2 hafta uchun 14 ta to'liq libosni rejalashtiring" }
  ];
}
if (!uz.styleGoals.weeklyChallengesList) {
  uz.styleGoals.weeklyChallengesList = [
    { title: "Takrorlanmas hafta", description: "Bu hafta har kuni boshqa libos kiying" },
    { title: "Monokrom dushanba", description: "Dushanba bir rangli libos yarating" },
    { title: "Aksessuar fokus", description: "Bu hafta har bir libosga yangi aksessuar qo'shing" },
    { title: "Shkafni qayta kashf qilish", description: "Oylardan kiyilmagan 3 ta narsani kiying" }
  ];
}

// Remove [EN] prefix placeholders
// Clean any remaining string placeholders
[en, ru, uz].forEach((lang, i) => {
  const names = ['en', 'ru', 'uz'];
  const cleaned = removePlaceholders(lang);
  if (names[i] === 'en') Object.assign(en, cleaned);
  else if (names[i] === 'ru') Object.assign(ru, cleaned);
  else if (names[i] === 'uz') Object.assign(uz, cleaned);
});

// 10. Fix uz.json quality issues
// "bulot" -> "bulut" in privacy section
if (uz.privacyPolicy && uz.privacyPolicy.dataStorage && uz.privacyPolicy.dataStorage.body) {
  uz.privacyPolicy.dataStorage.body = uz.privacyPolicy.dataStorage.body.replace('bulot saqlashda', 'bulut saqlashda');
}

// "Clothing" left in English in wardrobe section
if (uz.wardrobe && uz.wardrobe.clothing === "Clothing") {
  uz.wardrobe.clothing = "Kiyim";
}

// Fix: "oxgi" -> "oxirgi" in uz.json 
if (uz.privacyPolicy && uz.privacyPolicy.lastUpdated) {
  uz.privacyPolicy.lastUpdated = uz.privacyPolicy.lastUpdated.replace('Oxgi', 'Oxirgi');
}
if (uz.termsOfService && uz.termsOfService.lastUpdated) {
  uz.termsOfService.lastUpdated = uz.termsOfService.lastUpdated.replace('Oxgi', 'Oxirgi');
}

// Write files
fs.writeFileSync(enPath, JSON.stringify(en, null, 2) + '\n');
fs.writeFileSync(ruPath, JSON.stringify(ru, null, 2) + '\n');
fs.writeFileSync(uzPath, JSON.stringify(uz, null, 2) + '\n');

console.log('All three locale files fixed!');
console.log('en.json keys:', Object.keys(en).length, 'top-level sections');
console.log('ru.json keys:', Object.keys(ru).length, 'top-level sections');
console.log('uz.json keys:', Object.keys(uz).length, 'top-level sections');
