import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  TextInput,
  ScrollView,
  Image,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import axios from 'axios';

const { width } = Dimensions.get('window');

const API_URL = 'https://aiwardrobe-ivh4.onrender.com';

// Occasion cards with icons and gradients
const occasions = [
  { id: 'date', label: 'Date', icon: 'heart', colors: ['#FF6B9D', '#FE8DC3'] },
  { id: 'coffee', label: 'Coffee', icon: 'cafe', colors: ['#8B5E3C', '#A67C52'] },
  { id: 'interview', label: 'Interview', icon: 'briefcase', colors: ['#4F46E5', '#6366F1'] },
  { id: 'party', label: 'Party', icon: 'sparkles', colors: ['#F59E0B', '#FBBF24'] },
  { id: 'gym', label: 'Gym', icon: 'fitness', colors: ['#10B981', '#34D399'] },
  { id: 'casual', label: 'Casual', icon: 'shirt', colors: ['#6B7280', '#9CA3AF'] },
  { id: 'beach', label: 'Beach', icon: 'sunny', colors: ['#06B6D4', '#22D3EE'] },
  { id: 'formal', label: 'Formal', icon: 'medal', colors: ['#8B5CF6', '#A78BFA'] },
];

const AIOutfitGenerator = () => {
  const navigation = useNavigation();
  const [selectedOccasion, setSelectedOccasion] = useState('');
  const [styleInput, setStyleInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [outfits, setOutfits] = useState([]);
  const [error, setError] = useState('');

  const generateOutfits = async () => {
    if (!selectedOccasion && !styleInput.trim()) {
      setError('Please select an occasion or describe your style');
      return;
    }

    setLoading(true);
    setError('');
    setOutfits([]);

    try {
      console.log('🎨 Generating outfits:', { selectedOccasion, styleInput });

      const response = await axios.post(`${API_URL}/api/generate-outfits`, {
        occasion: selectedOccasion,
        stylePreferences: styleInput.trim(),
        limit: 5
      }, {
        timeout: 30000
      });

      if (response.data.success && response.data.outfits.length > 0) {
        setOutfits(response.data.outfits);
        console.log(`✅ Found ${response.data.outfits.length} outfits`);
      } else {
        setError('No matching outfits found. Try different preferences!');
      }
    } catch (err) {
      console.error('Outfit generation error:', err);
      setError('Failed to generate outfits. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: '#fff' }}>
      <LinearGradient
        colors={['#ffffff', '#f9fafb', '#f3f4f6']}
        style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}
      />

      <SafeAreaView style={{ flex: 1 }}>
        {/* Header */}
        <View style={{ flexDirection: 'row', alignItems: 'center', paddingHorizontal: 20, paddingVertical: 16 }}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={{ marginRight: 16 }}>
            <Ionicons name="chevron-back" size={28} color="#1a1a1a" />
          </TouchableOpacity>
          <Text style={{ fontSize: 24, fontWeight: '800', color: '#1a1a1a', flex: 1 }}>
            AI Stylist
          </Text>
          <Ionicons name="sparkles" size={24} color="#F59E0B" />
        </View>

        <ScrollView showsVerticalScrollIndicator={false}>
          {/* Hero Section */}
          <View style={{ paddingHorizontal: 20, paddingTop: 8, paddingBottom: 24 }}>
            <Text style={{ fontSize: 16, color: '#6b7280', lineHeight: 24 }}>
              Tell me about your occasion and style preferences. I'll create the perfect outfit for you! ✨
            </Text>
          </View>

          {/* Occasion Selector */}
          <View style={{ paddingHorizontal: 20, marginBottom: 24 }}>
            <Text style={{ fontSize: 18, fontWeight: '700', color: '#1a1a1a', marginBottom: 16 }}>
              Choose Occasion
            </Text>
            <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 12 }}>
              {occasions.map((occasion) => (
                <TouchableOpacity
                  key={occasion.id}
                  onPress={() => setSelectedOccasion(occasion.id)}
                  style={{ width: (width - 52) / 2 }}
                >
                  <LinearGradient
                    colors={selectedOccasion === occasion.id ? occasion.colors : ['#f9fafb', '#f3f4f6']}
                    style={{
                      paddingVertical: 20,
                      borderRadius: 16,
                      alignItems: 'center',
                      borderWidth: 2,
                      borderColor: selectedOccasion === occasion.id ? 'transparent' : '#e5e7eb',
                    }}
                  >
                    <Ionicons
                      name={occasion.icon as any}
                      size={32}
                      color={selectedOccasion === occasion.id ? '#fff' : '#1a1a1a'}
                    />
                    <Text
                      style={{
                        marginTop: 8,
                        fontSize: 14,
                        fontWeight: '600',
                        color: selectedOccasion === occasion.id ? '#fff' : '#1a1a1a',
                      }}
                    >
                      {occasion.label}
                    </Text>
                  </LinearGradient>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Style Input */}
          <View style={{ paddingHorizontal: 20, marginBottom: 24 }}>
            <Text style={{ fontSize: 18, fontWeight: '700', color: '#1a1a1a', marginBottom: 12 }}>
              Describe Your Style
            </Text>
            <View
              style={{
                backgroundColor: '#fff',
                borderRadius: 16,
                padding: 16,
                borderWidth: 2,
                borderColor: '#e5e7eb',
                minHeight: 120,
              }}
            >
              <TextInput
                placeholder="e.g., minimalist, comfortable, neutral colors, elegant..."
                placeholderTextColor="#9ca3af"
                value={styleInput}
                onChangeText={setStyleInput}
                multiline
                style={{
                  fontSize: 16,
                  color: '#1a1a1a',
                  flex: 1,
                  textAlignVertical: 'top',
                }}
                maxLength={200}
              />
            </View>
            <Text style={{ fontSize: 12, color: '#9ca3af', marginTop: 8 }}>
              {styleInput.length}/200 characters
            </Text>
          </View>

          {/* Generate Button */}
          <View style={{ paddingHorizontal: 20, marginBottom: 24 }}>
            <TouchableOpacity
              onPress={generateOutfits}
              disabled={loading}
              activeOpacity={0.8}
            >
              <LinearGradient
                colors={['#1a1a1a', '#000000']}
                style={{
                  paddingVertical: 18,
                  borderRadius: 16,
                  alignItems: 'center',
                  flexDirection: 'row',
                  justifyContent: 'center',
                }}
              >
                {loading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <>
                    <Ionicons name="sparkles-outline" size={20} color="#fff" style={{ marginRight: 8 }} />
                    <Text style={{ fontSize: 18, fontWeight: '700', color: '#fff' }}>
                      Generate Outfits
                    </Text>
                  </>
                )}
              </LinearGradient>
            </TouchableOpacity>
          </View>

          {/* Error Message */}
          {error && (
            <View style={{ paddingHorizontal: 20, marginBottom: 24 }}>
              <View style={{ backgroundColor: '#FEF2F2', padding: 16, borderRadius: 12, borderLeftWidth: 4, borderLeftColor: '#EF4444' }}>
                <Text style={{ color: '#DC2626', fontSize: 14 }}>{error}</Text>
              </View>
            </View>
          )}

          {/* Results */}
          {outfits.length > 0 && (
            <View style={{ paddingHorizontal: 20, marginBottom: 40 }}>
              <Text style={{ fontSize: 20, fontWeight: '700', color: '#1a1a1a', marginBottom: 16 }}>
                Your Perfect Outfits ({outfits.length})
              </Text>

              {outfits.map((outfit, index) => (
                <View
                  key={outfit.id}
                  style={{
                    marginBottom: 24,
                    backgroundColor: '#fff',
                    borderRadius: 20,
                    overflow: 'hidden',
                    shadowColor: '#000',
                    shadowOffset: { width: 0, height: 4 },
                    shadowOpacity: 0.1,
                    shadowRadius: 12,
                    elevation: 5,
                  }}
                >
                  {/* Main Image */}
                  <Image
                    source={{ uri: outfit.mainImage }}
                    style={{ width: '100%', height: 400 }}
                    resizeMode="cover"
                  />

                  {/* Match Score Badge */}
                  <View
                    style={{
                      position: 'absolute',
                      top: 16,
                      right: 16,
                      backgroundColor: 'rgba(0,0,0,0.7)',
                      paddingHorizontal: 12,
                      paddingVertical: 6,
                      borderRadius: 20,
                    }}
                  >
                    <Text style={{ color: '#fff', fontSize: 12, fontWeight: '600' }}>
                      {Math.round(outfit.matchScore * 100)}% Match
                    </Text>
                  </View>

                  <View style={{ padding: 20 }}>
                    {/* Description */}
                    <Text style={{ fontSize: 16, color: '#1a1a1a', marginBottom: 12, lineHeight: 24 }}>
                      {outfit.description}
                    </Text>

                    {/* Items */}
                    <View style={{ marginBottom: 16 }}>
                      <Text style={{ fontSize: 14, fontWeight: '600', color: '#6b7280', marginBottom: 8 }}>
                        Items Included:
                      </Text>
                      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginLeft: -4 }}>
                        {outfit.items.map((item, idx) => (
                          <View
                            key={idx}
                            style={{
                              marginRight: 12,
                              alignItems: 'center',
                              width: 80,
                            }}
                          >
                            <Image
                              source={{ uri: item.image }}
                              style={{
                                width: 80,
                                height: 80,
                                borderRadius: 12,
                                marginBottom: 6,
                              }}
                            />
                            <Text
                              style={{
                                fontSize: 11,
                                color: '#6b7280',
                                textAlign: 'center',
                              }}
                              numberOfLines={2}
                            >
                              {item.name}
                            </Text>
                          </View>
                        ))}
                      </ScrollView>
                    </View>

                    {/* Styling Tips */}
                    <View
                      style={{
                        backgroundColor: '#F9FAFB',
                        padding: 12,
                        borderRadius: 12,
                        borderLeftWidth: 3,
                        borderLeftColor: '#4F46E5',
                      }}
                    >
                      <Text style={{ fontSize: 12, color: '#4b5563', lineHeight: 18 }}>
                        💡 <Text style={{ fontWeight: '600' }}>Styling Tip:</Text> {outfit.stylingTips}
                      </Text>
                    </View>
                  </View>
                </View>
              ))}
            </View>
          )}
        </ScrollView>
      </SafeAreaView>
    </View>
  );
};

export default AIOutfitGenerator;
