"""
🎬 Outfit Change Detection v3 - AGGRESSIVE
Forces detection of multiple outfits in fashion videos.

Strategy:
1. Try similarity-based detection first
2. If only 1 outfit found, force split based on frame count
3. Assumes fashion videos have outfit changes
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
import base64

logger = logging.getLogger(__name__)


@dataclass
class OutfitSegment:
    """A segment of video containing one outfit"""
    start_frame: int
    end_frame: int
    outfit_id: int
    representative_frame: int


def decode_frame(base64_str: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ Frame decode error: {e}")
        return None


def compute_features(frame: np.ndarray) -> np.ndarray:
    """Extract color features from clothing regions"""
    if frame is None:
        return np.zeros(60)
    
    h, w = frame.shape[:2]
    
    # Upper body (20-50% height)
    y1_up, y2_up = int(h*0.2), int(h*0.5)
    x1, x2 = int(w*0.15), int(w*0.85)
    upper = frame[y1_up:y2_up, x1:x2]
    
    # Lower body (45-80% height)
    y1_lo, y2_lo = int(h*0.45), int(h*0.8)
    lower = frame[y1_lo:y2_lo, x1:x2]
    
    features = []
    
    for region in [upper, lower]:
        if region.size == 0:
            features.extend([0] * 30)
            continue
        
        # HSV histogram - focus on Hue (color)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [6], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [6], [0, 256]).flatten()
        
        # Normalize
        hist_h = hist_h / (hist_h.sum() + 1e-8)
        hist_s = hist_s / (hist_s.sum() + 1e-8)
        hist_v = hist_v / (hist_v.sum() + 1e-8)
        
        # Mean color
        mean_bgr = np.mean(region, axis=(0, 1)) / 255.0
        
        features.extend(hist_h)
        features.extend(hist_s)
        features.extend(hist_v)
        features.extend(mean_bgr)
    
    return np.array(features)


def compute_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    """Cosine similarity between feature vectors"""
    if len(f1) != len(f2) or len(f1) == 0:
        return 0.0
    
    dot = np.dot(f1, f2)
    norm = np.linalg.norm(f1) * np.linalg.norm(f2)
    
    return max(0, min(1, dot / norm)) if norm > 1e-8 else 0.0


def detect_outfit_changes(
    frames_base64: List[str],
    similarity_threshold: float = 0.35,  # Very low - aggressive detection
    min_expected_outfits: int = 2  # Force at least 2 outfits if video is long enough
) -> List[OutfitSegment]:
    """
    Detect outfit changes with aggressive fallback.
    
    If no changes detected but video has enough frames,
    force-split into multiple outfits.
    """
    n_frames = len(frames_base64)
    
    print(f"🎬 OUTFIT DETECTION v3: {n_frames} frames, threshold={similarity_threshold}")
    
    if n_frames == 0:
        return []
    
    if n_frames <= 2:
        return [OutfitSegment(0, n_frames-1, 1, 0)]
    
    # Decode and extract features
    features = []
    for i, b64 in enumerate(frames_base64):
        frame = decode_frame(b64)
        feat = compute_features(frame)
        features.append(feat)
    
    # Compare consecutive frames
    similarities = []
    print("   Frame comparisons:")
    for i in range(n_frames - 1):
        sim = compute_similarity(features[i], features[i+1])
        similarities.append(sim)
        print(f"     {i} → {i+1}: {sim:.3f} {'🔄' if sim < similarity_threshold else '✓'}")
    
    # Find change points
    change_points = [0]
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            change_points.append(i + 1)
            print(f"   ✅ CHANGE at frame {i+1} (sim={sim:.3f})")
    
    # If no changes found but we have enough frames, force split
    if len(change_points) == 1 and n_frames >= 4:
        # Use minimum/average similarity to find best split points
        min_sim_idx = np.argmin(similarities)
        min_sim = similarities[min_sim_idx]
        
        print(f"   ⚠️ No changes found, forcing split...")
        print(f"   Lowest similarity at frame {min_sim_idx+1}: {min_sim:.3f}")
        
        # Force split into segments
        if n_frames >= 6:
            # 3+ outfits - split at lowest similarity points
            sorted_indices = np.argsort(similarities)
            
            # Take top 2-3 lowest similarity points as splits
            num_splits = min(3, n_frames // 3)
            split_points = sorted(sorted_indices[:num_splits])
            
            change_points = [0]
            for sp in split_points:
                change_points.append(sp + 1)
            
            print(f"   Forced splits at frames: {change_points[1:]}")
        else:
            # 2 outfits - split at lowest similarity
            change_points = [0, min_sim_idx + 1]
            print(f"   Forced split at frame {min_sim_idx + 1}")
    
    # Build segments
    segments = []
    for i, start in enumerate(change_points):
        end = change_points[i+1] - 1 if i+1 < len(change_points) else n_frames - 1
        rep = (start + end) // 2
        
        segments.append(OutfitSegment(
            start_frame=start,
            end_frame=end,
            outfit_id=i + 1,
            representative_frame=rep
        ))
    
    print(f"🎬 RESULT: {len(segments)} outfit(s)")
    for seg in segments:
        print(f"   Outfit {seg.outfit_id}: frames {seg.start_frame}-{seg.end_frame}")
    
    return segments


def group_items_by_outfit(
    items: List[Dict],
    outfit_segments: List[OutfitSegment],
    total_frames: int
) -> List[Dict]:
    """Assign outfit_id to each item"""
    n_outfits = len(outfit_segments)
    n_items = len(items)
    
    if n_outfits <= 1 or n_items == 0:
        for item in items:
            item['outfit_id'] = 1
        return items
    
    # Distribute items across outfits
    # Strategy: items that appear in similar frame indices go together
    
    for item in items:
        frame_indices = item.get('frameIndices', [])
        
        if frame_indices:
            # Use first frame to determine outfit
            first_frame = frame_indices[0]
            
            for seg in outfit_segments:
                if seg.start_frame <= first_frame <= seg.end_frame:
                    item['outfit_id'] = seg.outfit_id
                    break
            else:
                item['outfit_id'] = 1
        else:
            # No frame info - assign evenly
            idx = items.index(item)
            items_per_outfit = max(1, n_items // n_outfits)
            outfit_idx = min(idx // items_per_outfit, n_outfits - 1)
            item['outfit_id'] = outfit_segments[outfit_idx].outfit_id
    
    return items
