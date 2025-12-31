"""
Multi-Frame Clothing Analysis Module
Combines detections from multiple video frames for higher accuracy

Uses a voting and confidence aggregation system to:
- Eliminate false positives that only appear in 1-2 frames
- Boost confidence for items detected consistently
- Select the best quality cutout from multiple frames
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class MultiFrameAnalyzer:
    """
    Analyzes multiple video frames and combines detections for higher confidence.
    Uses voting, averaging, and quality selection to achieve maximum accuracy.
    
    Example usage:
        analyzer = MultiFrameAnalyzer()
        frame_results = [segment_frame(f) for f in frames]
        combined = analyzer.analyze_frames(frame_results)
    """
    
    def __init__(
        self, 
        min_frame_agreement: float = 0.6,  # 🎯 4-Layer Stack: 60% agreement required
        confidence_boost: float = 0.15
    ):
        """
        Initialize the multi-frame analyzer.
        
        Args:
            min_frame_agreement: Minimum fraction of frames that must agree on a detection
                                (e.g., 0.6 = item must appear in at least 60% of frames)
                                Updated to 0.6 as part of 4-Layer Assurance Stack
            confidence_boost: Maximum confidence boost for consistent detections
        """
        self.min_agreement = min_frame_agreement
        self.confidence_boost = confidence_boost
    
    def analyze_frames(
        self, 
        frame_results: List[Dict], 
        strategy: str = "voting"
    ) -> Dict[str, Any]:
        """
        Combine detections from multiple frames.
        
        Args:
            frame_results: List of segment-all results from each frame.
                          Each should have {"success": bool, "items": [...]}
            strategy: Combination strategy
                     - "voting": Keep items that appear in majority of frames
                     - "union": Keep all detected items (may have false positives)
                     - "intersection": Only keep items in ALL frames (may miss items)
        
        Returns:
            Combined detection result with:
            - items: List of confident detections with boosted confidence
            - framesAnalyzed: Number of frames processed
            - agreementScores: Per-item agreement scores
        """
        if not frame_results:
            return {
                "success": False,
                "items": [],
                "framesAnalyzed": 0,
                "error": "No frames provided"
            }
        
        # Filter successful results
        valid_results = [r for r in frame_results if r.get("success", False)]
        
        if not valid_results:
            return {
                "success": False,
                "items": [],
                "framesAnalyzed": len(frame_results),
                "error": "All frames failed processing"
            }
        
        if len(valid_results) == 1:
            # Single frame - return as-is
            result = valid_results[0].copy()
            result["framesAnalyzed"] = 1
            result["multiFrame"] = False
            return result
        
        n_frames = len(valid_results)
        
        # === COLLECT ALL DETECTIONS BY CATEGORY ===
        category_detections = defaultdict(list)  # category -> list of (frame_idx, item)
        
        for frame_idx, result in enumerate(valid_results):
            for item in result.get("items", []):
                category = item.get("category", "unknown")
                category_detections[category].append({
                    "frame": frame_idx,
                    "item": item,
                    "confidence": item.get("confidence", 0),
                    "cutoutImage": item.get("cutoutImage"),
                    "attributes": item.get("attributes")
                })
        
        logger.info(f"Multi-frame analysis: {n_frames} frames, {len(category_detections)} categories")
        
        # === APPLY STRATEGY ===
        if strategy == "voting":
            final_items = self._voting_strategy(category_detections, n_frames)
        elif strategy == "union":
            final_items = self._union_strategy(category_detections)
        elif strategy == "intersection":
            final_items = self._intersection_strategy(category_detections, n_frames)
        else:
            final_items = self._voting_strategy(category_detections, n_frames)
        
        return {
            "success": True,
            "items": final_items,
            "framesAnalyzed": n_frames,
            "multiFrame": True,
            "strategy": strategy
        }
    
    def _voting_strategy(
        self, 
        category_detections: Dict[str, List], 
        n_frames: int
    ) -> List[Dict]:
        """
        Keep items that appear in majority of frames.
        Select best quality instance and AGGREGATE all attributes for final result.
        """
        final_items = []
        min_appearances = max(1, int(n_frames * self.min_agreement))
        
        for category, detections in category_detections.items():
            appearances = len(detections)
            
            if appearances >= min_appearances:
                # Item appears in enough frames - include it
                agreement_ratio = appearances / n_frames
                
                # Find the best detection (highest confidence) for base item
                best_detection = max(detections, key=lambda d: d["confidence"])
                best_item = best_detection["item"].copy()
                
                # Boost confidence based on agreement
                original_conf = best_item.get("confidence", 0.5)
                boosted_conf = min(0.99, original_conf + (agreement_ratio * self.confidence_boost))
                best_item["confidence"] = round(boosted_conf, 3)
                
                # === AGGREGATE DETAILED ATTRIBUTES ACROSS ALL FRAMES ===
                all_items = [d["item"] for d in detections]
                all_confidences = [d["confidence"] for d in detections]
                
                # Aggregate specific type (Fashion-CLIP classification)
                best_item["specificType"] = self._aggregate_specific_types(all_items, all_confidences)
                
                # Aggregate colors with weighted voting
                best_item["colors"] = self._aggregate_colors(all_items, all_confidences)
                if best_item["colors"]:
                    best_item["primaryColor"] = best_item["colors"][0].get("name", best_item.get("primaryColor", "unknown"))
                    best_item["colorHex"] = best_item["colors"][0].get("hex", best_item.get("colorHex", "#000000"))
                
                # Aggregate patterns
                best_item["pattern"] = self._aggregate_patterns(all_items, all_confidences)
                
                # Aggregate materials
                best_item["material"] = self._aggregate_materials(all_items, all_confidences)
                
                # Aggregate detailed features (closures, collars, sleeves, etc.)
                best_item["features"] = self._aggregate_features(all_items, all_confidences)
                
                # Aggregate style info from Fashion-CLIP
                best_item["styleInfo"] = self._aggregate_style_info(all_items, all_confidences)
                
                # Add multi-frame metadata
                best_item["multiFrameData"] = {
                    "framesDetected": appearances,
                    "totalFrames": n_frames,
                    "agreement": round(agreement_ratio, 2),
                    "originalConfidence": round(original_conf, 3),
                    "confidenceBoost": round(boosted_conf - original_conf, 3)
                }
                
                final_items.append(best_item)
                logger.info(f"  ✅ {category}: {appearances}/{n_frames} frames (conf: {original_conf:.2f} → {boosted_conf:.2f})")
            else:
                logger.info(f"  🚫 {category}: only {appearances}/{n_frames} frames (need {min_appearances})")
        
        # Sort by confidence
        final_items.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return final_items
    
    def _aggregate_specific_types(
        self, 
        items: List[Dict], 
        confidences: List[float]
    ) -> Optional[str]:
        """
        Aggregate specific type classifications using weighted voting.
        Returns the most common specific type, weighted by confidence.
        """
        type_scores = defaultdict(float)
        
        for item, conf in zip(items, confidences):
            specific_type = item.get("specificType")
            if specific_type:
                type_scores[specific_type] += conf
        
        if not type_scores:
            return None
        
        # Return highest scoring type
        best_type = max(type_scores.items(), key=lambda x: x[1])
        return best_type[0]
    
    def _aggregate_colors(
        self, 
        items: List[Dict], 
        confidences: List[float]
    ) -> List[Dict]:
        """
        Aggregate colors across frames with weighted voting.
        Returns combined color palette with percentages.
        """
        color_scores = defaultdict(lambda: {"hex": None, "total": 0, "count": 0})
        
        for item, conf in zip(items, confidences):
            # Check for detailed colors
            attrs = item.get("attributes", {})
            colors = attrs.get("colors", [])
            
            if colors:
                for color in colors:
                    name = color.get("name", "unknown")
                    hex_val = color.get("hex", "#000000")
                    pct = color.get("percentage", 0.5)
                    
                    color_scores[name]["hex"] = hex_val
                    color_scores[name]["total"] += pct * conf
                    color_scores[name]["count"] += 1
            else:
                # Fallback to primaryColor
                primary = item.get("primaryColor")
                hex_val = item.get("colorHex", "#000000")
                if primary:
                    color_scores[primary]["hex"] = hex_val
                    color_scores[primary]["total"] += conf
                    color_scores[primary]["count"] += 1
        
        if not color_scores:
            return []
        
        # Calculate weighted percentages and sort
        result = []
        total_score = sum(c["total"] for c in color_scores.values())
        
        for name, data in color_scores.items():
            if total_score > 0:
                percentage = round(data["total"] / total_score, 3)
            else:
                percentage = 0
            
            result.append({
                "name": name,
                "hex": data["hex"],
                "percentage": percentage,
                "framesDetected": data["count"]
            })
        
        # Sort by percentage descending
        result.sort(key=lambda x: x["percentage"], reverse=True)
        return result[:5]  # Top 5 colors
    
    def _aggregate_patterns(
        self, 
        items: List[Dict], 
        confidences: List[float]
    ) -> Dict:
        """
        Aggregate pattern detections using weighted voting.
        """
        pattern_scores = defaultdict(lambda: {"confidence_sum": 0, "count": 0, "descriptions": []})
        
        for item, conf in zip(items, confidences):
            attrs = item.get("attributes", {})
            pattern = attrs.get("pattern", {})
            
            if pattern:
                pattern_type = pattern.get("type", "solid")
                pattern_conf = pattern.get("confidence", 0.5)
                description = pattern.get("description", "")
                
                pattern_scores[pattern_type]["confidence_sum"] += pattern_conf * conf
                pattern_scores[pattern_type]["count"] += 1
                if description:
                    pattern_scores[pattern_type]["descriptions"].append(description)
        
        if not pattern_scores:
            return {"type": "solid", "confidence": 0.5, "description": "Solid color"}
        
        # Find best pattern
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1]["confidence_sum"])
        pattern_type = best_pattern[0]
        data = best_pattern[1]
        
        # Average confidence
        avg_conf = data["confidence_sum"] / data["count"] if data["count"] > 0 else 0.5
        
        # Most common description
        description = data["descriptions"][0] if data["descriptions"] else f"{pattern_type.capitalize()} pattern"
        
        return {
            "type": pattern_type,
            "confidence": round(avg_conf, 3),
            "description": description,
            "framesDetected": data["count"]
        }
    
    def _aggregate_materials(
        self, 
        items: List[Dict], 
        confidences: List[float]
    ) -> Dict:
        """
        Aggregate material predictions using weighted voting.
        """
        material_scores = defaultdict(lambda: {"confidence_sum": 0, "count": 0, "textures": []})
        
        for item, conf in zip(items, confidences):
            attrs = item.get("attributes", {})
            material = attrs.get("material", {})
            
            if material:
                mat_type = material.get("material", "cotton")
                mat_conf = material.get("confidence", 0.5)
                texture = material.get("texture", "woven")
                
                material_scores[mat_type]["confidence_sum"] += mat_conf * conf
                material_scores[mat_type]["count"] += 1
                material_scores[mat_type]["textures"].append(texture)
        
        if not material_scores:
            return {"material": "unknown", "texture": "unknown", "confidence": 0.5}
        
        # Find best material
        best_material = max(material_scores.items(), key=lambda x: x[1]["confidence_sum"])
        mat_type = best_material[0]
        data = best_material[1]
        
        avg_conf = data["confidence_sum"] / data["count"] if data["count"] > 0 else 0.5
        
        # Most common texture
        texture_counts = Counter(data["textures"])
        most_common_texture = texture_counts.most_common(1)[0][0] if texture_counts else "woven"
        
        return {
            "material": mat_type,
            "texture": most_common_texture,
            "confidence": round(avg_conf, 3),
            "framesDetected": data["count"]
        }
    
    def _aggregate_features(
        self, 
        items: List[Dict], 
        confidences: List[float]
    ) -> Dict:
        """
        Aggregate detailed clothing features (closure, collar, sleeves, pockets, etc.)
        """
        # Collect all feature dictionaries
        all_features = []
        for item in items:
            attrs = item.get("attributes", {})
            features = attrs.get("detailedFeatures", {})
            if features:
                all_features.append(features)
        
        if not all_features:
            return {}
        
        aggregated = {}
        
        # Aggregate closure
        closure_types = []
        zipper_votes = []
        button_counts = []
        for f in all_features:
            closure = f.get("closure", {})
            if closure:
                if closure.get("type"):
                    closure_types.append(closure["type"])
                if "hasZipper" in closure:
                    zipper_votes.append(closure["hasZipper"])
                if closure.get("buttonCount"):
                    button_counts.append(closure["buttonCount"])
        
        if closure_types:
            most_common = Counter(closure_types).most_common(1)[0][0]
            aggregated["closure"] = {
                "type": most_common,
                "hasZipper": sum(zipper_votes) > len(zipper_votes) / 2 if zipper_votes else False,
                "buttonCount": int(np.median(button_counts)) if button_counts else 0
            }
        
        # Aggregate collar
        collar_types = []
        for f in all_features:
            collar = f.get("collar", {})
            if collar and collar.get("type"):
                collar_types.append(collar["type"])
        
        if collar_types:
            aggregated["collar"] = {
                "type": Counter(collar_types).most_common(1)[0][0]
            }
        
        # Aggregate sleeves
        sleeve_lengths = []
        sleeve_styles = []
        for f in all_features:
            sleeves = f.get("sleeves", {})
            if sleeves:
                if sleeves.get("length"):
                    sleeve_lengths.append(sleeves["length"])
                if sleeves.get("style"):
                    sleeve_styles.append(sleeves["style"])
        
        if sleeve_lengths or sleeve_styles:
            aggregated["sleeves"] = {}
            if sleeve_lengths:
                aggregated["sleeves"]["length"] = Counter(sleeve_lengths).most_common(1)[0][0]
            if sleeve_styles:
                aggregated["sleeves"]["style"] = Counter(sleeve_styles).most_common(1)[0][0]
        
        # Aggregate pockets
        pocket_counts = []
        pocket_types = []
        for f in all_features:
            pockets = f.get("pockets", {})
            if pockets:
                if pockets.get("count"):
                    pocket_counts.append(pockets["count"])
                if pockets.get("types"):
                    pocket_types.extend(pockets["types"])
        
        if pocket_counts:
            aggregated["pockets"] = {
                "count": int(np.median(pocket_counts)),
                "types": list(set(pocket_types))[:4]  # Unique types, max 4
            }
        
        # Aggregate fit
        fits = []
        for f in all_features:
            if f.get("fit"):
                fits.append(f["fit"])
        if fits:
            aggregated["fit"] = Counter(fits).most_common(1)[0][0]
        
        # Aggregate length
        lengths = []
        for f in all_features:
            if f.get("length"):
                lengths.append(f["length"])
        if lengths:
            aggregated["length"] = Counter(lengths).most_common(1)[0][0]
        
        # Aggregate special features (union of all detected)
        special = set()
        for f in all_features:
            special_list = f.get("specialFeatures", [])
            if special_list:
                special.update(special_list)
        if special:
            aggregated["specialFeatures"] = list(special)
        
        return aggregated
    
    def _aggregate_style_info(
        self, 
        items: List[Dict], 
        confidences: List[float]
    ) -> Dict:
        """
        Aggregate Fashion-CLIP style information.
        """
        style_scores = defaultdict(float)
        all_tags = []
        total_confidence = 0
        
        for item, conf in zip(items, confidences):
            attrs = item.get("attributes", {})
            style_info = attrs.get("styleInfo", {})
            
            if style_info:
                style = style_info.get("style")
                if style:
                    style_scores[style] += conf
                    total_confidence += conf
                
                tags = style_info.get("tags", [])
                all_tags.extend(tags)
        
        if not style_scores:
            return {}
        
        # Best style
        best_style = max(style_scores.items(), key=lambda x: x[1])
        
        # Aggregate tags (by frequency)
        tag_counts = Counter(all_tags)
        top_tags = [tag for tag, _ in tag_counts.most_common(5)]
        
        return {
            "style": best_style[0],
            "tags": top_tags,
            "confidence": round(best_style[1] / len(items), 3) if items else 0
        }

    
    def _union_strategy(self, category_detections: Dict[str, List]) -> List[Dict]:
        """
        Keep all detected items from all frames.
        Use highest confidence version of each.
        """
        final_items = []
        
        for category, detections in category_detections.items():
            best = max(detections, key=lambda d: d["confidence"])
            final_items.append(best["item"].copy())
        
        return final_items
    
    def _intersection_strategy(
        self, 
        category_detections: Dict[str, List],
        n_frames: int
    ) -> List[Dict]:
        """
        Only keep items that appear in ALL frames.
        Most conservative approach.
        """
        final_items = []
        
        for category, detections in category_detections.items():
            if len(detections) == n_frames:
                best = max(detections, key=lambda d: d["confidence"])
                item = best["item"].copy()
                item["confidence"] = min(0.99, item.get("confidence", 0.5) + self.confidence_boost)
                final_items.append(item)
        
        return final_items
    
    def get_temporal_consistency(
        self, 
        frame_results: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate temporal consistency scores for each category.
        Higher score = more consistent detection across frames.
        
        Returns:
            Dictionary mapping category -> consistency score (0-1)
        """
        if not frame_results:
            return {}
        
        n_frames = len(frame_results)
        category_counts = Counter()
        
        for result in frame_results:
            for item in result.get("items", []):
                category_counts[item.get("category", "unknown")] += 1
        
        return {
            cat: count / n_frames 
            for cat, count in category_counts.items()
        }


def analyze_video_frames(
    frames_base64: List[str], 
    segment_func,
    min_agreement: float = 0.6  # 🎯 4-Layer Stack: 60% agreement
) -> Dict[str, Any]:
    """
    Utility function to analyze multiple video frames.
    
    Args:
        frames_base64: List of base64-encoded video frames
        segment_func: Function to segment each frame (should accept base64 and return result dict)
        min_agreement: Minimum fraction of frames for an item to be included
    
    Returns:
        Combined multi-frame result
    """
    analyzer = MultiFrameAnalyzer(min_frame_agreement=min_agreement)
    
    # Process each frame
    frame_results = []
    for i, frame in enumerate(frames_base64):
        try:
            result = segment_func(frame)
            if result.get("success"):
                frame_results.append(result)
                logger.info(f"Frame {i}: {len(result.get('items', []))} items detected")
            else:
                logger.warning(f"Frame {i}: Processing failed")
        except Exception as e:
            logger.warning(f"Frame {i} error: {e}")
    
    # Combine results
    combined = analyzer.analyze_frames(frame_results, strategy="voting")
    
    return combined


def select_best_frames(
    frames_base64: List[str],
    n_best: int = 5,
    quality_func=None
) -> List[str]:
    """
    Select the best N frames from a video for analysis.
    
    Args:
        frames_base64: List of all frame base64 strings
        n_best: Number of best frames to select
        quality_func: Optional function to score frame quality
    
    Returns:
        List of best frame base64 strings
    """
    if len(frames_base64) <= n_best:
        return frames_base64
    
    if quality_func:
        # Score and sort frames
        scored = [(i, quality_func(f)) for i, f in enumerate(frames_base64)]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_indices = [s[0] for s in scored[:n_best]]
    else:
        # Evenly sample frames
        step = len(frames_base64) / n_best
        best_indices = [int(i * step) for i in range(n_best)]
    
    return [frames_base64[i] for i in sorted(best_indices)]
