/**
 * BentoGrid Component
 * Adaptive modular grid layout for GenUI-ready interfaces
 * Inspired by Apple's widget system and Japanese bento boxes
 */

import React, { useMemo, useCallback } from 'react';
import {
    View,
    StyleSheet,
    useWindowDimensions,
    StyleProp,
    ViewStyle,
} from 'react-native';
import Animated, {
    FadeIn,
    FadeInDown,
    FadeInUp,
    Layout,
} from 'react-native-reanimated';
import { LiquidGlass2026Theme } from '../../constants/LiquidGlass2026Theme';
import { useReducedMotion } from '../../hooks/useAccessibility';

// ============================================
// TYPES
// ============================================

export interface BentoGridProps {
    /** Number of columns (auto-calculated if not provided) */
    columns?: 2 | 3 | 4;
    /** Gap between items */
    gap?: number;
    /** Container padding */
    padding?: number;
    /** Children items */
    children: React.ReactNode;
    /** Additional container styles */
    style?: StyleProp<ViewStyle>;
    /** Enable staggered entrance animation */
    animated?: boolean;
    /** Animation delay between items (ms) */
    staggerDelay?: number;
}

export interface BentoItemProps {
    /** Column span (1-4) */
    colSpan?: 1 | 2 | 3 | 4;
    /** Row span (1-2) */
    rowSpan?: 1 | 2;
    /** Aspect ratio preset */
    aspectRatio?: 'square' | 'wide' | 'tall' | 'auto';
    /** Children content */
    children: React.ReactNode;
    /** Additional item styles */
    style?: StyleProp<ViewStyle>;
    /** Item index for animation delay */
    index?: number;
    /** Enable entrance animation */
    animated?: boolean;
}

// ============================================
// BENTO GRID CONTAINER
// ============================================

export const BentoGrid: React.FC<BentoGridProps> = ({
    columns: customColumns,
    gap = LiquidGlass2026Theme.spacing.bento.gap,
    padding = LiquidGlass2026Theme.spacing.screenPadding,
    children,
    style,
    animated = true,
    staggerDelay = 50,
}) => {
    const { width: screenWidth } = useWindowDimensions();
    const reducedMotion = useReducedMotion();

    // Calculate columns if not provided
    const columns = useMemo(() => {
        if (customColumns) return customColumns;
        return LiquidGlass2026Theme.bento.getColumns(screenWidth);
    }, [customColumns, screenWidth]);

    // Calculate cell width
    const cellWidth = useMemo(() => {
        const availableWidth = screenWidth - (padding * 2);
        const totalGaps = (columns - 1) * gap;
        return (availableWidth - totalGaps) / columns;
    }, [screenWidth, padding, columns, gap]);

    // Clone children with grid context
    const gridContext = useMemo(() => ({
        columns,
        cellWidth,
        gap,
        animated: animated && !reducedMotion,
        staggerDelay,
    }), [columns, cellWidth, gap, animated, reducedMotion, staggerDelay]);

    // Enhance children with grid props
    const enhancedChildren = useMemo(() => {
        return React.Children.map(children, (child, index) => {
            if (!React.isValidElement(child)) return child;

            return React.cloneElement(child as React.ReactElement<BentoItemProps>, {
                index,
                animated: gridContext.animated,
                // Pass grid context through props
                ...(child.props as object),
            });
        });
    }, [children, gridContext]);

    return (
        <View
            style={[
                styles.grid,
                {
                    paddingHorizontal: padding,
                    gap,
                },
                style,
            ]}
        >
            {enhancedChildren}
        </View>
    );
};

// ============================================
// BENTO ITEM
// ============================================

export const BentoItem: React.FC<BentoItemProps> = ({
    colSpan = 1,
    rowSpan = 1,
    aspectRatio = 'auto',
    children,
    style,
    index = 0,
    animated = true,
}) => {
    const { width: screenWidth } = useWindowDimensions();
    const reducedMotion = useReducedMotion();

    // Calculate item dimensions
    const dimensions = useMemo(() => {
        const columns = LiquidGlass2026Theme.bento.getColumns(screenWidth);
        const padding = LiquidGlass2026Theme.spacing.screenPadding;
        const gap = LiquidGlass2026Theme.spacing.bento.gap;

        const availableWidth = screenWidth - (padding * 2);
        const totalGaps = (columns - 1) * gap;
        const cellWidth = (availableWidth - totalGaps) / columns;

        // Calculate width based on span
        const itemWidth = (cellWidth * colSpan) + ((colSpan - 1) * gap);

        // Calculate height based on aspect ratio
        let itemHeight: number | undefined;

        switch (aspectRatio) {
            case 'square':
                itemHeight = cellWidth * rowSpan + ((rowSpan - 1) * gap);
                break;
            case 'wide':
                itemHeight = cellWidth * 0.6 * rowSpan;
                break;
            case 'tall':
                itemHeight = cellWidth * 1.5 * rowSpan;
                break;
            case 'auto':
            default:
                itemHeight = undefined; // Let content determine height
        }

        return {
            width: itemWidth,
            height: itemHeight,
            minHeight: LiquidGlass2026Theme.spacing.bento.itemMinHeight,
        };
    }, [screenWidth, colSpan, rowSpan, aspectRatio]);

    // Animation configuration
    const entering = useMemo(() => {
        if (!animated || reducedMotion) return undefined;

        return FadeInDown
            .delay(index * 50)
            .duration(LiquidGlass2026Theme.animation.duration.normal)
            .springify()
            .damping(LiquidGlass2026Theme.animation.spring.smooth.damping)
            .stiffness(LiquidGlass2026Theme.animation.spring.smooth.stiffness);
    }, [animated, reducedMotion, index]);

    const Container = animated && !reducedMotion ? Animated.View : View;

    return (
        <Container
            style={[
                styles.item,
                {
                    width: dimensions.width,
                    height: dimensions.height,
                    minHeight: dimensions.minHeight,
                },
                style,
            ]}
            entering={entering}
            layout={Layout.springify()}
        >
            {children}
        </Container>
    );
};

// ============================================
// PRESET BENTO ITEMS
// ============================================

interface BentoPresetProps {
    children: React.ReactNode;
    style?: StyleProp<ViewStyle>;
    index?: number;
    animated?: boolean;
}

/** Small square item (1x1) */
export const BentoSmall: React.FC<BentoPresetProps> = (props) => (
    <BentoItem colSpan={1} rowSpan={1} aspectRatio="square" {...props} />
);

/** Medium wide item (2x1) */
export const BentoMedium: React.FC<BentoPresetProps> = (props) => (
    <BentoItem colSpan={2} rowSpan={1} aspectRatio="wide" {...props} />
);

/** Large item (2x2) */
export const BentoLarge: React.FC<BentoPresetProps> = (props) => (
    <BentoItem colSpan={2} rowSpan={2} aspectRatio="square" {...props} />
);

/** Tall item (1x2) */
export const BentoTall: React.FC<BentoPresetProps> = (props) => (
    <BentoItem colSpan={1} rowSpan={2} aspectRatio="tall" {...props} />
);

/** Full width item */
export const BentoWide: React.FC<BentoPresetProps> = (props) => (
    <BentoItem colSpan={3} rowSpan={1} aspectRatio="wide" {...props} />
);

/** Hero item (full width, 2 rows) */
export const BentoHero: React.FC<BentoPresetProps> = (props) => (
    <BentoItem colSpan={4} rowSpan={2} aspectRatio="wide" {...props} />
);

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    grid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        alignItems: 'flex-start',
    },
    item: {
        overflow: 'hidden',
    },
});

// ============================================
// EXPORTS
// ============================================

export default BentoGrid;
