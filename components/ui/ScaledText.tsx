import React from 'react';
import { StyleSheet, Text, TextProps, TextStyle } from 'react-native';
import { useTranslation } from 'react-i18next';

interface ScaledTextProps extends TextProps {
  tKey?: string;
  tOptions?: Record<string, unknown>;
  enText?: string;
  minScale?: number;
}

function flattenTextStyle(style: TextProps['style']): TextStyle {
  return (StyleSheet.flatten(style) as TextStyle) || {};
}

function extractBaseFontSize(style: TextProps['style']): number {
  return flattenTextStyle(style).fontSize || 14;
}

export const ScaledText: React.FC<ScaledTextProps> = ({
  tKey,
  tOptions,
  style,
  children,
  enText: explicitEnText,
  minScale = 0.7,
  ...props
}) => {
  const { t, i18n: i18nInstance } = useTranslation();

  const displayText = tKey ? t(tKey, tOptions) : undefined;

  const currentText =
    typeof displayText === 'string'
      ? displayText
      : typeof children === 'string'
        ? children
        : '';

  let enText = explicitEnText;
  if (!enText && tKey) {
    try {
      const resolved = i18nInstance.t(tKey, { lng: 'en', ...tOptions });
      enText = typeof resolved === 'string' ? resolved : '';
    } catch {
      enText = currentText;
    }
  }

  if (!enText) enText = currentText;

  const baseFontSize = extractBaseFontSize(style);

  let scale = 1;
  if (currentText.length > enText.length && enText.length > 0) {
    scale = Math.max(minScale, enText.length / currentText.length);
  }

  const flatStyle = flattenTextStyle(style);

  const scaledStyle: TextStyle = {
    ...flatStyle,
    fontSize: baseFontSize * scale,
  };

  return (
    <Text style={scaledStyle} {...props}>
      {displayText ?? children}
    </Text>
  );
};

export default ScaledText;
