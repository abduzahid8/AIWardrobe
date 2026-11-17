import React, { useEffect, useState } from 'react';
import {
  Alert,
  Keyboard,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  Share,
  StyleSheet,
  TextInput,
  View,
} from 'react-native';
import Constants from 'expo-constants';
import { useTranslation } from 'react-i18next';
import { ScaledText } from './ui/ScaledText';
import { authEvents } from '../store/authEvents';
import {
  BETA_FEATURES,
  CORRECTION_FIELDS,
  ISSUE_TYPES,
  makeBetaFeedback,
  revertBetaCorrection,
  type BetaFeature,
  type BetaFeedback,
  type CorrectionField,
  type IssueType,
  type ResultDiagnostics,
} from '../src/services/launch/validationLoop';

/** Optional, session-local evidence: the user reviews the exact export before sharing. */
export function BetaFeedbackButton({
  feature = 'general',
  diagnostics = {},
}: {
  feature?: BetaFeature;
  diagnostics?: ResultDiagnostics;
}) {
  const { t } = useTranslation();
  const [visible, setVisible] = useState(false),
    [selectedFeature, setFeature] = useState(feature);
  const [helpful, setHelpful] = useState<boolean | null>(null),
    [issueType, setIssue] = useState<IssueType>('other');
  const [comment, setComment] = useState(''),
    [field, setField] = useState<CorrectionField | null>(null),
    [before, setBefore] = useState(''),
    [after, setAfter] = useState('');
  const [record, setRecord] = useState<BetaFeedback | null>(null);
  const reset = () => {
    Keyboard.dismiss();
    setVisible(false);
    setRecord(null);
    setComment('');
    setBefore('');
    setAfter('');
    setField(null);
    setHelpful(null);
    setFeature(feature);
    setIssue('other');
  };
  useEffect(() => {
    const logout = authEvents.onLogout(reset),
      login = authEvents.onLogin(reset);
    return () => {
      logout();
      login();
    };
    // Reset private drafts at every identity boundary.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [feature]);
  const button = (label: string, action: () => void, selected = false, disabled = false) => (
    <Pressable
      key={label}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ selected, disabled }}
      disabled={disabled}
      onPress={action}
      style={[styles.button, selected && styles.selected, disabled && { opacity: 0.4 }]}
    >
      <ScaledText style={styles.text}>{label}</ScaledText>
    </Pressable>
  );
  return (
    <>
      {button(t('beta.report'), () => setVisible(true))}
      <Modal visible={visible} animationType="slide" onRequestClose={reset}>
        <KeyboardAvoidingView
          style={styles.viewport}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        >
          <ScrollView
            contentContainerStyle={styles.sheet}
            keyboardShouldPersistTaps="handled"
            keyboardDismissMode="on-drag"
          >
            <ScaledText style={styles.title}>{t('beta.report')}</ScaledText>
            <ScaledText style={styles.text}>{t('beta.privacy')}</ScaledText>
            {record ? (
              <>
                <ScaledText selectable style={styles.text}>
                  {JSON.stringify(record, null, 2)}
                </ScaledText>
                {record.correction &&
                  !record.correction.revertedAt &&
                  button(t('beta.undo'), () => setRecord(revertBetaCorrection(record)))}
                {button(t('beta.edit'), () => setRecord(null))}
                {button(t('beta.share'), () => {
                  void Share.share({ message: JSON.stringify(record, null, 2) }).catch(() =>
                    Alert.alert(t('beta.shareFailed'))
                  );
                })}
              </>
            ) : (
              <>
                <ScaledText style={styles.text}>{t('beta.feature')}</ScaledText>
                <View style={styles.options}>
                  {BETA_FEATURES.map((value) =>
                    button(
                      t(`beta.features.${value}`),
                      () => setFeature(value),
                      value === selectedFeature
                    )
                  )}
                </View>
                <ScaledText style={styles.text}>{t('beta.helpful')}</ScaledText>
                <View style={styles.options}>
                  {button(t('beta.yes'), () => setHelpful(true), helpful === true)}
                  {button(t('beta.no'), () => setHelpful(false), helpful === false)}
                </View>
                <ScaledText style={styles.text}>{t('beta.issue')}</ScaledText>
                <View style={styles.options}>
                  {ISSUE_TYPES.map((value) =>
                    button(t(`beta.issues.${value}`), () => setIssue(value), value === issueType)
                  )}
                </View>
                <TextInput
                  accessibilityLabel={t('beta.comment')}
                  placeholder={t('beta.comment')}
                  placeholderTextColor="#555"
                  value={comment}
                  onChangeText={setComment}
                  maxLength={500}
                  multiline
                  style={styles.input}
                />
                <ScaledText style={styles.text}>{t('beta.correction')}</ScaledText>
                <View style={styles.options}>
                  {CORRECTION_FIELDS.map((value) =>
                    button(
                      t(`beta.fields.${value}`),
                      () => setField(field === value ? null : value),
                      field === value
                    )
                  )}
                </View>
                {field && (
                  <>
                    <TextInput
                      accessibilityLabel={t('beta.before')}
                      placeholder={t('beta.before')}
                      placeholderTextColor="#555"
                      value={before}
                      onChangeText={setBefore}
                      maxLength={100}
                      autoCorrect={false}
                      returnKeyType="done"
                      onSubmitEditing={Keyboard.dismiss}
                      style={styles.input}
                    />
                    <TextInput
                      accessibilityLabel={t('beta.after')}
                      placeholder={t('beta.after')}
                      placeholderTextColor="#555"
                      value={after}
                      onChangeText={setAfter}
                      maxLength={100}
                      autoCorrect={false}
                      returnKeyType="done"
                      onSubmitEditing={Keyboard.dismiss}
                      style={styles.input}
                    />
                  </>
                )}
                {button(
                  t('beta.review'),
                  () => {
                    Keyboard.dismiss();
                    setRecord(
                      makeBetaFeedback({
                        appVersion: Constants.expoConfig?.version || 'unknown',
                        feature: selectedFeature,
                        helpful: helpful!,
                        issueType,
                        comment,
                        diagnostics,
                        ...(field ? { correction: { field, before, after } } : {}),
                      })
                    );
                  },
                  false,
                  helpful === null || (!!field && !after.trim())
                )}
              </>
            )}
            {button(t('beta.close'), reset)}
          </ScrollView>
        </KeyboardAvoidingView>
      </Modal>
    </>
  );
}
const styles = StyleSheet.create({
  viewport: { flex: 1, backgroundColor: '#fff' },
  sheet: { padding: 24, paddingTop: 64, paddingBottom: 50, gap: 12, backgroundColor: '#fff' },
  title: { fontSize: 22, fontWeight: '700', color: '#111' },
  text: { color: '#111', fontSize: 15 },
  options: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  button: {
    minHeight: 44,
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#edf0f5',
    justifyContent: 'center',
  },
  selected: { borderWidth: 2, borderColor: '#1949a0' },
  input: {
    minHeight: 48,
    padding: 12,
    borderWidth: 1,
    borderColor: '#777',
    borderRadius: 10,
    color: '#111',
    fontSize: 16,
  },
});
