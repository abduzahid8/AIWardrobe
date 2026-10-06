// Re-exports for the outfit-generator feature
// NOTE: useOutfitGeneration / useAnchorOutfitGeneration / mixedOutfitService
// were deleted (2026-10-01) — zero call sites; AIOutfitmaker and Home use
// their own live paths. Do not re-add parallel generators here.
export { useItemSelection, getMacroCategory } from './hooks/useItemSelection';
export { useWardrobeLoader } from './hooks/useWardrobeLoader';
export { default as ModeToggle } from './components/ModeToggle';
export { default as VibeCard } from './components/VibeCard';
export { default as PromptInput } from './components/PromptInput';
export { default as CategorySection } from './components/CategorySection';
export { default as OutfitFAB } from './components/OutfitFAB';
export { default as OutfitCollageDisplay } from './components/OutfitCollageDisplay';
export { default as OutfitResultsView } from './components/OutfitResultsView';
export { default as GenerateOutfitButton } from './components/GenerateOutfitButton';
export { default as GenerationProgressOverlay } from './components/GenerationProgressOverlay';
export { default as OutfitResultSheet } from './components/OutfitResultSheet';
export * from './types';
