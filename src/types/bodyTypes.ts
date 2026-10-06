/**
 * Body-type taxonomy shared by BodyProfile, CreateAvatar, and avatarStore.
 *
 * Previously lived in `features/try-on/utils/mannequin3D.ts` (deleted with
 * the Virtual Try-On feature). The 3D mannequin renderer is gone; the
 * taxonomy stays because body-fit onboarding depends on it.
 */
export const BODY_TYPES = [
  { id: "ectomorph", label: "Slim", icon: "body-outline", desc: "Lean build" },
  { id: "average", label: "Average", icon: "person-outline", desc: "Balanced proportions" },
  { id: "mesomorph", label: "Muscular", icon: "barbell-outline", desc: "Muscular frame" },
  { id: "endomorph", label: "Fat", icon: "pizza-outline", desc: "Heavy set, wider build" },
] as const;

export type BodyTypeId = typeof BODY_TYPES[number]["id"];
