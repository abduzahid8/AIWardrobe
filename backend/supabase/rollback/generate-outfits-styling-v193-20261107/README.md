# Rollback v193 → verified v192 source

`v192-source.tar.gz` содержит `index.ts`, скачанный из ACTIVE generate-outfits
v192 и побайтово сверенный с bundle commit `52a97e3f3595` до правок.
Source SHA-256: `6e126cc5944d41f5666cfcec9dca36bc57b62877acb87266408a886e9a603d39`.
Hashes архива, candidate bundle и всех его inputs находятся в `manifest.json`.
Снимок v191 в `generate-outfits-selection-quality-20261107` сохранён.

Для отката из этого worktree:

```sh
SUPABASE_ACCESS_TOKEN_FILE=/secure/path/to/token node scripts/validation/styling-v193-release.cjs rollback
```

Скрипт откажется перезаписывать более новую deployment version/fingerprint.
Он проверяет candidate original source, archive hash и извлечённый `index.ts`,
развёртывает только generate-outfits с прежним `verify_jwt=false`, затем
скачивает оригинал и проверяет SHA-256, ACTIVE, auth config, остальные функции
и CORS/auth smoke. Supabase присвоит новую монотонную version, хотя исходник
восстановится от v192. Схема и пользовательские данные не изменяются.

Текущая ветка содержит изменения для приложения; rollback здесь относится
только к Edge Function. Результат deployment/verification записан в
`docs/reports/styling-quality-v193-20261107/`.
