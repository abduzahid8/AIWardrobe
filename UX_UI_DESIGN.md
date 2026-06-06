# UX/UI Design Reference

> **⚠️ MANDATORY: Before making any UX/UI design changes, you MUST read this file first.**

## Design System

This project uses [**Open Design**](open-design/) as its design system source of truth.

- **Repo:** https://github.com/nexu-io/open-design.git
- **Location in project:** [`open-design/`](open-design/)
- **Design systems catalog:** [`open-design/design-systems/`](open-design/design-systems/)
- **Design templates:** [`open-design/design-templates/`](open-design/design-templates/)
- **Docs:** [`open-design/docs/`](open-design/docs/)

## Design Workflow

1. **Read this file** — always start here before any UI/UX work.
2. **Check the design systems** in `open-design/design-systems/` to find a matching brand/pattern.
3. **Browse design templates** in `open-design/design-templates/` for reference implementations.
4. **Consult the docs** in `open-design/docs/` for architecture and guidelines.
5. **Make your changes** applying the design system tokens, components, and patterns found above.
6. **Update this file** if your change introduces a new design convention or pattern.

## Updating Open Design

To pull the latest design system updates:

```bash
git submodule update --remote open-design
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `open-design/design-systems/` | Brand-grade design systems (colors, typography, components) |
| `open-design/design-templates/` | Ready-to-use UI templates and pages |
| `open-design/docs/` | Architecture, ADRs, deployment, and guidelines |
| `open-design/craft/` | Design craft tools and workflows |
| `open-design/skills/` | Reusable design skills for agents |

## Questions?

If you're unsure about a design decision, check the existing patterns in `open-design/` first, then consult the team.
