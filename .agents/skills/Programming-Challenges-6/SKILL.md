---
name: programming-challenges-6-conventions
description: Development conventions and patterns for Programming-Challenges-6. Python project with mixed commits.
---

# Programming Challenges 6 Conventions

> Generated from [saint2706/Programming-Challenges-6](https://github.com/saint2706/Programming-Challenges-6) on 2026-10-16

## Overview

This skill teaches Claude the development patterns and conventions used in Programming-Challenges-6.

## Tech Stack

- **Primary Language**: Python
- **Architecture**: type-based module organization
- **Test Location**: mixed
- **Test Framework**: vitest

## When to Use This Skill

Activate this skill when:
- Making changes to this repository
- Adding new features following established patterns
- Writing tests that match project conventions
- Creating commits with proper message format

## Commit Conventions

Follow these commit message conventions based on 200 analyzed commits.

### Commit Style: Mixed Style

### Prefixes Used

- `chore`
- `feat`
- `fix`

### Message Guidelines

- Average message length: ~65 characters
- Keep first line concise and descriptive
- Use imperative mood ("Add feature" not "Added feature")


*Commit message example*

```text
test(web): let the configurator's camera test pass at a software frame rate, and give CI runs more time per test
```

*Commit message example*

```text
feat(web): add 3D Product Configurator with a head-coupled window preview (#18)
```

*Commit message example*

```text
fix(ml): write results the way dprint formats them
```

*Commit message example*

```text
ci: install web challenges with pnpm when a pnpm-lock.yaml is present
```

*Commit message example*

```text
docs: add Unlicense and security policy (#115)
```

*Commit message example*

```text
[ImgBot] Optimize images (#117)
```

*Commit message example*

```text
feat(web): add Browser-Based Code Playground with Live Preview (#17)
```

*Commit message example*

```text
feat(web): add Real-Time Collaborative Whiteboard (#16)
```

## Architecture

### Project Structure: Single Package

This project uses **type-based** module organization.

### Configuration Files

- `.github/ci-tools/package.json`
- `.github/workflows/ci.yml`
- `.github/workflows/codeql.yml`
- `.github/workflows/dependabot-automerge.yml`
- `.github/workflows/dependency-review.yml`
- `.github/workflows/link-check.yml`
- `.github/workflows/lint-and-format.yml`
- `.github/workflows/scorecard.yml`
- `.github/workflows/workflow-lint.yml`
- `challenges/Web Development Showcase/3D Product Configurator/package.json`
- `challenges/Web Development Showcase/3D Product Configurator/tsconfig.json`
- `challenges/Web Development Showcase/3D Product Configurator/vite.config.ts`
- `challenges/Web Development Showcase/Animated SVG Icon Library with Hover States/package.json`
- `challenges/Web Development Showcase/Before-After Image Comparison Slider Component/package.json`
- `challenges/Web Development Showcase/Before-After Image Comparison Slider Component/tsconfig.json`
- `challenges/Web Development Showcase/Browser-Based Code Playground with Live Preview/package.json`
- `challenges/Web Development Showcase/Browser-Based Code Playground with Live Preview/tsconfig.json`
- `challenges/Web Development Showcase/Browser-Based Code Playground with Live Preview/vite.config.ts`
- `challenges/Web Development Showcase/Client-Side Data Table with Sort-Filter-CSV Export/package.json`
- `challenges/Web Development Showcase/Command Palette (Cmd+K) UI for a Static Site/package.json`
- `challenges/Web Development Showcase/Custom Cursor & Micro-Interaction Playground/package.json`
- `challenges/Web Development Showcase/Drag-to-Reorder Photo Album with Masonry Layout/package.json`
- `challenges/Web Development Showcase/Interactive Resume Builder with Live PDF Export/package.json`
- `challenges/Web Development Showcase/Interactive Resume Timeline with Scroll Animations/package.json`
- `challenges/Web Development Showcase/Multi-Step Wizard Form with Progress Bar/package.json`
- `challenges/Web Development Showcase/Real-Time Collaborative Whiteboard/package.json`
- `challenges/Web Development Showcase/Real-Time Collaborative Whiteboard/tsconfig.json`
- `challenges/Web Development Showcase/Real-Time Collaborative Whiteboard/vite.config.ts`
- `challenges/Web Development Showcase/Real-Time Collaborative Whiteboard/vitest.config.ts`
- `challenges/Web Development Showcase/Responsive Email Template Builder with Live Preview/package.json`
- `challenges/Web Development Showcase/Scroll-Linked Storytelling Page/package.json`
- `challenges/Web Development Showcase/Single-Page Event Invitation with RSVP/package.json`
- `challenges/Web Development Showcase/Theme Customizer Playground/package.json`
- `challenges/Web Development Showcase/Typing-Effect Hero Banner Generator/package.json`
- `challenges/Web Development Showcase/Typing-Effect Hero Banner Generator/tsconfig.json`

### Guidelines

- Group code by type (components, services, utils)
- Keep related functionality in the same type folder
- Avoid circular dependencies between type folders

## Code Style

### Language: Python

### Naming Conventions

| Element | Convention |
|---------|------------|
| Files | camelCase |
| Functions | camelCase |
| Classes | PascalCase |
| Constants | SCREAMING_SNAKE_CASE |

### Import Style: Relative Imports

### Export Style: Named Exports


*Preferred import style*

```typescript
// Use relative imports
import { Button } from '../components/Button'
import { useAuth } from './hooks/useAuth'
```

*Preferred export style*

```typescript
// Use named exports
export function calculateTotal() { ... }
export const TAX_RATE = 0.1
export interface Order { ... }
```

## Testing

### Test Framework: vitest

### File Pattern: `*.test.js`

### Test Types

- **Unit tests**: Test individual functions and components in isolation
- **E2e tests**: Test complete user flows through the application

### Coverage

This project has coverage reporting configured. Aim for 80%+ coverage.


*Test file structure*

```typescript
import { describe, it, expect } from 'vitest'

describe('MyFunction', () => {
  it('should return expected result', () => {
    const result = myFunction(input)
    expect(result).toBe(expected)
  })
})
```

## Error Handling

### Error Handling Style: Try-Catch Blocks


*Standard error handling pattern*

```typescript
try {
  const result = await riskyOperation()
  return result
} catch (error) {
  console.error('Operation failed:', error)
  throw new Error('User-friendly message')
}
```

## Common Workflows

These workflows were detected from analyzing commit patterns.

### Feature Development

Standard feature implementation workflow

**Frequency**: ~11 times per month

**Steps**:
1. Add feature implementation
2. Add tests for feature
3. Update documentation

**Files typically involved**:
- `challenges/web development showcase/before-after image comparison slider component/*`
- `challenges/web development showcase/before-after image comparison slider component/demo/img/*`
- `challenges/web development showcase/before-after image comparison slider component/demo/*`
- `**/*.test.*`
- `**/api/**`

**Example commit sequence**:
```
feat: add Automatic Invoice Generator & PDF Emailer challenge
feat: add PDF Merge-Split-Watermark Toolkit challenge
fix: restore Practical Software 22/23 tracker rows, mark Algorithmic 22/23 implemented
```


## Best Practices

Based on analysis of the codebase, follow these practices:

### Do

- Write tests using vitest
- Follow *.test.js naming pattern
- Use camelCase for file names
- Prefer named exports

### Don't

- Don't skip tests for new features
- Don't deviate from established patterns without discussion

---

*This skill was auto-generated by [ECC Tools](https://ecc.tools). Review and customize as needed for your team.*
