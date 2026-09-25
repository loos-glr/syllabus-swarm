---
name: GLR Media Creative
colors:
  surface: '#faf9fd'
  surface-dim: '#dbd9dd'
  surface-bright: '#faf9fd'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f5f3f7'
  surface-container: '#efedf1'
  surface-container-high: '#e9e7ec'
  surface-container-highest: '#e3e2e6'
  on-surface: '#1b1b1f'
  on-surface-variant: '#424936'
  inverse-surface: '#2f3034'
  inverse-on-surface: '#f2f0f4'
  outline: '#727a64'
  outline-variant: '#c2cab1'
  surface-tint: '#416900'
  primary: '#416900'
  on-primary: '#ffffff'
  primary-container: '#76b800'
  on-primary-container: '#284300'
  inverse-primary: '#95da32'
  secondary: '#5f5e5e'
  on-secondary: '#ffffff'
  secondary-container: '#e5e2e1'
  on-secondary-container: '#656464'
  tertiary: '#2540ff'
  on-tertiary: '#ffffff'
  tertiary-container: '#939fff'
  on-tertiary-container: '#001dbb'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#aff74e'
  primary-fixed-dim: '#95da32'
  on-primary-fixed: '#102000'
  on-primary-fixed-variant: '#304f00'
  secondary-fixed: '#e5e2e1'
  secondary-fixed-dim: '#c8c6c5'
  on-secondary-fixed: '#1c1b1b'
  on-secondary-fixed-variant: '#474646'
  tertiary-fixed: '#dfe0ff'
  tertiary-fixed-dim: '#bcc2ff'
  on-tertiary-fixed: '#000a63'
  on-tertiary-fixed-variant: '#0023d9'
  background: '#faf9fd'
  on-background: '#1b1b1f'
  surface-variant: '#e3e2e6'
  electric-lime: '#A6E22E'
  pure-black: '#000000'
  pure-white: '#FFFFFF'
  surface-subtle: '#F4F4F6'
  border-structural: '#E5E7EB'
  accent-ultramarine: '#002BFF'
typography:
  display-xl:
    fontFamily: Space Grotesk
    fontSize: 64px
    fontWeight: '700'
    lineHeight: 68px
    letterSpacing: -0.04em
  display-xl-mobile:
    fontFamily: Space Grotesk
    fontSize: 40px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.03em
  headline-lg:
    fontFamily: Space Grotesk
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 52px
    letterSpacing: -0.03em
  headline-lg-mobile:
    fontFamily: Space Grotesk
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Space Grotesk
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 38px
    letterSpacing: -0.02em
  headline-sm:
    fontFamily: Space Grotesk
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 30px
    letterSpacing: -0.01em
  title-md:
    fontFamily: Space Grotesk
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 26px
    letterSpacing: 0em
  body-lg:
    fontFamily: Hanken Grotesk
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Hanken Grotesk
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Hanken Grotesk
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-lg:
    fontFamily: Space Grotesk
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 18px
    letterSpacing: 0.04em
  label-md:
    fontFamily: Space Grotesk
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.06em
  label-code:
    fontFamily: Space Grotesk
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 14px
    letterSpacing: 0.1em
spacing:
  gutter: 1.5rem
  gutter-mobile: 1rem
  margin: 3rem
  margin-mobile: 1.25rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

This design system establishes a high-energy, authoritative, and contemporary media-tech identity tailored for Grafisch Lyceum Rotterdam. It reflects the dynamic tension between industrial precision and rebellious creative youth culture. Targeted at prospective creators, current design/tech students, faculty, and industry partners, the visual language balances professional academic credibility with the raw punch of cutting-edge creative media.

The design movement is **High-Contrast Brutalist Modernism** fused with **Clean Editorial Tech**. It pairs stark monochromatic scaffolding (pure inky blacks and crisp paper whites) with an unmistakable electric lime/chartreuse signal green and precise cobalt blue accents. The visual tone is bold, deliberate, and unapologetic—featuring heavy structural lines, hyper-legible geometric typography, tight modular density, and razor-sharp content containment.

## Colors

The palette is engineered around high optical vibration and absolute contrast. 

- **Primary (`#76B800` / `electric-lime` `#A6E22E`):** The signature GLR acid/chartreuse green. Used for critical focal points, primary interactive highlights, badges, cursor states, and key navigational milestones. It signifies energy, tech innovation, and fresh creative output.
- **Secondary (`#111111` / `pure-black` `#000000`):** Heavy carbon black provides structural framing, typography grounding, high-impact inverted section blocks, and robust border rules.
- **Tertiary (`#002BFF`):** Ultramarine cobalt blue, reserved for hyper-specific interactions, student showcases, tech-track callouts, and secondary informational accents.
- **Neutrals (`pure-white` `#FFFFFF`, `surface-subtle` `#F4F4F6`, and `border-structural` `#E5E7EB`):** Form the clean, clinical canvas allowing visual student work and media assets to breathe without distraction.

Color rules:
- Never wash out primary green; pair it directly against solid black `#000000` or stark white `#FFFFFF` for guaranteed WCAG AA+ compliance.
- Interactive states prioritize high-contrast inversions (e.g., lime filling to ink black with white text, or pure black filling with lime text).

## Typography

The typographic hierarchy channels the spirit of GLR's historic affinity for structured European grotesques (Nimbus Sans, Sequel 100 Black) while maintaining maximum screen readability and expressive flair.

- **Headlines & Labels (`Space Grotesk`):** Chosen for its unapologetic geometric mechanics, angular cuts, and editorial edge. Used across hero displays, section titles, metadata badges, and navigation items. Display scales use tight negative tracking (`-0.03em` to `-0.04em`) to build visual authority. Labels frequently utilize uppercase styling with generous tracking (`0.04em` to `0.1em`).
- **Body Text (`Hanken Grotesk`):** Chosen for its exceptional technical clarity, neutral proportions, and high readability in long-form editorial contexts and course curriculum overviews.

Font hierarchy relies heavily on strong scale contrasts—pairing massive display titles with disciplined, highly legible small-caps metadata and clean body runs.

## Layout & Spacing

The layout is built upon an architectural 12-column fluid grid system with structural grid borders. Layout blocks embrace hard division lines reminiscent of print production sheets and media layouts.

- **Desktop (>= 1024px):** 12-column grid, `margin` of `3rem` (48px), and column `gutter` of `1.5rem` (24px).
- **Tablet (768px - 1023px):** 8-column grid, `margin` of `2rem` (32px), `gutter` of `1.25rem` (20px).
- **Mobile (< 768px):** 4-column grid, `margin-mobile` of `1.25rem` (20px), `gutter-mobile` of `1rem` (16px).

Elements snap flush against structural guidelines. Vertical rhythm follows multiples of 8px (`space-sm` at 8px, `space-md` at 16px, `space-lg` at 24px, and `space-xl` at 40px), creating disciplined alignment between imagery, typographic columns, and interactive modules.

## Elevation & Depth

This design system deliberately rejects heavy skeuomorphism and diffuse, fuzzy ambient drop shadows in favor of **Crisp Architectural Depth, Surface Layering, and Hard Offset Shadows**.

1. **Structural Borders:** Elevation level 0 to 1 relies on 1px solid high-contrast borders (`#111111` or `#E5E7EB`) that compartmentalize information without blur.
2. **Hard Kinetic Shadows (Brutalist Depth):** Interactive elevated surfaces (such as cards on hover, active badges, or popovers) employ hard-edge, zero-blur displacement shadows:
   - Default card hover: `box-shadow: 4px 4px 0px 0px #111111;`
   - Dark mode or active highlight: `box-shadow: 4px 4px 0px 0px #76B800;`
3. **Tonal Contrast:** Depth is established by overlaying solid containers—e.g., placing a pure white or pure black media card onto an off-white `#F4F4F6` canvas, anchored with sharp border outlines.

## Shapes

The shape system adopts a **Sharp (`0`)** aesthetic philosophy. Corners are kept at `0px` radius across all standard UI surfaces, including buttons, cards, tags, text fields, and image viewports.

This sharp, non-rounded treatment reinforces the technical heritage of graphic arts, screen printing, editorial layout grids, and campus architectural signage. When interactive feedback occurs, transformation happens through solid line shifts, color inversions, and hard-edge shadow offsets rather than curvature morphing.

## Components

### Buttons
- **Primary:** Background `#76B800` (or `#A6E22E`), text `#000000`, 0px border-radius, 1px solid `#000000`. Font `label-lg` uppercase. Hover shifts to `#000000` background with `#A6E22E` text and a `3px 3px 0px 0px #A6E22E` hard offset shadow.
- **Secondary / Ghost:** Transparent background with a 1.5px solid `#111111` border, text `#111111`. Hover fills to `#111111` with `#FFFFFF` text.
- **Tertiary (Cobalt Link):** Text `#002BFF` with an aggressive underline offset (`text-underline-offset: 4px`), transitioning to a black background badge on hover.

### Chips & Badges
- Compact rectangular tags with `0px` radius. Padding `space-xs` vertical, `space-sm` horizontal.
- Types:
  - *Program / Status Badge:* Black background `#000000`, text `#A6E22E`, `label-code` uppercase.
  - *Filter Chip:* White surface with 1px solid `#E5E7EB` border. Active state: `#76B800` background with `#000000` bold text.

### Cards
- **Editorial / Course Card:** White background `#FFFFFF`, 1px solid `#111111` border, `0px` corners. Inner padding `space-lg`. Media containers inside cards have sharp edges and an optional subtle 1px border. Hover triggers a `4px 4px 0px 0px #111111` brutalist shadow and slight -2px translateY displacement.
- **Spotlight Dark Card:** Solid `#111111` background with stark `#FFFFFF` typography and `#76B800` meta badges.

### Lists
- Clean modular lists divided by sharp 1px borders (`#E5E7EB`). Hovering over a list item highlights the background in `#F4F4F6` and animates an arrow indicator in `#76B800`.
- Data lists feature small-caps `label-md` headings in muted gray `#6B7280` followed by bold `Space Grotesk` values.

### Form Inputs & Checkboxes
- **Input Fields:** 1.5px solid `#111111` border, `#FFFFFF` background, `0px` border-radius. Padding `space-md`. Focus state features a 2px solid `#76B800` border with a 2px hard black outline.
- **Checkboxes & Radios:** Sharp square boxes (`0px` radius). Checked state fills with solid `#000000` containing a `#76B800` sharp geometric check mark.

### Additional Media-Centric Components
- **Marquee Ticker:** High-contrast ribbon with `#76B800` background, `#000000` uppercase typography in `label-lg`, running horizontal news and open-day alerts.
- **Media Frame Viewport:** Image and video containers framed by thin crosshair guides or technical dimension marks (e.g., "GLR // 16:9") at the corners.