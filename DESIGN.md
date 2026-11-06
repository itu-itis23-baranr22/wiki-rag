---
name: Wiki-RAG
description: A local question-answering tool where every citation is a highlighter stroke across its source sentence.
colors:
  toner: "#111111"
  photocopy-grey: "#e8e9e6"
  sheet-white: "#f3f4f1"
  field-white: "#fbfbf9"
  muted-toner: "#55574f"
  hairline: "#c3c5be"
  ink-yellow: "#f4ff3a"
  ink-pink: "#ff9ccb"
  ink-green: "#8cffb0"
  ink-orange: "#ffc266"
  ink-blue: "#9edcff"
  danger: "#a3123f"
  danger-dark-theme: "#ff6b5e"
  negative-sheet: "#1b1c19"
  negative-field: "#232420"
  negative-muted: "#a9aba4"
  negative-hairline: "#3d3e39"
typography:
  display:
    fontFamily: "Anek Latin, Segoe UI, system-ui, sans-serif"
    fontSize: "2rem"
    fontWeight: 800
    lineHeight: 1.1
  title:
    fontFamily: "Anek Latin, Segoe UI, system-ui, sans-serif"
    fontSize: "1.375rem"
    fontWeight: 800
    lineHeight: 1.2
  answer:
    fontFamily: "Anek Latin, Segoe UI, system-ui, sans-serif"
    fontSize: "1.1875rem"
    fontWeight: 400
    lineHeight: 1.55
  source:
    fontFamily: "Tinos, Times New Roman, serif"
    fontSize: "1.09375rem"
    fontWeight: 400
    lineHeight: 1.62
  body:
    fontFamily: "Anek Latin, Segoe UI, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Anek Latin, Segoe UI, system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 800
    letterSpacing: "0.08em"
  display-phone:
    fontFamily: "Anek Latin, Segoe UI, system-ui, sans-serif"
    fontSize: "1.625rem"
    fontWeight: 800
    lineHeight: 1.1
  body-small:
    fontFamily: "Anek Latin, Segoe UI, system-ui, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.5
  control:
    fontFamily: "Anek Latin, Segoe UI, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 600
  caption:
    fontFamily: "Anek Latin, Segoe UI, system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 400
rounded:
  none: "0"
spacing:
  gutter-phone: "16px"
  gutter-desktop: "24px"
  group: "12px"
  section: "44px"
components:
  button-primary:
    backgroundColor: "{colors.toner}"
    textColor: "{colors.photocopy-grey}"
    rounded: "{rounded.none}"
    padding: "8px 16px"
  button-primary-hover:
    backgroundColor: "{colors.field-white}"
    textColor: "{colors.toner}"
  input:
    backgroundColor: "{colors.field-white}"
    textColor: "{colors.toner}"
    rounded: "{rounded.none}"
    padding: "8px 10px"
  nav-current:
    backgroundColor: "{colors.ink-yellow}"
    textColor: "{colors.toner}"
    padding: "2px 8px"
  refusal-block:
    backgroundColor: "{colors.toner}"
    textColor: "{colors.sheet-white}"
    padding: "18px 20px"
---

# Design System: Wiki-RAG

## Overview

**Creative North Star: "The Marked-Up Reading Packet"**

Wiki-RAG looks like a photocopied course reader someone has been through with a highlighter. That is not decoration. The product's claim is that an answer can be checked, and a stroke of the same color on an answer sentence and on its source sentence shows that claim working.

The screen stays a working tool: a top bar, two panes, standard form controls. The world contributes type, palette, density and one move, the highlighter stroke. It is flat toner on grey paper, with no texture, grain or faked physical material. The dark theme is a negative photostat of the same page, and the fluorescent inks stay fluorescent on it.

**Key Characteristics:**

- Two voices: heavy sans for what the model wrote, photocopied book serif for what Wikipedia says.
- Fluorescent ink means "this came from a source", and each ink belongs to one source number.
- Square corners and 2px toner rules everywhere.
- Refusing is a solid toner block, as deliberate as an answer.
- Honest about waiting: stages and elapsed time, never a bare spinner.

## Colors

Photocopy neutrals carry the page, and five fluorescent inks do one job.

### Primary

- **Toner** (`toner`): text, rules, the top bar, primary buttons and the refusal block. In the dark theme it becomes the page ground.
- **Photocopy Grey** (`photocopy-grey`): the page ground in the light theme, and the text and rules in the dark theme.

### Secondary

- **The five inks** (`ink-yellow`, `ink-pink`, `ink-green`, `ink-orange`, `ink-blue`): the highlight on an answer sentence, its source sentence and its citation chip. Source numbers cycle through them in that order. Text on an ink is always toner, in both themes.

### Neutral

- **Sheet White** (`sheet-white`): the source sheet and notices, one step lighter than the page.
- **Field White** (`field-white`): inputs and selects.
- **Muted Toner** (`muted-toner`): supporting text, always at readable contrast. Dark theme: `negative-muted`.
- **Hairline** (`hairline`): quiet dividers between rows. Dark theme: `negative-hairline`.
- **Danger** (`danger`, dark theme `danger-dark-theme`): setup problems and errors only. It is deliberately distinct from every ink.

### Named Rules

**The One Job Rule.** Wherever sources are shown, an ink means a source. It never marks status, selection, warnings or a button's hover. Outside that job, yellow marks the current tab, model activity, and the row under the pointer on screens showing no sources. On Measurements, where no source numbers appear, yellow draws a gain and pink a loss.

**The Plain Selection Rule.** Selected text is inverted toner. A fluorescent selection would look like a citation to source 1.

## Typography

**Interface and model voice:** Anek Latin, self-hosted as a variable font (400 to 800).
**Source voice:** Tinos, self-hosted.
**Code:** the system monospace, only for commands and code.

**Character:** a sturdy, slightly wide sans at heavy weights against the plain book serif of a photocopied page. The contrast is functional: it tells you at a glance who is speaking.

### Hierarchy

- **Display** (800, 2rem, 1.1): page headings and the current question. 1.625rem on phones.
- **Title** (800, 1.375rem, 1.2): section headings and earlier questions in a conversation.
- **Answer** (400, 1.1875rem, 1.55): the model's answer, at most 62ch.
- **Source** (400, 1.09375rem, 1.62): source passages, at most 72ch.
- **Body** (400, 1rem, 1.5): interface text.
- **Label** (800, 0.75rem, 0.08em tracking, uppercase): page names on passages and group headings on the contents page.

### Named Rules

**The Two Voices Rule.** Text from Wikipedia is set in the serif, including page names in Contents and Library. Everything the model or the interface says is in the sans.

**The Turkish Case Rule.** Uppercase English names carry `lang="en"`, so Turkish casing never turns "Hagia Sophia" into "HAGİA SOPHİA".

## Layout

The Ask view is two equal panes split by a 2px rule, each scrolling on its own, with the question box pinned at the bottom of the left pane. Library and Measurements are a single column up to 62rem. Below 861px everything becomes one scrolling page with the source sheet after the answer, the navigation on its own row, and the question box pinned to the bottom edge. Density stays high in the source sheet, where each passage is a tight module under a rule. Sizes are a fixed rem scale, not fluid.

## Elevation & Depth

There are no shadows. Depth comes from the two neutral layers and from rules, and the settings popover is separated by its 2px border alone.

**The Flat Toner Rule.** Nothing casts a shadow, and nothing imitates a physical material. A photocopy is flat.

## Shapes

Every corner is square (`rounded.none`), including inputs, buttons and the slider thumb. The only round shapes are the score dots on Measurements, which are data marks. Borders are 2px toner for structure, 1px hairline for rows.

## Components

### Highlight and citation

The signature. When an answer finishes, one stroke runs left to right across each answer sentence and its source sentence together: 320ms with a strong ease-out, 70ms apart, then the transition is removed. Hovering or focusing a citation leaves only that pair inked, instantly, because it happens constantly. With reduced motion the strokes fade in. Citation chips are small squares in their source's ink. A dashed chip means the sentence was matched by comparing text rather than cited by the model, and a dashed "no source" label means nothing supports it.

### Buttons

- **Shape:** square (0).
- **Primary:** toner fill with grey text, weight 800.
- **Hover:** inverts to field white with toner text.
- **Press:** scale 0.97 over 120ms.
- **Disabled:** no fill, muted text, not-allowed cursor.

### Inputs

- **Style:** 2px toner border on field white, 16px text or larger so phones don't zoom.
- **Focus:** a second 2px toner outline.
- **Slider:** 2px track with a square 12 by 20px thumb.

### Navigation

Text links in the top bar. The current view gets a yellow tab, and other links get a toner underline on hover. In the dark theme the bar inverts to grey with toner text.

### Refusal

A solid toner block in the answer's place: the refusal in 800 weight with one line explaining what was checked. The source sheet beside it shows the passages it checked, muted, with no highlights at all.

### Measurements dumbbell

One row per question type with its question count. A hollow dot marks the original version and a solid dot the current one, joined by a stroke: yellow for a gain, pink for a loss. The values are always printed beside the row.

## Do's and Don'ts

### Do:

- **Do** keep each ink tied to one source number wherever sources are shown.
- **Do** set anything quoted from Wikipedia in the serif.
- **Do** show real recorded content in demonstrations, and label recordings as recordings.
- **Do** self-host every font. The app must work offline and contact nothing while answering.
- **Do** name the problem and the exact command in setup and error messages.

### Don't:

- **Don't** use an ink for warnings, status, selection or button hovers.
- **Don't** add shadows, grain, paper texture or other imitation material.
- **Don't** round corners on controls.
- **Don't** animate hovers or anything triggered by the keyboard. The stroke happens once per answer.
- **Don't** use a font or asset from a CDN.
