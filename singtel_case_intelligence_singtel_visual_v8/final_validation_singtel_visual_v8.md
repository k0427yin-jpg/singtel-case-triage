# Final validation — Singtel visual v8

Prototype version: `2026-09-25-singtel-visual-v8`  
Validation date: 25 September 2026  
Live public deployment: pending deployment of this package  
Local runtime validation: passed with Streamlit 1.50.0

## Visual scope

This release changes presentation and navigation only. It replaces the dark
sidebar-dashboard treatment with a bright service-site layout informed by the
current Singtel mobile and support pages:

- white horizontal brand and workspace navigation;
- Singtel-red primary actions, red-to-magenta accents and pale rose notices;
- charcoal typography, light-grey page bands and flatter white service cards;
- a vertical customer support journey instead of two dashboard panels;
- a compact five-stage staff workflow strip;
- developer API-key settings inside the Processing workspace.

The OpenAI prompt, model, case scenarios, 87-point rule calculation,
escalation conditions, validation gates, human-decision controls and shared
session state were not changed.

## Automated checks

- Python source parsed successfully.
- Existing automated regression suite: **30/30 passed**.
- Prompt contract unchanged: length 3,165 and SHA-256
  `12044f34b95d9aac0b24f63f08674859f80f77db2bcca593bff377dc7e74da18`.
- Streamlit runtime rendered Customer, Service officer, Review queue,
  Processing and Audit without exceptions.
- Customer submission produced a `WEB-...` case reference and the same case
  appeared in Service officer.
- The Processing workspace exposed the session-only developer key field when
  no server-side secret was configured.

## Deployment checks still required

After uploading this exact package, verify the public URL in a private window:

1. The top navigation shows all five workspaces without opening a sidebar.
2. The page header is white/light rose with a red gradient accent; no dark
   navy dashboard shell remains.
3. Customer submission, officer assessment, human decision and customer status
   still update in one browser session.
4. The Processing workspace shows the real OpenAI response ID, requested and
   returned model, timestamps and structured response after a valid run.
5. Capture the four Section 4.3 screenshots from this same version: input,
   AI processing, output and business action.

## Design references

- Singtel mobile: https://www.singtel.com/personal/products-services/mobile
- Singtel support: https://www.singtel.com/personal/support
- Singtel broadband support: https://www.singtel.com/personal/support/broadband

The interface is a classroom prototype inspired by those public design cues.
It is not represented as an official Singtel product.
