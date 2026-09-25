# Final validation — role-workspaces-v6

Prototype version: `2026-09-25-role-workspaces-v6`  
Validation date: 25 September 2026  
Live-browser status: pending deployment of this version  
Offline status: 29 regression tests passed; Python syntax compiled successfully

This log must be completed from the deployed version. Do not copy earlier model outputs into the observed-result column. Scenario controls prefill inputs only; classifications and summaries must come from the real OpenAI API, and scores/routes must come from the unchanged Python rules.

| Case | Purpose | Expected deterministic behaviour | Observed model output and route | Decision / record | Evidence | Status |
|---|---|---|---|---|---|---|
| A / REG-01 | Multiple high-risk factors | 87 rule points; all applicable independent escalation triggers; Specialist route | Pending live run | Accept recommendation; verify matching Case ID, Assessment ID, Ticket ID and version | Four Section 4.3 screenshots | Pending |
| B / REG-04 | High urgency with lower points | 55 rule points; High urgency can independently trigger Specialist route | Pending live run | Record first observed result | Appendix | Pending |
| C / REG-05 | High complexity without work impact | 50 rule points; High complexity can independently trigger Specialist route | Pending live run | Record first observed result | Appendix | Pending |
| D / REG-02 | Routine resolved case | 10 rule points; no deterministic trigger if AI returns Low urgency/complexity | Pending live run | Accept or No decision as specified | Appendix | Pending |
| E1 / REG-03 | Same fault, limited context | Score derives only from supplied structured fields and returned sentiment | Pending live run | Record first observed result | Appendix | Pending |
| E2 / REG-03B | Same fault, repeated context | Changed structured/history/emotional context changes rule inputs without preset output | Pending live run | Record first observed result | Appendix | Pending |
| F / CHECK-CONFLICT | Contradictory contacts | Block before API; exact discrepancy; no current output/record | Not applicable — no API call | No Ticket ID | Appendix | Pending browser evidence |
| G / CHECK-TWICE | False-positive regression | “Restarted the router twice” is not two support contacts | Pending live run | No discrepancy block | Appendix | Pending |
| H / LIMIT-01 | Long-running issue with incomplete history | Use current rules only; limitation statement remains visible | Pending live run | Do not infer safety or resolution from a low score/Standard route | Appendix | Pending |

## Human-oversight controls

| Control | Expected result | Evidence | Status |
|---|---|---|---|
| Blank or spaces-only Case ID | Block analysis; clear current processing evidence and record | Screenshot | Pending browser evidence |
| Empty message and history | Block analysis; clear current processing evidence and record | Screenshot | Pending browser evidence |
| Modify without a real change | Block; no new simulated record | Screenshot | Pending browser evidence |
| Modify with a change but no reason | Block; no new simulated record | Screenshot | Pending browser evidence |
| Modify with a change and reason | Preserve original and final queue/action, reason, Assessment ID and Ticket ID | Screenshot | Pending browser evidence |
| Escalate manually from Standard without reason | Block; no new simulated record | Screenshot | Pending browser evidence |
| Escalate manually from Standard with reason | Specialist route; preserve original/final decision and reason | Screenshot | Pending browser evidence |
| No decision | Awaiting officer decision; no simulated record created | Screenshot | Pending browser evidence |
| Edit any assessed input | Clear current result, processing record and simulated ticket in all current workspaces | Before/after screenshots | Pending browser evidence |
| Customer/officer sync | Customer status changes after valid officer decision for the same case | Paired screenshots | Pending browser evidence |

## Section 4.3 evidence set — same REG-01 run

All four screenshots must visibly use the same Case ID, Assessment ID and prototype version.

1. **Input:** Service Officer Workspace — complete REG-01 structured fields, history and current message; include the current trace strip after analysis.
2. **AI processing:** Processing & Audit Workspace — Validated case context and OpenAI response, including requested/returned model, response ID, timestamps and the actual structured response.
3. **Output:** Service Officer Workspace — Case intelligence plus Priority and recommended route, including summary, 87 rule points, breakdown, triggers, queue and action.
4. **Action:** Service Officer Workspace — valid officer decision and Simulated case record, including Case ID, Assessment ID, Ticket ID, timestamp, version and original/final recommendation.

## Deployment verification

- Public URL: pending
- Incognito / second-device access: pending
- Login required: must be no
- URL contains `chatgpt`: must be no
- Server-side secret configured: pending
- API key absent from screenshots and downloads: pending
- Page states “Synthetic data only” and “No live Singtel connection”: implemented; pending browser confirmation
