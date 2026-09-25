"""
Singtel AI Complex Case Intelligence and Escalation System — prototype
Streamlit + OpenAI API version.

Pipeline:
1. Synthetic case input (this app, form or free text — structured fields,
   previous interaction history, and the current customer message)
2. AI classification + summary (OpenAI API call — genuine AI functionality;
   the summary draws on both the interaction history and current message)
2b. Deterministic missing-information check (plain Python — not the LLM;
    moved here after testing showed prompt-only detection was inconsistent)
3. Deterministic rule-based scoring (plain Python — not the LLM)
4. Human decision gate (Accept / Modify / Escalate)
5. Simulated case record (generated only after a decision, by this app,
   with a real timestamp and generated ID — never invented by the model)

No live Singtel systems are contacted. All data is synthetic.
"""

import json
import random
import re
import string
from datetime import datetime, timezone, timedelta

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Singtel Case Triage (Prototype)", page_icon="\U0001F4E1", layout="wide")

SGT = timezone(timedelta(hours=8))
APP_VERSION = "2026-09-25-role-workspaces-v6"
MODEL_NAME = "gpt-4o-mini"
MAX_RULE_POINTS = 87

SYSTEM_PROMPT = """You are an AI case-assessment assistant for a prototype called
"Singtel AI Complex Case Intelligence and Escalation System". You provide decision
support for a human service officer handling SIMULATED Singtel home-broadband cases.

Use only the information given in the case: the structured fields, the
previous interaction history (if provided), and the current customer
message. Do not invent facts, prior contacts, or Singtel system data beyond
what is explicitly given. Do not make the final decision — a human service
officer retains final authority.

Classify the case and return STRICT JSON with exactly these keys:
{
  "intent": "<one short sentence>",
  "complexity": "Low" | "Medium" | "High",
  "urgency": "Low" | "Medium" | "High",
  "sentiment": "Positive" | "Neutral" | "Frustrated",
  "summary": "<concise grounded summary of the case, drawing on the previous
    interaction history and the current customer message together, based
    only on the information given>"
}

Complexity classification rules:
- High: at least two previous support contacts AND the problem remains unresolved,
  OR the case clearly involves substantial specialist complexity.
- Medium: needs additional investigation/coordination but does not meet High criteria.
- Low: straightforward, no repeated unresolved contact history.
Work/critical-activity impact affects Urgency, not Complexity, by itself.

Urgency classification rules:
- High: an unresolved problem significantly affects the customer's work or another
  critical activity.
- Medium: meaningful impact or requires attention, no serious immediate consequence.
- Low: limited immediate impact.
Wanting a fast fix does not by itself mean High urgency.

Sentiment classification rules:
- Determine sentiment ONLY from the emotional wording in the customer message.
- Do NOT infer sentiment from previous contacts, unresolved status, work impact,
  urgency, complexity, service severity, or repeated support history.
- Frustrated: only when the customer explicitly expresses frustration, anger,
  annoyance, disappointment, or similarly clear negative emotion.
- Neutral: the customer describes a problem, inconvenience, repeated contacts,
  unresolved service, work impact, or asks for help WITHOUT explicit negative
  emotional language.
- Positive: the customer explicitly expresses satisfaction or positive emotion.

Examples:
- "I am extremely frustrated because this keeps happening." -> Frustrated
- "I contacted support twice and the issue is still unresolved. Please investigate." -> Neutral
- "I cannot join work calls. Please help when possible." -> Neutral

IMPORTANT: A serious or urgent case is NOT automatically a frustrated customer.
If there is no explicit emotional evidence in the customer message, classify
sentiment as Neutral.

Summary rules:
- Include documented troubleshooting or prior actions and any pending next steps
  when present, such as a router restart or scheduled line test.
- Distinguish completed actions from planned actions; do not infer their outcome.
- Keep these facts concise and grounded in the supplied history and message.

Return ONLY the JSON object, no other text.
"""


DEFAULT_HISTORY = (
    "Contact 1 (5 days ago): Customer reported intermittent disconnections. "
    "Advised to restart the router.\n"
    "Contact 2 (3 days ago): Issue persisted after restart. A line test was "
    "scheduled.\n"
    "Contact 3 (1 day ago): Customer called again; problem still unresolved.\n"
    "Contact 4 (today): Customer followed up because disconnections continued "
    "to interrupt work calls; no resolution was recorded."
)

DEFAULT_MESSAGE = (
    "My home broadband keeps disconnecting and this problem has still not "
    "been fixed. I have contacted support four times already. I work from "
    "home and the connection drops are preventing me from joining client "
    "video calls. I am extremely frustrated because I have had to explain "
    "the same problem repeatedly."
)

# Scenario controls prefill inputs only. They never prescribe a model output,
# rule score or route. Every valid run still calls the configured model and
# applies the unchanged Python logic below.
CASE_SCENARIOS = {
    "A · Multiple high-risk factors": {
        "case_id": "REG-01", "prior_contacts": 4, "unresolved": True,
        "work_impact": True, "history_text": DEFAULT_HISTORY,
        "case_text": DEFAULT_MESSAGE,
    },
    "B · High urgency, lower rule points": {
        "case_id": "REG-04", "prior_contacts": 1, "unresolved": True,
        "work_impact": True,
        "history_text": "Support advised a router restart yesterday. No resolution was recorded.",
        "case_text": (
            "The broadband is still disconnecting. I restarted the router twice, "
            "but I still cannot join client video calls while working from home. "
            "Please investigate."
        ),
    },
    "C · High complexity, no work impact": {
        "case_id": "REG-05", "prior_contacts": 3, "unresolved": True,
        "work_impact": False,
        "history_text": (
            "Contact 1: A router restart was recommended.\n"
            "Contact 2: A remote line test was completed.\n"
            "Contact 3: The intermittent issue remained unresolved."
        ),
        "case_text": (
            "The broadband remains intermittent on all devices after three support "
            "contacts. Please continue the investigation."
        ),
    },
    "D · Routine resolved case": {
        "case_id": "REG-02", "prior_contacts": 0, "unresolved": False,
        "work_impact": False, "history_text": "",
        "case_text": (
            "Wi-Fi was slow on my laptop for about 20 minutes yesterday evening. "
            "I restarted the router and laptop, and it is working normally now. "
            "I am happy that the issue is resolved."
        ),
    },
    "E1 · Same fault, limited context": {
        "case_id": "REG-03", "prior_contacts": 0, "unresolved": True,
        "work_impact": False, "history_text": "",
        "case_text": (
            "My home broadband is not working on all devices since this morning. "
            "I have not tried restarting the router yet. Please help."
        ),
    },
    "E2 · Same fault, repeated context": {
        "case_id": "REG-03B", "prior_contacts": 2, "unresolved": True,
        "work_impact": False,
        "history_text": (
            "Contact 1: Customer reported no connection on all devices. A router restart was advised.\n"
            "Contact 2: The same fault continued after the restart; a line test was scheduled."
        ),
        "case_text": (
            "My home broadband is still not working on all devices after two support contacts. "
            "I am annoyed that the same fault continues. Please investigate."
        ),
    },
    "F · Contradictory contact count": {
        "case_id": "CHECK-CONFLICT", "prior_contacts": 4, "unresolved": True,
        "work_impact": False, "history_text": "",
        "case_text": (
            "I contacted support twice. My broadband is still not working. "
            "Please investigate."
        ),
    },
    "G · Router restarted twice": {
        "case_id": "CHECK-TWICE", "prior_contacts": 1, "unresolved": True,
        "work_impact": True,
        "history_text": "Support advised a router restart yesterday.",
        "case_text": (
            "I restarted the router twice today, but the broadband is still "
            "intermittent and I cannot join work calls."
        ),
    },
    "H · Long-running issue, incomplete history": {
        "case_id": "LIMIT-01", "prior_contacts": 1, "unresolved": True,
        "work_impact": False,
        "history_text": "One short note records an earlier broadband enquiry.",
        "case_text": (
            "This broadband problem has continued for a long time. I believe I have "
            "asked for help many times before, but only limited details are available "
            "here. The connection is still intermittent. Please review the case."
        ),
    },
}

LONG_TERM_LIMITATION = (
    "This prototype can assess only the information supplied in the current "
    "structured fields, interaction history and message. It does not have access "
    "to the customer’s complete long-term contact record, network telemetry, "
    "technician records, outage confirmation or account data. A low score or "
    "Standard route does not prove that the service problem is minor or resolved."
)


# ---------------------------------------------------------------------------
# Step 2: AI classification (genuine AI functionality — real API call)
# ---------------------------------------------------------------------------

def classify_case(case_text: str, api_key: str) -> dict:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case_text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw_content = response.choices[0].message.content
    result = json.loads(raw_content)
    result["_api_evidence"] = {
        "response_id": response.id,
        "returned_model": response.model,
        "raw_response": raw_content,
    }
    return result


# ---------------------------------------------------------------------------
# Step 2b: Deterministic missing-information check — plain Python, NOT the
# LLM. Initial testing showed prompt-only missing-information detection was
# inconsistent (see Section 4.4): the same completeness rules were applied
# unevenly across cases. This keyword-based check is transparent and repeats
# identically on the same input, matching the design principle already used
# for priority scoring: the LLM handles semantic interpretation (intent,
# complexity, urgency, sentiment, summary), Python handles auditable rules.
# ---------------------------------------------------------------------------

def check_missing_information(case_text: str) -> str:
    text = case_text.lower()
    missing = []

    duration_terms = [
        "minute", "minutes", "hour", "hours", "day", "days",
        "week", "weeks", "since this morning", "since yesterday",
        "for a while",
    ]
    scope_terms = [
        "all devices", "all services", "only my", "only one",
        "laptop only", "phone only", "some devices",
    ]
    troubleshooting_terms = [
        "restart", "restarted", "reset", "checked the cables",
        "check the cables", "reboot", "troubleshoot",
    ]
    status_terms = [
        "not working", "fully down", "disconnected",
        "disconnecting", "intermittent", "slow",
        "working", "working normally", "resolved",
        "back to normal", "restored", "fixed",
    ]

    if not any(term in text for term in duration_terms):
        missing.append("duration")
    if not any(term in text for term in scope_terms):
        missing.append("scope")
    if not any(term in text for term in troubleshooting_terms):
        missing.append("troubleshooting")
    if not any(term in text for term in status_terms):
        missing.append("current status")

    return ", ".join(missing) if missing else "None"


# ---------------------------------------------------------------------------
# Step 1b: Input quality gate — plain Python. Blocks the AI call rather than
# generating an assessment from insufficient information.
# ---------------------------------------------------------------------------

def has_meaningful_content(text: str, min_len: int = 10) -> bool:
    return bool(text) and len(text.strip()) >= min_len


# ---------------------------------------------------------------------------
# Step 2c: Deterministic contradiction check — plain Python, NOT the LLM.
# Looks for a contact count mentioned in the free text that does not match
# the structured "Previous support contacts" field, so the officer can
# verify which is correct before acting on the assessment.
# ---------------------------------------------------------------------------

_SUFFIXED_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_STANDALONE_NUMBER_WORDS = {"once": 1, "twice": 2}

# Counts must refer explicitly to support contact, never arbitrary actions
# such as restarting a router twice or checking cables two times.
_CONTACT_NUMBER = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
_CONTACT_COUNT = rf"(?:once|twice|{_CONTACT_NUMBER}\s+times)"
_CONTACT_CONTEXT_PATTERN = re.compile(
    rf"\b(?:contacted|called|spoke\s+to|reached)\s+"
    rf"(?:(?:the\s+)?(?:support|customer\s+service|helpdesk|hotline|"
    rf"Singtel|you)(?:\s+team)?\s+)?"
    rf"(?:already\s+)?({_CONTACT_COUNT})\b",
    re.IGNORECASE,
)
_EXPLICIT_CONTACT_PATTERN = re.compile(
    rf"\b({_CONTACT_NUMBER})\s+(?:(?:previous|prior|support)\s+)*contacts\b",
    re.IGNORECASE,
)


def extract_mentioned_contact_counts(text: str) -> list:
    text = text.lower()
    mentions = _CONTACT_CONTEXT_PATTERN.findall(text)
    mentions += _EXPLICIT_CONTACT_PATTERN.findall(text)
    counts = []
    for mention in mentions:
        word = mention.split()[0]
        if word in _STANDALONE_NUMBER_WORDS:
            counts.append(_STANDALONE_NUMBER_WORDS[word])
        elif word.isdigit():
            counts.append(int(word))
        else:
            counts.append(_SUFFIXED_NUMBER_WORDS[word])
    return counts


def detect_contact_count_discrepancy(prior_contacts: int, case_text: str, history_text: str):
    combined = f"{case_text}\n{history_text}"
    for mentioned in extract_mentioned_contact_counts(combined):
        if mentioned != prior_contacts:
            return (
                f"The customer message or interaction history mentions "
                f"{mentioned} prior contact(s), but the 'Previous support "
                f"contacts' field is set to {prior_contacts}."
            )
    return None


# ---------------------------------------------------------------------------
# Step 3: Deterministic rule-based scoring — plain Python, NOT the LLM.
# This is the "automation logic" component the assessment brief asks for,
# kept separate from the AI classification step above.
# ---------------------------------------------------------------------------

def calculate_priority(prior_contacts: int, unresolved: bool, work_impact: bool,
                        sentiment: str, complexity: str, urgency: str) -> dict:
    breakdown = []
    score = 10
    breakdown.append(("Base", 10))

    pts = 20 if prior_contacts >= 2 else 0
    score += pts
    breakdown.append((f"Contacts >= 2 ({prior_contacts})", pts))

    pts = 20 if unresolved else 0
    score += pts
    breakdown.append(("Unresolved", pts))

    pts = 25 if work_impact else 0
    score += pts
    breakdown.append(("Work impact", pts))

    pts = 12 if sentiment.lower() == "frustrated" else 0
    score += pts
    breakdown.append(("Frustrated sentiment", pts))

    triggers = []
    if urgency.lower() == "high":
        triggers.append("High urgency")
    if complexity.lower() == "high":
        triggers.append("High complexity")
    if score > 70:
        triggers.append("Priority score > 70")
    escalate = bool(triggers)

    breakdown_str = " + ".join(f"{label} {val}" for label, val in breakdown) + f" = {score}"

    return {
        "score": score,
        "breakdown": breakdown_str,
        "escalation_recommended": escalate,
        "trigger": "; ".join(triggers) if triggers else "None",
    }


# ---------------------------------------------------------------------------
# Step 5: Simulated business action — generated by this app's own code,
# only after an explicit human decision. Never invented by the model.
# ---------------------------------------------------------------------------

def create_simulated_record(case_id: str, priority_score: int, assigned_queue: str,
                             officer_decision: str, approved_action: str,
                             original_queue: str = "", original_action: str = "",
                             override_reason: str = "", assessment_id: str = "") -> dict:
    case_id = case_id.strip()
    if not case_id:
        return {
            "status": "BLOCKED - a non-blank Case ID is required",
            "ticket_id": "",
            "timestamp": "",
        }

    valid_decisions = {"accept", "modify", "escalate"}
    if officer_decision.lower() not in valid_decisions:
        return {
            "status": "BLOCKED - no valid officer decision (Accept, Modify or Escalate)",
            "ticket_id": "",
            "timestamp": "",
        }

    override_required = (
        officer_decision.lower() == "modify"
        or (officer_decision.lower() == "escalate"
            and (assigned_queue != original_queue or approved_action != original_action))
    )
    if override_required and not override_reason.strip():
        return {
            "status": "BLOCKED - an override reason is required",
            "ticket_id": "",
            "timestamp": "",
        }

    assessment_id = assessment_id.strip()
    if not assessment_id:
        return {
            "status": "BLOCKED - a traceable Assessment ID is required",
            "ticket_id": "",
            "timestamp": "",
        }

    ticket_id = "SIM-" + datetime.now(SGT).strftime("%Y%m%d") + "-" + \
        "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    timestamp = datetime.now(SGT).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": "Created for simulation",
        "case_id": case_id,
        "priority_score": priority_score,
        "score_unit": "rule points",
        "max_rule_points": MAX_RULE_POINTS,
        "prototype_version": APP_VERSION,
        "assessment_id": assessment_id,
        "assigned_queue": assigned_queue,
        "human_decision": officer_decision,
        "approved_action": approved_action,
        "original_queue": original_queue,
        "original_action": original_action,
        "override_reason": override_reason.strip(),
        "ticket_id": ticket_id,
        "timestamp": timestamp,
    }


def clear_assessment(state):
    """Discard all outputs tied to a previous assessment, including its ticket."""
    for key in ("result", "priority", "record", "assessed_inputs", "discrepancy",
                "processing_details", "decision_draft", "validation_event"):
        state[key] = None
    state["decision_state"] = "Awaiting analysis"


def invalidate_changed_inputs(state, inputs):
    snapshot = state.get("assessed_inputs")
    if snapshot is not None and inputs != snapshot:
        clear_assessment(state)
        state["assessment_invalidated"] = True
        return True
    return False


def run_assessment(state, inputs, api_key):
    """Validate before model use; publish a complete result only on success."""
    clear_assessment(state)
    state["assessment_invalidated"] = True
    attempted_at = datetime.now(SGT)

    def block_before_api(message):
        state["validation_event"] = {
            "validation_id": "VALIDATE-" + attempted_at.strftime("%Y%m%d-%H%M%S-%f"),
            "case_id": inputs.get("case_id", "").strip() or "Not supplied",
            "status": "Blocked before API call",
            "message": message,
            "attempted_inputs": dict(inputs),
            "timestamp_sgt": attempted_at.isoformat(timespec="seconds"),
            "prototype_version": APP_VERSION,
        }
        return message

    if not inputs["case_id"].strip():
        return block_before_api("Enter a Case ID before analysing the case. Spaces alone are not valid.")
    if not has_meaningful_content(inputs["case_text"]) and not has_meaningful_content(inputs["history_text"]):
        return block_before_api(
            "Not enough information to assess this case. Enter a customer "
            "message or previous interaction history with enough detail "
            "before analysing the case."
        )
    discrepancy = detect_contact_count_discrepancy(
        inputs["prior_contacts"], inputs["case_text"], inputs["history_text"],
    )
    if discrepancy:
        return block_before_api(
            f"Discrepancy detected — officer verification required. {discrepancy} "
            "No AI call was made. Correct the case information before continuing."
        )
    if not api_key:
        return block_before_api(
            "Prototype processing is not configured. Add the developer API key in Processing & audit."
        )
    ai_case_input = f"""Service: Home Broadband
Previous support contacts: {inputs['prior_contacts']}
Issue still unresolved: {"Yes" if inputs['unresolved'] else "No"}
Affects work / critical activity: {"Yes" if inputs['work_impact'] else "No"}

Previous interaction history:
{inputs['history_text'].strip() or "None recorded."}

Customer message:
{inputs['case_text']}"""
    started_at = datetime.now(SGT)
    try:
        ai_result = classify_case(ai_case_input, api_key)
        completed_at = datetime.now(SGT)
        api_evidence = ai_result.pop("_api_evidence", {})
        # Snapshot only the model output, before Python adds its keyword check.
        model_output = dict(ai_result)
        # Validate required model fields before committing any UI state.
        for key in ("intent", "summary"):
            if not isinstance(ai_result.get(key), str) or not ai_result[key].strip():
                raise ValueError(f"Invalid model field: {key}")
        for key, allowed in {
            "complexity": {"Low", "Medium", "High"},
            "urgency": {"Low", "Medium", "High"},
            "sentiment": {"Positive", "Neutral", "Frustrated"},
        }.items():
            if ai_result.get(key) not in allowed:
                raise ValueError(f"Invalid model field: {key}")
        ai_result["missing_info"] = check_missing_information(inputs["case_text"])
        priority = calculate_priority(
            inputs["prior_contacts"], inputs["unresolved"], inputs["work_impact"],
            ai_result["sentiment"], ai_result["complexity"], ai_result["urgency"],
        )
    except Exception:
        return "Case analysis failed. No current result or ticket is available. Please try again."
    state["processing_details"] = {
        "assessment_id": "ASSESS-" + started_at.strftime("%Y%m%d-%H%M%S-%f"),
        "prototype_version": APP_VERSION,
        "case_id": inputs["case_id"].strip(),
        "requested_model": MODEL_NAME,
        "returned_model": api_evidence.get("returned_model"),
        "response_id": api_evidence.get("response_id"),
        "started_at_sgt": started_at.isoformat(timespec="seconds"),
        "completed_at_sgt": completed_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round((completed_at - started_at).total_seconds(), 3),
        "case_context": ai_case_input,
        "system_prompt": SYSTEM_PROMPT,
        "model_output": model_output,
        "raw_response": api_evidence.get("raw_response"),
    }
    state["result"] = ai_result
    state["priority"] = priority
    state["case_id"] = inputs["case_id"].strip()
    state["assessed_inputs"] = dict(inputs)
    state["assessment_run"] = state.get("assessment_run", 0) + 1
    state["assessment_invalidated"] = False
    state["decision_state"] = "Awaiting officer decision"
    return None


def recommended_route(priority: dict) -> tuple:
    if priority["escalation_recommended"]:
        return (
            "Specialist broadband escalation queue",
            "Route for escalation and further investigation.",
        )
    return (
        "Standard broadband support queue",
        "Provide standard troubleshooting guidance; no escalation required based on current information.",
    )


def extract_operational_notes(history_text: str) -> dict:
    """Extract auditable note fragments for review; this does not affect scoring."""
    fragments = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", history_text or "")
        if part.strip()
    ]
    troubleshooting_terms = ("restart", "reset", "reboot", "cable", "line test", "diagnostic")
    pending_terms = ("scheduled", "booked", "pending", "follow-up", "follow up", "appointment", "engineer")
    troubleshooting = [part for part in fragments if any(term in part.lower() for term in troubleshooting_terms)]
    pending = [part for part in fragments if any(term in part.lower() for term in pending_terms)]
    return {
        "troubleshooting": " ".join(troubleshooting) or "Not documented in the supplied history.",
        "pending_action": " ".join(pending) or "Not documented in the supplied history.",
    }


def officer_final_status(state) -> dict:
    decision_state = state.get("decision_state", "Awaiting analysis")
    record = state.get("record")
    if decision_state == "Recommendation accepted" and record:
        return {"label": decision_state, "detail": "A simulated case record was created.", "tone": "success"}
    if decision_state == "Recommendation modified" and record:
        return {"label": decision_state, "detail": "The original and revised actions are retained in the record.", "tone": "success"}
    if decision_state == "Escalated for specialist review" and record:
        return {"label": decision_state, "detail": "The specialist route and officer reason are recorded.", "tone": "success"}
    if decision_state == "Assessment blocked":
        return {"label": decision_state, "detail": "No simulated record created.", "tone": "error"}
    if state.get("result"):
        return {"label": "Awaiting officer decision", "detail": "No simulated record created.", "tone": "warning"}
    return {"label": "Awaiting analysis", "detail": "No current case intelligence or simulated record.", "tone": "neutral"}


def validate_officer_decision(priority: dict, decision: str, proposed_queue: str,
                              proposed_action: str, override_reason: str) -> dict:
    """Validate and normalise a human decision without creating a ticket."""
    default_queue, default_action = recommended_route(priority)
    final_queue = proposed_queue.strip()
    final_action = proposed_action.strip()
    reason = override_reason.strip()

    if decision == "Accept":
        if final_queue != default_queue or final_action != default_action:
            return {
                "valid": False,
                "error": (
                    "Accept recommendation is available only when the queue and action are unchanged. "
                    "Use Modify recommendation for revised details."
                ),
            }
    elif decision == "Modify":
        if final_queue == default_queue and final_action == default_action:
            return {
                "valid": False,
                "error": (
                    "Modify recommendation requires an actual change to the queue or action. "
                    "Use Accept recommendation if the suggested route is correct."
                ),
            }
        if not reason:
            return {"valid": False, "error": "Enter an officer reason before selecting Modify recommendation."}
    elif decision == "Escalate":
        if priority["escalation_recommended"]:
            return {
                "valid": False,
                "error": (
                    "Specialist escalation is already recommended. Use Accept recommendation "
                    "or Modify recommendation."
                ),
            }
        custom_note = final_action if final_action != default_action else ""
        final_queue = "Specialist broadband escalation queue"
        final_action = "Route for escalation and further investigation."
        if custom_note:
            final_action += f" Officer note: {custom_note}"
        if not reason:
            return {
                "valid": False,
                "error": "Enter an officer reason before changing the escalation recommendation.",
            }
    else:
        return {"valid": False, "error": "No valid officer decision was selected."}

    if not final_queue or not final_action:
        return {
            "valid": False,
            "error": "A final queue and action are required before recording a decision.",
        }
    return {
        "valid": True,
        "decision": decision,
        "final_queue": final_queue,
        "final_action": final_action,
        "override_reason": reason,
        "original_queue": default_queue,
        "original_action": default_action,
    }


def customer_status_from_state(state):
    """Return customer-safe status copy without exposing internal AI or rule data."""
    request = state.get("customer_request")
    if not request:
        return {
            "stage": "Not submitted",
            "title": "Tell us what is happening",
            "message": "Submit a support request to receive a case reference.",
            "tone": "neutral",
        }

    record = state.get("record")
    request_case_id = request.get("case_id")
    current_case_id = (state.get("assessed_inputs") or {}).get("case_id")
    same_assessed_case = current_case_id == request_case_id
    same_record_case = bool(record and record.get("case_id") == request_case_id)

    if state.get("decision_state") == "Assessment blocked" and same_assessed_case:
        return {
            "stage": "Under review",
            "title": "Your request is still being reviewed",
            "message": "No support action has been confirmed yet. Your case remains open.",
            "tone": "warning",
        }

    if same_record_case and record.get("status") == "Created for simulation":
        specialist = "specialist" in record.get("assigned_queue", "").lower()
        if specialist:
            return {
                "stage": "Specialist review arranged",
                "title": "Your case has been referred for further review",
                "message": (
                    "A specialist support review has been arranged. This does not "
                    "mean the service issue is resolved; an officer will follow up."
                ),
                "tone": "success",
                "ticket_id": record.get("ticket_id"),
            }
        return {
            "stage": "Support action confirmed",
            "title": "Your support request has been reviewed",
            "message": (
                "A service officer has confirmed the next support action. "
                "The team will follow up using the contact channel for this case."
            ),
            "tone": "success",
            "ticket_id": record.get("ticket_id"),
        }

    if state.get("result") and same_assessed_case:
        return {
            "stage": "Officer review",
            "title": "Your request is being reviewed",
            "message": "A service officer is reviewing the case before confirming the next action.",
            "tone": "info",
        }

    return {
        "stage": "Request received",
        "title": "We have received your support request",
        "message": "Your case is waiting for service review. No outcome has been confirmed yet.",
        "tone": "info",
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

# Role-separated interface. Only the selected workspace is rendered, so the
# customer journey, officer tools and technical evidence are visually and
# operationally distinct while reading the same in-session state.

WORKSPACES = [
    "01 · Customer support",
    "02 · Service officer",
    "03 · Review queue",
    "04 · Processing & audit",
    "05 · Audit trail",
]


def initialise_ui_state():
    defaults = {
        "case_draft": dict(CASE_SCENARIOS["A · Multiple high-risk factors"]),
        "case_revision": 0,
        "loaded_scenario": "A · Multiple high-risk factors",
        "result": None,
        "priority": None,
        "record": None,
        "assessment_run": 0,
        "assessed_inputs": None,
        "discrepancy": None,
        "processing_details": None,
        "assessment_invalidated": False,
        "assessment_error": None,
        "last_attempt_inputs": None,
        "decision_state": "Awaiting analysis",
        "decision_draft": None,
        "validation_event": None,
        "customer_request": None,
        "audit_log": [],
        "api_key_cache": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_page_header(number, eyebrow, title, subtitle, role):
    st.markdown(
        f"""
        <div class="page-hero">
          <div class="hero-topline"><span>{number} / {eyebrow}</span><span class="role-badge">{role}</span></div>
          <h1>{title}<span class="red-dot">.</span></h1>
          <p>{subtitle}</p>
          <div class="hero-tags"><span>Synthetic data only</span><span>No live Singtel connection</span><span>{APP_VERSION}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trace_strip():
    details = st.session_state.get("processing_details")
    if not details:
        return
    with st.container(border=True):
        st.caption("CURRENT PROCESSING TRACE")
        case_col, assessment_col, version_col = st.columns([1, 2, 2])
        case_col.markdown(f"**Case ID**  \n`{details['case_id']}`")
        assessment_col.markdown(f"**Assessment ID**  \n`{details['assessment_id']}`")
        version_col.markdown(f"**Prototype version**  \n`{details['prototype_version']}`")


def render_empty_state(title, message):
    st.markdown(
        f'<div class="empty-panel"><h3>{title}</h3><p>{message}</p></div>',
        unsafe_allow_html=True,
    )


def show_status_message(status):
    message = f"**{status['label']}** — {status['detail']}"
    if status["tone"] == "success":
        st.success(message)
    elif status["tone"] == "error":
        st.error(message)
    elif status["tone"] == "warning":
        st.warning(message)
    else:
        st.info(message)


def render_case_intelligence():
    result = st.session_state.result
    if not result:
        return
    with st.container(border=True):
        st.markdown('<div class="step-label">02 / CASE INTELLIGENCE</div>', unsafe_allow_html=True)
        st.subheader("Case intelligence")
        st.caption("Classification and a grounded summary based only on the supplied case information.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Complexity", result["complexity"])
        col2.metric("Urgency", result["urgency"])
        col3.metric("Sentiment", result["sentiment"])
        st.markdown(f"**Intent**  \n{result['intent']}")
        st.markdown(f"**Grounded case summary**  \n{result['summary']}")
        with st.expander("Message-only reference checklist"):
            st.write(result["missing_info"])
            st.caption(
                "This keyword checklist reads only the current customer message. It is a reference aid, "
                "not a complete case-level information check, safety check or approval gate."
            )
        assessed = st.session_state.get("assessed_inputs") or {}
        if assessed.get("case_id", "").strip() == "LIMIT-01":
            st.warning(LONG_TERM_LIMITATION)


def render_priority_route():
    priority = st.session_state.priority
    if not priority:
        return
    queue, action = recommended_route(priority)
    with st.container(border=True):
        st.markdown('<div class="step-label">03 / PRIORITY AND ROUTE</div>', unsafe_allow_html=True)
        st.subheader("Priority and recommended route")
        points_col, route_col, recommendation_col = st.columns([1, 1.2, 1])
        points_col.metric("Rule points", priority["score"])
        route_col.metric("Recommended route", "Specialist" if priority["escalation_recommended"] else "Standard")
        recommendation_col.metric("Escalation", "Yes" if priority["escalation_recommended"] else "No")
        st.caption(f"The current fixed rules have a maximum of {MAX_RULE_POINTS} rule points. This is not a percentage.")
        st.markdown(f"**Score breakdown**  \n{priority['breakdown']}")
        st.markdown(f"**Independent escalation triggers**  \n{priority['trigger']}")
        route_left, route_right = st.columns(2)
        with route_left:
            st.markdown("**Recommended queue**")
            st.write(queue)
        with route_right:
            st.markdown("**Recommended action**")
            st.write(action)
        st.caption(
            "Previous contacts, unresolved status, work impact and AI-classified sentiment contribute points. "
            "High urgency and High complexity are separate escalation conditions and do not add points."
        )


def append_audit_record(record):
    if not record or not record.get("ticket_id"):
        return
    existing_ids = {item.get("ticket_id") for item in st.session_state.audit_log}
    if record["ticket_id"] not in existing_ids:
        st.session_state.audit_log.append(dict(record))


def render_decision_panel(location):
    priority = st.session_state.priority
    details = st.session_state.processing_details
    if not priority or not details:
        return

    default_queue, default_action = recommended_route(priority)
    record = st.session_state.get("record")
    same_record = bool(record and record.get("assessment_id") == details["assessment_id"])
    if same_record:
        with st.container(border=True):
            st.markdown('<div class="step-label">04 / OFFICER REVIEW</div>', unsafe_allow_html=True)
            st.subheader("Officer review and decision")
            st.success(f"Decision recorded: {st.session_state.decision_state}")
            st.write(
                "This assessment is locked after a valid decision so the customer status, "
                "review queue and audit trail cannot diverge. Analyse a new or changed case "
                "to create another decision record."
            )
        return

    initial_queue = record["assigned_queue"] if same_record else default_queue
    initial_action = record["approved_action"] if same_record else default_action
    initial_reason = record.get("override_reason", "") if same_record else ""
    key_suffix = details["assessment_id"]

    with st.container(border=True):
        st.markdown('<div class="step-label">04 / OFFICER REVIEW</div>', unsafe_allow_html=True)
        st.subheader("Officer review and decision")
        st.write("A service officer retains final authority and can accept, revise or override the recommendation.")
        st.caption(
            "Accept keeps the recommendation unchanged. Modify requires a real change and a reason. "
            "Escalate manually is used when the officer chooses a specialist route."
        )
        queue = st.text_input("Final assigned queue", value=initial_queue, key=f"queue_{key_suffix}")
        action_note = st.text_area("Final action / case notes", value=initial_action, key=f"action_{key_suffix}", height=110)
        override_reason = st.text_area(
            "Officer reason (required for a modification or changed escalation route)",
            value=initial_reason,
            key=f"reason_{key_suffix}",
            height=90,
        )

        button_cols = st.columns(4)
        accept_clicked = button_cols[0].button("Accept recommendation", type="primary", key=f"accept_{key_suffix}")
        modify_clicked = button_cols[1].button("Modify recommendation", key=f"modify_{key_suffix}")
        escalate_clicked = button_cols[2].button(
            "Escalate manually",
            key=f"escalate_{key_suffix}",
            disabled=priority["escalation_recommended"],
            help=(
                "The recommendation already uses the specialist route; use Accept recommendation "
                "or Modify recommendation."
                if priority["escalation_recommended"] else
                "Override the Standard route and send the case for specialist review."
            ),
        )
        no_decision_clicked = button_cols[3].button("No decision", key=f"none_{key_suffix}")

        if no_decision_clicked:
            st.session_state.record = None
            st.session_state.decision_state = "Assessment blocked"
            st.warning("Assessment blocked — no officer decision was recorded. No simulated record created.")

        decision = (
            "Accept" if accept_clicked else
            "Modify" if modify_clicked else
            "Escalate" if escalate_clicked else
            None
        )
        if decision:
            prepared = validate_officer_decision(
                priority, decision, queue, action_note, override_reason,
            )
            if not prepared["valid"]:
                st.session_state.record = None
                st.session_state.decision_state = "Assessment blocked"
                st.error(f"{prepared['error']} No simulated record created.")
                return
            new_record = create_simulated_record(
                st.session_state.assessed_inputs["case_id"],
                priority["score"],
                prepared["final_queue"],
                prepared["decision"],
                prepared["final_action"],
                original_queue=prepared["original_queue"],
                original_action=prepared["original_action"],
                override_reason=prepared["override_reason"],
                assessment_id=details["assessment_id"],
            )
            if new_record["status"].startswith("BLOCKED"):
                st.session_state.record = None
                st.session_state.decision_state = "Assessment blocked"
                st.error(f"{new_record['status']}. No simulated record created.")
            else:
                st.session_state.record = new_record
                st.session_state.decision_state = {
                    "Accept": "Recommendation accepted",
                    "Modify": "Recommendation modified",
                    "Escalate": "Escalated for specialist review",
                }[decision]
                append_audit_record(new_record)
                st.success(st.session_state.decision_state)


def render_record_card():
    record = st.session_state.get("record")
    status = officer_final_status(st.session_state)
    show_status_message(status)
    if not record or record.get("status") != "Created for simulation":
        return

    with st.container(border=True):
        st.markdown('<div class="step-label">05 / SIMULATED CASE RECORD</div>', unsafe_allow_html=True)
        st.subheader("Approved action record")
        st.success("SIMULATED RECORD — DEMONSTRATION ONLY")
        case_col, assessment_col, ticket_col = st.columns([1, 1.6, 1.4])
        case_col.metric("Case ID", record["case_id"])
        assessment_col.markdown(f"**Assessment ID**  \n`{record['assessment_id']}`")
        ticket_col.markdown(f"**Ticket ID**  \n`{record['ticket_id']}`")
        decision_col, points_col, time_col = st.columns(3)
        decision_col.metric("Officer decision", record["human_decision"])
        points_col.metric("Rule points", record["priority_score"])
        time_col.markdown(f"**Timestamp (SGT)**  \n{record['timestamp']}")
        original_col, final_col = st.columns(2)
        with original_col:
            st.markdown("#### Original recommendation")
            st.markdown(f"**Queue**  \n{record['original_queue']}")
            st.markdown(f"**Action**  \n{record['original_action']}")
        with final_col:
            st.markdown("#### Final officer decision")
            st.markdown(f"**Queue**  \n{record['assigned_queue']}")
            st.markdown(f"**Action**  \n{record['approved_action']}")
        st.markdown(f"**Override reason**  \n{record['override_reason'] or 'Not required — the recommendation was accepted unchanged.'}")
        st.caption(
            f"Score unit: {record['score_unit']} · Maximum: {record['max_rule_points']} · "
            f"Prototype version: {record['prototype_version']}"
        )
        with st.expander("Technical record fields"):
            st.json(record)
        st.caption("Generated by this prototype. It does not exist in a live Singtel system.")


def render_customer_workspace():
    render_page_header(
        "01", "CUSTOMER WORKSPACE", "Home broadband support",
        "Report a service issue and follow the status of the support request.",
        "CUSTOMER VIEW",
    )
    st.info("Simulated customer journey — no account access, service change or live support request.")
    form_col, activity_col = st.columns([0.92, 1.08], gap="large")
    with form_col:
        with st.container(border=True):
            st.caption("CUSTOMER VIEW · SUPPORT JOURNEY")
            st.subheader("Tell us what is happening")
            st.write("Home Broadband")
            with st.form("customer_support_form", clear_on_submit=False):
                issue_type = st.selectbox(
                    "What do you need help with?",
                    ["Intermittent connection", "No connection", "Slow connection", "Wi-Fi setup or password", "Other"],
                )
                service_status = st.selectbox(
                    "What is the service status now?",
                    ["The issue is still happening", "The service is working now"],
                )
                duration = st.selectbox(
                    "How long has this been happening?",
                    ["Less than one hour", "Today", "2–3 days", "More than one week", "A long time", "Not sure"],
                )
                scope = st.selectbox(
                    "What is affected?",
                    ["All devices", "One device", "Some devices", "Not sure"],
                )
                previous_contacts = st.number_input(
                    "How many times have you contacted support about this issue?",
                    min_value=0,
                    value=0,
                )
                work_impact = st.checkbox("This is affecting work or another critical activity")
                customer_message = st.text_area(
                    "Tell us more",
                    placeholder="Describe what happened and any steps you have already tried.",
                    height=130,
                )
                customer_submit = st.form_submit_button("Submit support request", type="primary", use_container_width=True)

            if customer_submit:
                if not has_meaningful_content(customer_message):
                    st.error("Please add a little more detail before submitting the request.")
                else:
                    case_reference = "WEB-" + datetime.now(SGT).strftime("%Y%m%d") + "-" + "".join(
                        random.choices(string.ascii_uppercase + string.digits, k=6)
                    )
                    structured_message = (
                        f"Issue type: {issue_type}. Duration: {duration}. Scope: {scope}. "
                        f"Current status: {service_status}. Customer message: {customer_message.strip()}"
                    )
                    clear_assessment(st.session_state)
                    st.session_state.case_draft = {
                        "case_id": case_reference,
                        "prior_contacts": int(previous_contacts),
                        "unresolved": service_status == "The issue is still happening",
                        "work_impact": bool(work_impact),
                        "history_text": "",
                        "case_text": structured_message,
                    }
                    st.session_state.case_revision += 1
                    st.session_state.loaded_scenario = "Customer-submitted request"
                    st.session_state.assessment_invalidated = False
                    st.session_state.assessment_error = None
                    st.session_state.customer_request = {
                        "case_id": case_reference,
                        "submitted_at": datetime.now(SGT).strftime("%Y-%m-%d %H:%M:%S"),
                        "service": "Home Broadband",
                        "issue_type": issue_type,
                        "message": customer_message.strip(),
                    }
                    st.success(f"Request received. Your case reference is {case_reference}.")
            st.caption("Your message becomes available to a service officer in this browser session.")

    with activity_col:
        with st.container(border=True):
            st.caption("YOUR SUPPORT REQUEST · THIS BROWSER")
            st.subheader("Recent activity")
            request = st.session_state.customer_request
            if not request:
                render_empty_state(
                    "No support requests yet",
                    "Submit the form to receive a case reference and follow the latest status here.",
                )
            else:
                status = customer_status_from_state(st.session_state)
                st.markdown(f"**Case reference**  \n`{request['case_id']}`")
                st.caption(f"Submitted {request['submitted_at']} SGT · {request['service']}")
                st.markdown(f'<div class="customer-status"><span>{status["stage"]}</span><h3>{status["title"]}</h3><p>{status["message"]}</p></div>', unsafe_allow_html=True)
                if status.get("ticket_id"):
                    st.markdown(f"**Support record**  \n`{status['ticket_id']}`")
                stage_order = {
                    "Request received": 25,
                    "Officer review": 55,
                    "Under review": 65,
                    "Specialist review arranged": 100,
                    "Support action confirmed": 100,
                }
                st.progress(stage_order.get(status["stage"], 10) / 100)
                st.markdown("**What happens next**")
                st.write(
                    "Keep the case reference. A confirmed action appears here after a service officer reviews the request."
                )
            st.divider()
            st.caption(
                "This classroom prototype stores simulated requests only for the current app session. "
                "It cannot access a Singtel account or create a real support case."
            )


def render_officer_workspace(api_key):
    render_page_header(
        "02", "SERVICE OFFICER WORKSPACE", "Complex case intelligence",
        "Build the case context, review the recommendation and record the next action.",
        "STAFF FRONTEND",
    )
    st.caption(f"Current input source: {st.session_state.loaded_scenario}")
    scenario_col, load_col = st.columns([4, 1])
    with scenario_col:
        scenario_name = st.selectbox(
            "Load a synthetic validation case",
            list(CASE_SCENARIOS.keys()),
            index=None,
            placeholder="Choose a case template",
            help="Templates prefill inputs only. Outputs are produced by the live model and fixed Python rules.",
        )
    with load_col:
        st.write("")
        st.write("")
        load_case = st.button("Load case", use_container_width=True, disabled=scenario_name is None)
    if load_case:
        st.session_state.case_draft = dict(CASE_SCENARIOS[scenario_name])
        st.session_state.case_revision += 1
        st.session_state.loaded_scenario = scenario_name
        clear_assessment(st.session_state)
        st.session_state.assessment_invalidated = False
        st.session_state.assessment_error = None
        st.session_state.last_attempt_inputs = None
        st.rerun()

    draft = st.session_state.case_draft
    revision = st.session_state.case_revision
    with st.container(border=True):
        st.markdown('<div class="step-label">01 / CASE INPUT</div>', unsafe_allow_html=True)
        current_trace = st.session_state.get("processing_details")
        if current_trace:
            trace_case, trace_assessment, trace_version = st.columns([1, 2, 2])
            trace_case.caption(f"Case ID · {current_trace['case_id']}")
            trace_assessment.caption(f"Assessment ID · {current_trace['assessment_id']}")
            trace_version.caption(f"Version · {current_trace['prototype_version']}")
        st.subheader("Build the case context")
        st.caption("Confirm that the structured fields, interaction history and current message describe the same case.")
        case_col, contacts_col = st.columns(2)
        case_id = case_col.text_input("Case ID", value=draft["case_id"], key=f"staff_case_id_{revision}")
        prior_contacts = contacts_col.number_input(
            "Previous support contacts", min_value=0, value=int(draft["prior_contacts"]), key=f"staff_contacts_{revision}"
        )
        unresolved_col, impact_col = st.columns(2)
        unresolved = unresolved_col.checkbox(
            "Issue still unresolved", value=bool(draft["unresolved"]), key=f"staff_unresolved_{revision}"
        )
        work_impact = impact_col.checkbox(
            "Affects work or a critical activity", value=bool(draft["work_impact"]), key=f"staff_impact_{revision}"
        )
        history_text = st.text_area(
            "Previous interaction history",
            value=draft["history_text"],
            height=180,
            key=f"staff_history_{revision}",
            help="Most recent note last. Keep the count consistent with Previous support contacts.",
        )
        case_text = st.text_area(
            "Current customer message", value=draft["case_text"], height=160, key=f"staff_message_{revision}"
        )
        analyse_clicked = st.button("Analyse case", type="primary", key=f"analyse_{revision}")

    current_inputs = {
        "case_id": case_id,
        "prior_contacts": int(prior_contacts),
        "unresolved": bool(unresolved),
        "work_impact": bool(work_impact),
        "history_text": history_text,
        "case_text": case_text,
    }
    st.session_state.case_draft = dict(current_inputs)
    inputs_changed = invalidate_changed_inputs(st.session_state, current_inputs)
    if inputs_changed:
        st.session_state.assessment_error = None
        st.rerun()
    if (st.session_state.assessment_error
            and st.session_state.last_attempt_inputs != current_inputs):
        st.session_state.assessment_error = None
        st.session_state.validation_event = None
        st.session_state.decision_state = "Awaiting analysis"

    if analyse_clicked:
        st.session_state.last_attempt_inputs = dict(current_inputs)
        with st.spinner("Validating the case and preparing the recommendation..."):
            error = run_assessment(st.session_state, current_inputs, api_key)
        st.session_state.assessment_error = error
        if error:
            st.session_state.decision_state = "Assessment blocked"
        else:
            st.rerun()

    if st.session_state.assessment_error:
        st.error(st.session_state.assessment_error)
    elif st.session_state.get("assessment_invalidated"):
        st.warning("Inputs changed. Previous case intelligence, processing evidence and simulated record were cleared. Analyse the case again.")

    if st.session_state.result:
        render_trace_strip()
        render_case_intelligence()
        render_priority_route()
        render_decision_panel("officer")
        render_record_card()
    else:
        status = officer_final_status(st.session_state)
        show_status_message(status)


def render_review_queue():
    render_page_header(
        "03", "HUMAN OVERSIGHT", "Review queue",
        "Inspect the full case dossier before confirming or overriding the recommendation.",
        "STAFF REVIEW",
    )
    st.info("In-session simulated review queue — records are not stored after the application session ends.")
    if not st.session_state.result:
        if st.session_state.get("assessment_error"):
            st.error(f"Assessment blocked — {st.session_state.assessment_error}")
        event = st.session_state.get("validation_event")
        if event:
            with st.container(border=True):
                st.caption("BLOCKED INPUT REVIEW")
                case_col, status_col = st.columns(2)
                case_col.markdown(f"**Case ID**  \n`{event['case_id']}`")
                status_col.markdown(f"**Validation result**  \n{event['status']}")
                st.markdown(f"**Reason**  \n{event['message']}")
                attempted = event["attempted_inputs"]
                st.write(
                    f"Previous contacts: {attempted['prior_contacts']} · "
                    f"Unresolved: {'Yes' if attempted['unresolved'] else 'No'} · "
                    f"Work impact: {'Yes' if attempted['work_impact'] else 'No'}"
                )
                st.caption(
                    f"{event['validation_id']} · {event['timestamp_sgt']} · {event['prototype_version']} · "
                    "No model output, score or simulated record was created."
                )
        render_empty_state(
            "No assessed case in the queue",
            "Analyse a valid case in Service officer workspace. Invalid or changed inputs do not remain in this queue.",
        )
        return

    render_trace_strip()
    inputs = st.session_state.assessed_inputs
    result = st.session_state.result
    priority = st.session_state.priority
    details = st.session_state.processing_details
    queue, action = recommended_route(priority)
    notes = extract_operational_notes(inputs["history_text"])

    with st.container(border=True):
        st.caption("CASE DOSSIER")
        id_col, contacts_col, status_col, impact_col = st.columns(4)
        id_col.metric("Case ID", inputs["case_id"])
        contacts_col.metric("Previous contacts", inputs["prior_contacts"])
        status_col.metric("Unresolved", "Yes" if inputs["unresolved"] else "No")
        impact_col.metric("Work impact", "Yes" if inputs["work_impact"] else "No")
        message_col, history_col = st.columns(2)
        with message_col:
            st.markdown("**Current customer message**")
            st.write(inputs["case_text"])
        with history_col:
            st.markdown("**Previous interaction history**")
            st.write(inputs["history_text"] or "No history supplied.")
        operations_left, operations_right = st.columns(2)
        with operations_left:
            st.markdown("**Previous troubleshooting**")
            st.write(notes["troubleshooting"])
        with operations_right:
            st.markdown("**Pending action or scheduled follow-up**")
            st.write(notes["pending_action"])
        st.caption("Operational note fields are transparent reference extractions from the supplied history and do not affect scoring.")

    with st.container(border=True):
        st.caption("CASE INTELLIGENCE AND ROUTING")
        complexity_col, urgency_col, sentiment_col, points_col = st.columns(4)
        complexity_col.metric("Complexity", result["complexity"])
        urgency_col.metric("Urgency", result["urgency"])
        sentiment_col.metric("Sentiment", result["sentiment"])
        points_col.metric("Rule points", priority["score"])
        st.markdown(f"**Intent**  \n{result['intent']}")
        st.markdown(f"**Grounded summary**  \n{result['summary']}")
        route_col, action_col = st.columns(2)
        with route_col:
            st.markdown(f"**Recommended queue**  \n{queue}")
            st.markdown(f"**Escalation triggers**  \n{priority['trigger']}")
        with action_col:
            st.markdown(f"**Recommended action**  \n{action}")
            st.markdown(f"**Score breakdown**  \n{priority['breakdown']}")
        st.caption(
            f"Model: {details['returned_model'] or details['requested_model']} · "
            f"Version: {details['prototype_version']} · Completed: {details['completed_at_sgt']}"
        )
        if result["missing_info"] != "None":
            st.warning(
                "Message-only reference checklist: " + result["missing_info"] +
                ". Review the full dossier before requesting more information."
            )
        else:
            st.success("No keywords are flagged by the message-only reference checklist.")
        if inputs.get("case_id", "").strip() == "LIMIT-01":
            st.warning(LONG_TERM_LIMITATION)

    render_decision_panel("queue")
    render_record_card()


def render_processing_workspace(api_key, shared_key):
    render_page_header(
        "04", "PROCESSING & AUDIT WORKSPACE", "Processing evidence",
        "Trace validated input, the OpenAI response, independent Python rules and the officer decision.",
        "PROTOTYPE BACKEND",
    )
    st.warning(
        "Prototype processing and audit view — not a separately authenticated production administration system."
    )
    if shared_key:
        st.success("Server-side processing is configured. The API credential is not displayed.")
    elif api_key:
        st.info("A session-only developer key is configured. It is not included in processing evidence.")
    else:
        st.info("Open Prototype settings in the sidebar to configure processing for this session.")

    details = st.session_state.get("processing_details")
    if not details:
        if st.session_state.get("assessment_error"):
            st.error(st.session_state.assessment_error)
            if "No AI call was made" in st.session_state.assessment_error:
                st.info("No AI call was made. Correct the case information before continuing.")
        event = st.session_state.get("validation_event")
        if event:
            with st.container(border=True):
                st.subheader("Pre-call validation record")
                st.markdown(f"**Validation ID**  \n`{event['validation_id']}`")
                st.markdown(f"**Case ID**  \n`{event['case_id']}`")
                st.markdown(f"**Result**  \n{event['status']}")
                st.markdown(f"**Reason**  \n{event['message']}")
                st.caption(f"{event['timestamp_sgt']} · {event['prototype_version']}")
        render_empty_state(
            "No current processing record",
            "Analyse a valid case in Service officer workspace. Pre-call validation failures intentionally produce no OpenAI response record.",
        )
        return

    render_trace_strip()
    context_tab, response_tab, rules_tab, decision_tab = st.tabs([
        "Validated case context",
        "OpenAI response",
        "Python rules and routing",
        "Decision audit record",
    ])

    with context_tab:
        with st.container(border=True):
            st.subheader("Pre-call validation")
            st.success("Case ID, narrative content and detected contact-count consistency checks passed before the API call.")
            st.caption(
                "The contact-count check recognises explicit contact/support/call wording. It does not treat ‘restarted the router twice’ as two support contacts."
            )
            meta_left, meta_right = st.columns(2)
            meta_left.markdown(f"**Assessment ID**  \n`{details['assessment_id']}`")
            meta_left.markdown(f"**Case ID**  \n`{details['case_id']}`")
            meta_right.markdown(f"**Prototype version**  \n`{details['prototype_version']}`")
            meta_right.markdown("**Validation result**  \nPassed before API call")
        with st.container(border=True):
            st.subheader("Exact case context sent to the API")
            st.code(details["case_context"], language="text")
            with st.expander("System instructions used for this request"):
                st.code(details["system_prompt"], language="text")

    with response_tab:
        with st.container(border=True):
            st.subheader("OpenAI processing record")
            requested_col, returned_col, time_col = st.columns(3)
            requested_col.metric("Requested model", details["requested_model"])
            returned_col.metric("Returned model", details["returned_model"] or "Unavailable")
            time_col.metric("API round-trip", f"{details['elapsed_seconds']:.3f} s")
            st.markdown(f"**OpenAI response ID**  \n`{details['response_id'] or 'Unavailable'}`")
            time_left, time_right = st.columns(2)
            time_left.markdown(f"**Processing started (SGT)**  \n{details['started_at_sgt']}")
            time_right.markdown(f"**Processing completed (SGT)**  \n{details['completed_at_sgt']}")
            st.markdown("**Structured classification and summary returned by the API**")
            st.json(details["model_output"])
            with st.expander("Original API response text"):
                st.code(details["raw_response"] or "Unavailable", language="json")
            st.caption("No hidden model reasoning or chain of thought is requested or displayed.")

    with rules_tab:
        inputs = st.session_state.assessed_inputs
        result = st.session_state.result
        priority = st.session_state.priority
        queue, action = recommended_route(priority)
        with st.container(border=True):
            st.subheader("Fixed Python score")
            st.table([
                {"Factor": "Base", "Observed value": "Always", "Rule points": 10},
                {"Factor": "Previous contacts ≥ 2", "Observed value": inputs["prior_contacts"], "Rule points": 20 if inputs["prior_contacts"] >= 2 else 0},
                {"Factor": "Issue unresolved", "Observed value": "Yes" if inputs["unresolved"] else "No", "Rule points": 20 if inputs["unresolved"] else 0},
                {"Factor": "Work / critical activity impact", "Observed value": "Yes" if inputs["work_impact"] else "No", "Rule points": 25 if inputs["work_impact"] else 0},
                {"Factor": "Frustrated sentiment", "Observed value": result["sentiment"], "Rule points": 12 if result["sentiment"] == "Frustrated" else 0},
            ])
            points_col, max_col = st.columns(2)
            points_col.metric("Total rule points", priority["score"])
            max_col.metric("Maximum rule points", MAX_RULE_POINTS)
        with st.container(border=True):
            st.subheader("Independent routing result")
            st.markdown(f"**Escalation recommended**  \n{'Yes' if priority['escalation_recommended'] else 'No'}")
            st.markdown(f"**Triggers**  \n{priority['trigger']}")
            route_col, action_col = st.columns(2)
            route_col.markdown(f"**Recommended queue**  \n{queue}")
            action_col.markdown(f"**Recommended action**  \n{action}")
            st.caption("High urgency and High complexity can trigger escalation without contributing rule points.")

    with decision_tab:
        if st.session_state.record:
            render_record_card()
        else:
            status = officer_final_status(st.session_state)
            show_status_message(status)
            render_empty_state(
                "No final simulated record",
                "A valid Accept, Modify or Escalate manually decision is required before a ticket ID is generated.",
            )


def render_audit_trail():
    render_page_header(
        "05", "SESSION AUDIT", "Audit trail",
        "Review simulated decisions created during this app session.",
        "STAFF EVIDENCE",
    )
    st.info("In-session audit only — entries are not stored after the application session ends.")
    records = st.session_state.audit_log
    if not records:
        render_empty_state(
            "No approved actions in this session",
            "A record appears after an officer makes a valid Accept, Modify or Escalate manually decision.",
        )
        return

    latest = records[-1]
    st.caption(f"{len(records)} valid simulated decision record(s) in this session")
    st.table([
        {
            "Case ID": item["case_id"],
            "Assessment ID": item["assessment_id"],
            "Ticket ID": item["ticket_id"],
            "Decision": item["human_decision"],
            "Rule points": item["priority_score"],
            "Timestamp (SGT)": item["timestamp"],
        }
        for item in reversed(records)
    ])
    with st.container(border=True):
        st.subheader("Latest decision detail")
        original_col, final_col = st.columns(2)
        with original_col:
            st.markdown("**Original recommendation**")
            st.write(latest["original_queue"])
            st.write(latest["original_action"])
        with final_col:
            st.markdown("**Final officer decision**")
            st.write(latest["assigned_queue"])
            st.write(latest["approved_action"])
        st.markdown(f"**Override reason**  \n{latest['override_reason'] or 'Not required.'}")
        st.download_button(
            "Download latest simulated audit record",
            data=json.dumps(latest, indent=2),
            file_name=f"{latest['ticket_id']}.json",
            mime="application/json",
        )


initialise_ui_state()

st.markdown(
    """
    <style>
    .stApp { background:#F4F6F9; color:#192333; }
    .block-container { max-width:1240px; padding-top:1.5rem; padding-bottom:3rem; }
    [data-testid="stSidebar"] { background:#172333; border-right:0; min-width:285px; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span { color:#D5DDE8; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap:8px; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding:13px 14px; border-radius:11px; border-left:4px solid transparent;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background:#2B3545; border-left-color:#E31842;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color:#FFFFFF; font-weight:700; }
    [data-testid="stSidebar"] hr { border-color:#3B4657; }
    [data-testid="stSidebar"] input { color:#192333 !important; }
    h1,h2,h3,h4 { color:#192333 !important; letter-spacing:-.025em; }
    .sidebar-brand { padding:12px 8px 28px; }
    .brand-mark { width:36px; height:36px; border-radius:10px; background:#E31842; display:flex; align-items:center; justify-content:center; color:#FFF; font-weight:800; margin-bottom:12px; }
    .sidebar-brand h2 { color:#FFF !important; margin:0; font-size:23px; line-height:1.05; }
    .sidebar-brand p { color:#9CAABC !important; margin:8px 0 0; font-size:12px; letter-spacing:.13em; font-weight:700; }
    .side-note { border:1px solid #3B4657; border-radius:14px; padding:14px; margin-top:24px; color:#C6D0DC; font-size:13px; }
    .page-hero { background:#172333; border-radius:18px; padding:28px 32px; margin-bottom:26px; border-left:6px solid #E31842; box-shadow:0 12px 30px rgba(23,35,51,.10); }
    .hero-topline { display:flex; justify-content:space-between; align-items:center; color:#B8C3D2; font-size:12px; font-weight:800; letter-spacing:.14em; }
    .role-badge { border:1px solid #596576; border-radius:999px; padding:6px 11px; color:#FFF; }
    .page-hero h1 { color:#FFF !important; margin:16px 0 8px; font-size:36px; }
    .page-hero p { color:#D4DEEB; font-size:16px; margin:0; max-width:800px; }
    .red-dot { color:#E31842; }
    .hero-tags { display:flex; flex-wrap:wrap; gap:8px; margin-top:20px; }
    .hero-tags span { border:1px solid #596576; border-radius:999px; color:#F3F6FA; padding:5px 11px; font-size:11px; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:#FFF; border-radius:16px; box-shadow:0 8px 24px rgba(23,35,51,.04); }
    [data-testid="stMetric"] { background:#F3F5F8; padding:14px 16px; border-radius:12px; border:1px solid #E6EAF0; }
    [data-testid="stMetricValue"] { font-size:1.65rem; }
    .stButton > button, .stDownloadButton > button { border-radius:9px; min-height:42px; font-weight:650; }
    .stButton > button[kind="primary"] { background:#C4102C; border-color:#C4102C; color:#FFF; }
    [data-baseweb="tab-list"] { gap:18px; }
    [data-baseweb="tab-highlight"] { background:#C4102C; }
    .step-label { color:#B4102A; font-weight:800; font-size:12px; letter-spacing:.13em; margin:2px 0 4px; }
    .empty-panel { border:1px dashed #B9C4D2; background:#F8FAFC; border-radius:14px; padding:28px; margin:12px 0; text-align:center; }
    .empty-panel h3 { margin:0 0 8px; }
    .empty-panel p { color:#647287; margin:0; }
    .customer-status { background:#F7F9FC; border-left:5px solid #E31842; border-radius:12px; padding:20px; margin:18px 0; }
    .customer-status span { color:#B4102A; font-weight:800; font-size:12px; letter-spacing:.08em; }
    .customer-status h3 { margin:8px 0; }
    .customer-status p { color:#526176; margin:0; }
    @media(max-width:760px) {
        .block-container { padding-top:.8rem; }
        .page-hero { padding:22px 20px; }
        .page-hero h1 { font-size:28px; }
        .hero-topline { align-items:flex-start; gap:12px; }
        [data-testid="stSidebar"] { min-width:240px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    shared_key = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    shared_key = None


def save_api_key():
    st.session_state.api_key_cache = st.session_state.get("api_key_widget", "").strip()


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="brand-mark">S</div>
          <h2>Singtel Case<br>Intelligence</h2>
          <p>ISYS3482 · SERVICE PROTOTYPE</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    workspace = st.radio("WORKSPACE", WORKSPACES, label_visibility="visible", key="workspace_page")
    st.markdown('<div class="side-note"><b>Teaching simulation</b><br><br>Synthetic data only.<br>No live Singtel connection.</div>', unsafe_allow_html=True)
    if workspace == "04 · Processing & audit":
        with st.expander("Prototype settings"):
            st.caption(f"Version: {APP_VERSION}")
            if shared_key:
                st.success("Server-side API access configured")
            else:
                st.text_input(
                    "Developer OpenAI API key",
                    type="password",
                    key="api_key_widget",
                    on_change=save_api_key,
                    help="Session only. For public deployment, configure a server-side secret.",
                )

api_key = shared_key or st.session_state.get("api_key_cache", "")

if workspace == "01 · Customer support":
    render_customer_workspace()
elif workspace == "02 · Service officer":
    render_officer_workspace(api_key)
elif workspace == "03 · Review queue":
    render_review_queue()
elif workspace == "04 · Processing & audit":
    render_processing_workspace(api_key, shared_key)
else:
    render_audit_trail()

# End the current Streamlit run after rendering the selected workspace.
st.stop()
