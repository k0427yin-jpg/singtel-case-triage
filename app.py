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

st.set_page_config(
    page_title="Singtel Case Triage (Prototype)",
    page_icon="\U0001F4E1",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SGT = timezone(timedelta(hours=8))
APP_VERSION = "2026-09-26-provenance-lock-v10"
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
- If a contact count is identified as customer-reported, state that provenance
  explicitly and do not imply that enterprise case history was retrieved.
- For a customer-reported count, use attribution such as "The customer reports
  contacting support three times." Do not rewrite it as the unqualified fact
  "They have contacted support three times."
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

CUSTOMER_REPORTED_CONTACT_SOURCE = "Customer-reported in Customer view demo"
SYNTHETIC_TEMPLATE_CONTACT_SOURCE = "Synthetic validation template"
OFFICER_ENTERED_CONTACT_SOURCE = "Officer-entered structured field"


def create_simulated_record(case_id: str, priority_score: int, assigned_queue: str,
                             officer_decision: str, approved_action: str,
                             original_queue: str = "", original_action: str = "",
                             override_reason: str = "", assessment_id: str = "",
                             contact_count_source: str = OFFICER_ENTERED_CONTACT_SOURCE,
                             contact_count_verified: bool = True,
                             contact_count_acknowledged: bool = True) -> dict:
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
        "contact_count_source": contact_count_source,
        "contact_count_verified": bool(contact_count_verified),
        "contact_count_acknowledged": bool(contact_count_acknowledged),
        "ticket_id": ticket_id,
        "timestamp": timestamp,
    }


def clear_assessment(state):
    """Discard all outputs tied to a previous assessment, including its ticket."""
    for key in ("result", "priority", "record", "assessed_inputs", "discrepancy",
                "processing_details", "decision_draft", "validation_event"):
        state[key] = None
    state["decision_state"] = "Awaiting analysis"


def contact_count_provenance(inputs):
    """Describe where the structured contact count came from and its verification state."""
    source = inputs.get("contact_count_source") or OFFICER_ENTERED_CONTACT_SOURCE
    default_verified = source != CUSTOMER_REPORTED_CONTACT_SOURCE
    verified = bool(inputs.get("contact_count_verified", default_verified))
    acknowledged = bool(inputs.get("contact_count_acknowledged", source != CUSTOMER_REPORTED_CONTACT_SOURCE))
    return {
        "source": source,
        "verified": verified,
        "acknowledged": acknowledged,
        "customer_reported": source == CUSTOMER_REPORTED_CONTACT_SOURCE,
        "requires_acknowledgement": (
            source == CUSTOMER_REPORTED_CONTACT_SOURCE
            and not acknowledged
        ),
    }


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
    provenance = contact_count_provenance(inputs)
    if provenance["requires_acknowledgement"]:
        return block_before_api(
            "The previous-contact count was reported by the customer and is not verified "
            "against enterprise case history. A service officer must acknowledge this "
            "data limitation before analysis. No AI call was made."
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
    if provenance["customer_reported"] and provenance["acknowledged"]:
        verification_label = "Not verified against enterprise case history; source acknowledged by officer"
    elif provenance["customer_reported"]:
        verification_label = "Not verified against enterprise case history; customer-reported source"
    else:
        verification_label = "Supplied synthetic or officer-entered field"
    ai_case_input = f"""Service: Home Broadband
Previous support contacts: {inputs['prior_contacts']}
Contact-count source: {provenance['source']}
Contact-count verification: {verification_label}
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
        "contact_count_source": provenance["source"],
        "contact_count_verified": provenance["verified"],
        "contact_count_acknowledged": provenance["acknowledged"],
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


def latest_record_for_case(state, case_id):
    """Return the latest valid simulated decision for one case."""
    current = state.get("record")
    if (current and current.get("case_id") == case_id
            and current.get("status") == "Created for simulation"):
        return current
    for record in reversed(state.get("audit_log", [])):
        if (record.get("case_id") == case_id
                and record.get("status") == "Created for simulation"):
            return record
    return None


def customer_status_from_record(record):
    """Translate a final internal decision into conservative customer-safe copy."""
    specialist = "specialist" in record.get("assigned_queue", "").lower()
    decision = record.get("human_decision", "").lower()
    action = record.get("approved_action", "").lower()
    pending_confirmation = any(
        term in action for term in ("confirm", "pending", "whether", "supervisor review")
    )
    if specialist and (decision == "modify" or pending_confirmation):
        return {
            "stage": "Next step pending confirmation",
            "title": "The officer review is complete",
            "message": (
                "A service officer updated the proposed next step. Further specialist review is "
                "pending confirmation in this demonstration; no engineer visit or live service "
                "action has been arranged."
            ),
            "tone": "info",
            "ticket_id": record.get("ticket_id"),
            "progress": 100,
            "progress_label": "Case intake and decision stage complete",
            "next_step": (
                "The relevant service team would still need to confirm and carry out the approved "
                "next step in a real workflow."
            ),
        }
    if specialist:
        return {
            "stage": "Further review pending",
            "title": "The officer review is complete",
            "message": (
                "A service officer recorded a specialist route in this demonstration. The next team "
                "still needs to confirm and carry out the review, and the service issue is not assumed resolved."
            ),
            "tone": "info",
            "ticket_id": record.get("ticket_id"),
            "progress": 100,
            "progress_label": "Case intake and decision stage complete",
            "next_step": "A specialist team would confirm the follow-up action in a real service workflow.",
        }
    return {
        "stage": "Next step recorded",
        "title": "The officer review is complete",
        "message": (
            "A service officer recorded a proposed support action in this demonstration. "
            "The action has not been carried out, and the service issue is not assumed resolved."
        ),
        "tone": "info",
        "ticket_id": record.get("ticket_id"),
        "progress": 100,
        "progress_label": "Case intake and decision stage complete",
        "next_step": "The relevant service team would carry out the recorded action in a real workflow.",
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
            "progress": 0,
            "progress_label": "Request not submitted",
            "next_step": "Submit the form to start this demonstration journey.",
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
            "progress": 65,
            "progress_label": "Case intake complete · decision still pending",
            "next_step": "A service officer must record a valid decision before a next step appears here.",
        }

    if same_record_case and record.get("status") == "Created for simulation":
        return customer_status_from_record(record)

    if state.get("result") and same_assessed_case:
        return {
            "stage": "Officer review",
            "title": "Your request is being reviewed",
            "message": "A service officer is reviewing the case before confirming the next action.",
            "tone": "info",
            "progress": 55,
            "progress_label": "Case intake complete · officer decision pending",
            "next_step": "Wait for the officer decision stage to be completed in this demonstration.",
        }

    historic_record = latest_record_for_case(state, request_case_id)
    if historic_record:
        return customer_status_from_record(historic_record)

    return {
        "stage": "Request received",
        "title": "We have received your support request",
        "message": "Your case is waiting for service review. No outcome has been confirmed yet.",
        "tone": "info",
        "progress": 25,
        "progress_label": "Request received · officer review pending",
        "next_step": "A service officer will review the supplied information in this demonstration.",
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

# Role-separated interface. Only the selected workspace is rendered, so the
# customer journey, officer tools and technical evidence are visually and
# operationally distinct while reading the same in-session state.

WORKSPACES = [
    "01 · Customer view demo",
    "02 · Service officer",
    "03 · Case review",
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
        "workspace_page": WORKSPACES[1],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    legacy_workspace_names = {
        "01 · Customer support": "01 · Customer view demo",
        "03 · Review queue": "03 · Case review",
    }
    st.session_state.workspace_page = legacy_workspace_names.get(
        st.session_state.workspace_page, st.session_state.workspace_page,
    )
    if st.session_state.workspace_page not in WORKSPACES:
        st.session_state.workspace_page = WORKSPACES[1]


def render_page_header(number, eyebrow, title, subtitle, role):
    st.markdown(
        f"""
        <div class="page-hero">
          <div class="hero-accent"></div>
          <div class="hero-content">
            <div class="hero-topline"><span>{number} / {eyebrow}</span><span class="role-badge">{role}</span></div>
            <h1>{title}<span class="red-dot">.</span></h1>
            <p>{subtitle}</p>
            <div class="hero-tags"><span>Synthetic data only</span><span>No live Singtel connection</span><span>{APP_VERSION}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_switcher():
    """Always-visible, role-separated navigation for every app user."""
    st.markdown(
        """
        <div class="site-masthead">
          <div class="site-brand">
            <span class="site-wordmark">Singtel</span>
            <span class="site-product">Case Intelligence</span>
          </div>
          <div class="site-status"><span></span> Group 11 prototype</div>
        </div>
        <div class="demo-access-note"><b>Demo navigation</b> · Role views are shown together for assessment. Production authentication and role-based access are not implemented.</div>
        <div class="workspace-label">Choose a role view</div>
        """,
        unsafe_allow_html=True,
    )
    labels = ["Customer demo", "Service officer", "Case review", "Processing", "Audit"]
    columns = st.columns(5)
    for column, label, target in zip(columns, labels, WORKSPACES):
        if column.button(
            label,
            key=f"main_nav_{target}",
            type="primary" if st.session_state.workspace_page == target else "secondary",
            use_container_width=True,
        ):
            st.session_state.workspace_page = target
            st.rerun()
    st.markdown('<div class="nav-rule"></div>', unsafe_allow_html=True)


def render_customer_journey_steps():
    """Show the customer-facing support journey without exposing internal logic."""
    request = st.session_state.get("customer_request")
    request_case_id = request.get("case_id") if request else None
    assessed_case_id = (st.session_state.get("assessed_inputs") or {}).get("case_id")
    current_record = st.session_state.get("record")
    current_pending = bool(
        request and assessed_case_id == request_case_id
        and not (current_record and current_record.get("case_id") == request_case_id)
        and (st.session_state.get("result") or st.session_state.get("decision_state") == "Assessment blocked")
    )
    final_record = latest_record_for_case(st.session_state, request_case_id) if request else None
    active_step = 2 if current_pending else 3 if final_record else 2 if request else 1
    labels = [
        ("1", "Describe the issue", "Tell us what happened"),
        ("2", "Service review", "An officer checks the case"),
        ("3", "Follow the status", "See the confirmed next step"),
    ]
    cards = []
    for index, (number, title, detail) in enumerate(labels, start=1):
        state_class = "is-complete" if index < active_step else "is-active" if index == active_step else ""
        cards.append(
            f'<div class="journey-step {state_class}"><span>{number}</span>'
            f'<div><strong>{title}</strong><small>{detail}</small></div></div>'
        )
    st.markdown('<div class="journey-steps">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def render_staff_flow(active_step):
    """Provide a concise visual map of the staff workflow."""
    steps = [
        ("input", "Case input"),
        ("analysis", "AI analysis"),
        ("rules", "Rule result"),
        ("decision", "Human decision"),
        ("record", "Simulated record"),
    ]
    markup = []
    active_index = next((i for i, (key, _) in enumerate(steps) if key == active_step), 0)
    for index, (_, label) in enumerate(steps):
        state_class = "is-complete" if index < active_index else "is-active" if index == active_index else ""
        markup.append(f'<span class="staff-flow-step {state_class}">{index + 1}. {label}</span>')
    st.markdown('<div class="staff-flow">' + ''.join(markup) + '</div>', unsafe_allow_html=True)


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
        provenance = contact_count_provenance(st.session_state.assessed_inputs or {})
        if provenance["customer_reported"]:
            st.warning(
                "The contact-count points use a customer-reported number. The officer acknowledged its source, "
                "but this prototype did not retrieve or verify enterprise case history."
            )
        else:
            st.caption(f"Contact-count source: {provenance['source']}.")
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
                "case review and audit trail cannot diverge. Analyse a new or changed case "
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
                contact_count_source=contact_count_provenance(st.session_state.assessed_inputs)["source"],
                contact_count_verified=contact_count_provenance(st.session_state.assessed_inputs)["verified"],
                contact_count_acknowledged=contact_count_provenance(st.session_state.assessed_inputs)["acknowledged"],
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
        source = record.get("contact_count_source", OFFICER_ENTERED_CONTACT_SOURCE)
        if source == CUSTOMER_REPORTED_CONTACT_SOURCE:
            st.warning(
                "Contact count in this record was customer-reported and acknowledged by the officer, "
                "but was not verified against enterprise case history."
            )
        else:
            st.caption(f"Contact-count source: {source}.")
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
        "01", "CUSTOMER VIEW DEMO", "Home broadband support",
        "Report a service issue and follow the status of the support request.",
        "CUSTOMER VIEW DEMO",
    )
    st.markdown(
        '<div class="customer-notice"><b>Demonstration only</b><span>No account access, service change or live support request.</span></div>',
        unsafe_allow_html=True,
    )
    render_customer_journey_steps()

    with st.container(border=True):
        st.caption("CUSTOMER VIEW DEMO · HOME BROADBAND")
        st.subheader("Tell us what is happening")
        st.write("Complete the short form below. You will receive a case reference for this simulated request.")
        with st.form("customer_support_form", clear_on_submit=False):
            first_row = st.columns(2, gap="large")
            with first_row[0]:
                issue_type = st.selectbox(
                    "What do you need help with?",
                    ["Intermittent connection", "No connection", "Slow connection", "Wi-Fi setup or password", "Other"],
                )
            with first_row[1]:
                service_status = st.selectbox(
                    "What is the service status now?",
                    ["The issue is still happening", "The service is working now"],
                )
            second_row = st.columns(3, gap="large")
            with second_row[0]:
                duration = st.selectbox(
                    "How long has this been happening?",
                    ["Less than one hour", "Today", "2–3 days", "More than one week", "A long time", "Not sure"],
                )
            with second_row[1]:
                scope = st.selectbox(
                    "What is affected?",
                    ["All devices", "One device", "Some devices", "Not sure"],
                )
            with second_row[2]:
                previous_contacts = st.number_input(
                    "How many times do you recall contacting support about this issue?",
                    min_value=0,
                    value=0,
                    help="This is your own estimate. A service officer must verify it against any available case history before analysis.",
                )
            work_impact = st.checkbox("This is affecting work or another critical activity")
            customer_message = st.text_area(
                "Tell us more",
                placeholder="Describe what happened and any steps you have already tried.",
                height=130,
            )
            submit_col, note_col = st.columns([1, 2.4])
            with submit_col:
                customer_submit = st.form_submit_button("Submit support request", type="primary", use_container_width=True)
            with note_col:
                st.caption("Your message becomes available to a service officer in this browser session.")

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
                    "contact_count_source": CUSTOMER_REPORTED_CONTACT_SOURCE,
                    "contact_count_verified": False,
                    "contact_count_acknowledged": False,
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
                    "reported_contacts": int(previous_contacts),
                    "contact_count_source": CUSTOMER_REPORTED_CONTACT_SOURCE,
                }
                st.success(f"Request received. Your case reference is {case_reference}.")

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.caption("YOUR SUPPORT REQUEST · THIS BROWSER")
        st.subheader("Track your latest request")
        request = st.session_state.customer_request
        if not request:
            render_empty_state(
                "No support requests yet",
                "Submit the form above to receive a case reference and follow the latest status here.",
            )
        else:
            status = customer_status_from_state(st.session_state)
            status_meta, status_body = st.columns([0.9, 2.1], gap="large")
            with status_meta:
                st.markdown(f"**Case reference**  \n`{request['case_id']}`")
                st.caption(f"Submitted {request['submitted_at']} SGT")
                st.caption(request["service"])
                if status.get("ticket_id"):
                    st.markdown(f"**Simulated support record**  \n`{status['ticket_id']}`")
            with status_body:
                st.markdown(f'<div class="customer-status"><span>{status["stage"]}</span><h3>{status["title"]}</h3><p>{status["message"]}</p></div>', unsafe_allow_html=True)
                st.caption(status["progress_label"])
                st.progress(status["progress"] / 100)
                st.markdown("**What happens next**")
                st.write(status["next_step"])
        st.divider()
        st.caption(
            "This Group 11 prototype stores simulated requests only for the current app session. "
            "It cannot access a Singtel account or create a real support case."
        )


def render_officer_workspace(api_key):
    render_page_header(
        "02", "SERVICE OFFICER WORKSPACE", "Complex case intelligence",
        "Build the case context, review the recommendation and record the next action.",
        "STAFF FRONTEND",
    )
    officer_step = "record" if st.session_state.get("record") else "decision" if st.session_state.get("result") else "input"
    render_staff_flow(officer_step)
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
    contact_count_source = draft.get("contact_count_source") or (
        CUSTOMER_REPORTED_CONTACT_SOURCE
        if st.session_state.loaded_scenario == "Customer-submitted request"
        else SYNTHETIC_TEMPLATE_CONTACT_SOURCE
    )
    default_contact_acknowledgement = bool(
        draft.get("contact_count_acknowledged", contact_count_source != CUSTOMER_REPORTED_CONTACT_SOURCE)
    )
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
        if contact_count_source == CUSTOMER_REPORTED_CONTACT_SOURCE:
            st.warning(
                "Contact-count source: customer-reported in the demo form. It has not been retrieved or verified "
                "against Singtel case history. If used, the rules will treat it as an unverified structured input."
            )
            contact_count_acknowledged = st.checkbox(
                "I acknowledge that this contact count is customer-reported and unverified",
                value=default_contact_acknowledgement,
                key=f"staff_contacts_acknowledged_{revision}",
            )
            contact_count_verified = False
        else:
            contact_count_verified = True
            contact_count_acknowledged = True
            st.caption(f"Contact-count source: {contact_count_source}.")
        acknowledgement_required = (
            contact_count_source == CUSTOMER_REPORTED_CONTACT_SOURCE
            and not contact_count_acknowledged
        )
        analyse_clicked = st.button(
            "Analyse case",
            type="primary",
            key=f"analyse_{revision}",
            disabled=acknowledgement_required,
            help=(
                "Acknowledge the customer-reported, unverified source before analysis."
                if acknowledgement_required else
                "Validate the inputs, call the configured model and apply the fixed Python rules."
            ),
        )

    current_inputs = {
        "case_id": case_id,
        "prior_contacts": int(prior_contacts),
        "unresolved": bool(unresolved),
        "work_impact": bool(work_impact),
        "history_text": history_text,
        "case_text": case_text,
        "contact_count_source": contact_count_source,
        "contact_count_verified": bool(contact_count_verified),
        "contact_count_acknowledged": bool(contact_count_acknowledged),
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


def render_case_review():
    render_page_header(
        "03", "HUMAN OVERSIGHT", "Case review",
        "Inspect the full case dossier before confirming or overriding the recommendation.",
        "STAFF REVIEW",
    )
    render_staff_flow("decision")
    st.info(
        "Current in-session case dossier only. This prototype does not implement a selectable, sortable or persistent queue."
    )
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
            "No assessed case available for review",
            "Analyse a valid case in Service officer workspace. Invalid or changed inputs do not remain in this case view.",
        )
        return

    render_trace_strip()
    inputs = st.session_state.assessed_inputs
    result = st.session_state.result
    priority = st.session_state.priority
    details = st.session_state.processing_details
    queue, action = recommended_route(priority)
    notes = extract_operational_notes(inputs["history_text"])
    provenance = contact_count_provenance(inputs)

    with st.container(border=True):
        st.caption("CASE DOSSIER")
        id_col, contacts_col, status_col, impact_col = st.columns(4)
        id_col.metric("Case ID", inputs["case_id"])
        contacts_col.metric("Reported contacts", inputs["prior_contacts"])
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
        if provenance["customer_reported"]:
            st.warning(
                "Contact-count provenance: customer-reported in the demo form; acknowledged by the officer, "
                "but not retrieved or verified against enterprise case history."
            )
        else:
            st.caption(f"Contact-count provenance: {provenance['source']}.")

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
    render_staff_flow("analysis")
    st.warning(
        "Prototype processing and audit view — not a separately authenticated production administration system."
    )
    if not shared_key:
        with st.expander("Developer processing settings"):
            st.text_input(
                "Developer OpenAI API key",
                type="password",
                key="api_key_widget",
                on_change=save_api_key,
                help="Session only. For public deployment, configure a server-side secret.",
            )
            st.caption("The key is used only for this browser session and is never shown in processing evidence.")
    if shared_key:
        st.success("Server-side processing is configured. The API credential is not displayed.")
    elif api_key:
        st.info("A session-only developer key is configured. It is not included in processing evidence.")
    else:
        st.info("Open Developer processing settings above to configure processing for this session.")

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
            if details.get("contact_count_source") == CUSTOMER_REPORTED_CONTACT_SOURCE:
                st.warning(
                    "The contact count originated from the customer's demo submission. The officer acknowledged "
                    "the source; the prototype did not retrieve or verify enterprise case history."
                )
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
        provenance = contact_count_provenance(inputs)
        with st.container(border=True):
            st.subheader("Fixed Python score")
            st.table([
                {"Factor": "Base", "Observed value": "Always", "Rule points": 10},
                {
                    "Factor": "Previous contacts ≥ 2",
                    "Observed value": (
                        f"{inputs['prior_contacts']} · customer-reported, unverified; source acknowledged"
                        if provenance["customer_reported"] else
                        f"{inputs['prior_contacts']} · {provenance['source']}"
                    ),
                    "Rule points": 20 if inputs["prior_contacts"] >= 2 else 0,
                },
                {"Factor": "Issue unresolved", "Observed value": "Yes" if inputs["unresolved"] else "No", "Rule points": 20 if inputs["unresolved"] else 0},
                {"Factor": "Work / critical activity impact", "Observed value": "Yes" if inputs["work_impact"] else "No", "Rule points": 25 if inputs["work_impact"] else 0},
                {"Factor": "Frustrated sentiment", "Observed value": result["sentiment"], "Rule points": 12 if result["sentiment"] == "Frustrated" else 0},
            ])
            points_col, max_col = st.columns(2)
            points_col.metric("Total rule points", priority["score"])
            max_col.metric("Maximum rule points", MAX_RULE_POINTS)
            st.caption(
                "These deterministic Python rules use structured fields and AI-classified sentiment. "
                "An incorrect AI sentiment classification can therefore affect the 12-point sentiment component."
            )
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
    render_staff_flow("record")
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
        source = latest.get("contact_count_source", OFFICER_ENTERED_CONTACT_SOURCE)
        st.markdown(f"**Contact-count source**  \n{source}")
        if source == CUSTOMER_REPORTED_CONTACT_SOURCE:
            st.caption("Officer acknowledged the source; enterprise case-history verification was not performed.")
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
    :root {
        --sg-red:#E40046;
        --sg-red-dark:#C6003D;
        --sg-magenta:#C6006F;
        --sg-ink:#242429;
        --sg-muted:#666A73;
        --sg-canvas:#F7F7F8;
        --sg-blush:#FFF1F5;
        --sg-border:#E3E3E7;
    }
    .stApp { background:var(--sg-canvas); color:var(--sg-ink); }
    .block-container { max-width:1220px; padding-top:1rem; padding-bottom:4rem; }
    [data-testid="stHeader"] { background:rgba(247,247,248,.92); }
    h1,h2,h3,h4 { color:var(--sg-ink) !important; letter-spacing:-.028em; }
    p, label { line-height:1.55; }
    .site-masthead {
        display:flex; align-items:center; justify-content:space-between; gap:20px;
        padding:9px 2px 11px; border-bottom:1px solid var(--sg-border); margin-bottom:9px;
    }
    .site-brand { display:flex; align-items:baseline; gap:14px; }
    .site-wordmark { color:var(--sg-red); font-size:28px; font-weight:850; letter-spacing:-.055em; }
    .site-product { color:var(--sg-ink); font-size:16px; font-weight:700; }
    .site-status {
        display:flex; align-items:center; gap:8px; color:var(--sg-muted); font-size:12px;
        text-transform:uppercase; letter-spacing:.09em; font-weight:750;
    }
    .site-status span { width:8px; height:8px; border-radius:999px; background:var(--sg-red); }
    .demo-access-note { color:#555860; background:#FFF; border-left:3px solid var(--sg-red); padding:7px 10px; margin:0 0 10px; font-size:12px; }
    .demo-access-note b { color:#A9003B; }
    .workspace-label { color:#5F626A; font-size:12px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin-bottom:6px; }
    .nav-rule { height:1px; background:var(--sg-border); margin:8px 0 14px; }
    .page-hero {
        position:relative; overflow:hidden; background:linear-gradient(112deg,#FFFFFF 0%,#FFFFFF 64%,#FFF1F5 100%);
        border:1px solid var(--sg-border); border-radius:14px; margin-bottom:16px;
        box-shadow:0 6px 22px rgba(36,36,41,.045);
    }
    .hero-accent { height:4px; background:linear-gradient(90deg,var(--sg-red) 0%,var(--sg-magenta) 58%,#FF709A 100%); }
    .hero-content { padding:16px 24px 18px; }
    .hero-topline { display:flex; justify-content:space-between; align-items:center; color:var(--sg-red); font-size:12px; font-weight:850; letter-spacing:.13em; }
    .role-badge { border:1px solid #F0B8C8; background:var(--sg-blush); border-radius:999px; padding:6px 11px; color:#A9003B; }
    .page-hero h1 { margin:7px 0 5px; font-size:32px; line-height:1.08; }
    .page-hero p { color:var(--sg-muted); font-size:16px; margin:0; max-width:800px; }
    .red-dot { color:var(--sg-red); }
    .hero-tags { display:flex; flex-wrap:wrap; gap:7px; margin-top:10px; }
    .hero-tags span { border:1px solid var(--sg-border); background:#FFF; border-radius:999px; color:#4F525A; padding:4px 10px; font-size:12px; }
    .customer-notice {
        display:flex; align-items:center; gap:14px; background:var(--sg-blush); border:1px solid #F3C5D2;
        border-radius:12px; color:#5D3945; padding:14px 18px; margin-bottom:18px;
    }
    .customer-notice b { color:#A9003B; white-space:nowrap; }
    .journey-steps { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:0 0 24px; }
    .journey-step { display:flex; align-items:center; gap:12px; background:#FFF; border:1px solid var(--sg-border); border-radius:12px; padding:15px 17px; color:#73757D; }
    .journey-step > span { display:flex; align-items:center; justify-content:center; width:30px; height:30px; flex:0 0 30px; border-radius:999px; background:#F0F0F2; color:#70727A; font-weight:800; }
    .journey-step strong, .journey-step small { display:block; }
    .journey-step strong { color:var(--sg-ink); font-size:14px; }
    .journey-step small { margin-top:2px; color:#777A82; font-size:11px; }
    .journey-step.is-active { border-color:#E98AA5; background:var(--sg-blush); }
    .journey-step.is-active > span, .journey-step.is-complete > span { background:var(--sg-red); color:#FFF; }
    .journey-step.is-complete { border-color:#F2C4D1; }
    .staff-flow { display:flex; flex-wrap:wrap; align-items:center; gap:8px; padding:2px 0 14px; }
    .staff-flow-step { color:#777A82; background:#FFF; border:1px solid var(--sg-border); border-radius:999px; padding:7px 12px; font-size:11px; font-weight:700; }
    .staff-flow-step.is-active { color:#FFF; background:var(--sg-red); border-color:var(--sg-red); }
    .staff-flow-step.is-complete { color:#A50038; background:var(--sg-blush); border-color:#F2C4D1; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:#FFF; border-radius:13px; border-color:var(--sg-border) !important; box-shadow:0 3px 14px rgba(36,36,41,.035); }
    [data-testid="stMetric"] { background:#FFF; padding:14px 16px; border-radius:10px; border:1px solid var(--sg-border); border-top:3px solid var(--sg-red); }
    [data-testid="stMetricValue"] { font-size:1.65rem; color:var(--sg-ink); }
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius:999px; min-height:42px; font-weight:700; padding-left:18px; padding-right:18px; }
    .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background:var(--sg-red); border-color:var(--sg-red); color:#FFF; }
    .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover { background:var(--sg-red-dark); border-color:var(--sg-red-dark); }
    .stButton > button[kind="secondary"], .stDownloadButton > button { background:#FFF; border-color:#D3D3D8; color:var(--sg-ink); }
    .stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover { border-color:var(--sg-red); color:var(--sg-red); }
    [data-baseweb="tab-list"] { gap:20px; border-bottom:1px solid var(--sg-border); }
    [data-baseweb="tab-highlight"] { background:var(--sg-red); }
    [data-baseweb="tab"] { color:#5F6169; }
    .step-label { color:#B0003C; font-weight:850; font-size:11px; letter-spacing:.14em; margin:2px 0 5px; }
    .empty-panel { border:1px dashed #C8C8CE; background:#FAFAFB; border-radius:12px; padding:30px; margin:12px 0; text-align:center; }
    .empty-panel h3 { margin:0 0 8px; }
    .empty-panel p { color:#6C6E76; margin:0; }
    .customer-status { background:linear-gradient(100deg,var(--sg-blush),#FFF); border-left:4px solid var(--sg-red); border-radius:10px; padding:20px; margin:0 0 14px; }
    .customer-status span { color:#A9003B; font-weight:850; font-size:11px; letter-spacing:.09em; text-transform:uppercase; }
    .customer-status h3 { margin:8px 0; }
    .customer-status p { color:#5F6169; margin:0; }
    .section-spacer { height:20px; }
    div[data-baseweb="select"] > div, input, textarea { border-radius:9px !important; }
    [data-testid="stAlert"] { border-radius:11px; }
    @media(max-width:760px) {
        .block-container { padding-top:.6rem; }
        .site-masthead, .site-brand { align-items:flex-start; }
        .site-masthead { flex-direction:column; gap:8px; }
        .site-brand { flex-direction:column; gap:1px; }
        .hero-content { padding:15px 18px 17px; }
        .page-hero h1 { font-size:29px; }
        .hero-topline { align-items:flex-start; gap:12px; }
        .journey-steps { grid-template-columns:1fr; }
        .customer-notice { align-items:flex-start; flex-direction:column; gap:3px; }
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


api_key = shared_key or st.session_state.get("api_key_cache", "")
workspace = st.session_state.workspace_page
render_workspace_switcher()

if workspace == "01 · Customer view demo":
    render_customer_workspace()
elif workspace == "02 · Service officer":
    render_officer_workspace(api_key)
elif workspace == "03 · Case review":
    render_case_review()
elif workspace == "04 · Processing & audit":
    render_processing_workspace(api_key, shared_key)
else:
    render_audit_trail()

# End the current Streamlit run after rendering the selected workspace.
st.stop()
