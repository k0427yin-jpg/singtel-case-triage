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

st.set_page_config(page_title="Singtel Case Triage (Prototype)", page_icon="\U0001F4E1")

SGT = timezone(timedelta(hours=8))
APP_VERSION = "2026-09-25-state-gates-v2"
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


# ---------------------------------------------------------------------------
# Step 2: AI classification (genuine AI functionality — real API call)
# ---------------------------------------------------------------------------

def classify_case(case_text: str, api_key: str) -> dict:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case_text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


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
                             override_reason: str = "") -> dict:
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
                "case_id", "prior_contacts", "unresolved", "work_impact"):
        state[key] = None


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
    if not inputs["case_id"].strip():
        return "Enter a Case ID before running the assessment. Spaces alone are not valid."
    if not has_meaningful_content(inputs["case_text"]) and not has_meaningful_content(inputs["history_text"]):
        return ("Not enough information to assess this case. Enter a customer "
                "message or previous interaction history with enough detail "
                "before running the assessment.")
    discrepancy = detect_contact_count_discrepancy(
        inputs["prior_contacts"], inputs["case_text"], inputs["history_text"],
    )
    if discrepancy:
        return (f"Discrepancy detected — officer verification required. {discrepancy} "
                "Correct the inputs and run the assessment again. No AI call was made.")
    if not api_key:
        return "Enter your OpenAI API key in the sidebar first."
    ai_case_input = f"""Service: Home Broadband
Previous support contacts: {inputs['prior_contacts']}
Issue still unresolved: {"Yes" if inputs['unresolved'] else "No"}
Affects work / critical activity: {"Yes" if inputs['work_impact'] else "No"}

Previous interaction history:
{inputs['history_text'].strip() or "None recorded."}

Customer message:
{inputs['case_text']}"""
    try:
        ai_result = classify_case(ai_case_input, api_key)
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
        return "AI assessment failed. No current result or ticket is available. Please try again."
    state["result"] = ai_result
    state["priority"] = priority
    state["case_id"] = inputs["case_id"].strip()
    state["assessed_inputs"] = dict(inputs)
    state["assessment_run"] = state.get("assessment_run", 0) + 1
    state["assessment_invalidated"] = False
    return None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F3F1;
    }
    .block-container { padding-top: 1.5rem; max-width: 900px; }

    .singtel-header {
        background: #1E191A;
        border-left: 6px solid #EE133B;
        color: #FFFFFF;
        padding: 1.5rem 1.8rem;
        border-radius: 8px;
        margin-bottom: 1.6rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }
    .singtel-header h1 {
        color: #FFFFFF;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0 0 0.4rem 0;
    }
    .singtel-header p {
        color: #C9C5C4;
        font-size: 0.92rem;
        line-height: 1.5;
        margin: 0;
    }

    /* Section headings (1. Case input, 2. AI classification, ...) */
    h3 {
        border-left: 4px solid #C4102C;
        padding-left: 0.65rem;
        color: #1E191A !important;
        margin-top: 1.6rem !important;
    }

    /* Card containers: the input form and every st.container(border=True) */
    [data-testid="stForm"],
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E7E3DF;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
        padding: 0.4rem 0.2rem;
    }
    [data-testid="stForm"] { padding: 1.4rem 1.4rem 1rem 1.4rem; }

    /* Buttons */
    div.stButton > button {
        border-radius: 6px;
        padding: 0.5rem 1.1rem;
        transition: all 0.15s ease;
    }
    div.stButton > button[kind="primary"] {
        background-color: #C4102C;
        border: none;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #A50E26;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF;
        border: 1px solid #D8D4CF;
        color: #1E191A;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #C4102C;
        color: #C4102C;
    }
    </style>
    <div class="singtel-header">
        <h1>Singtel Case Triage — prototype</h1>
        <p>Decision-support prototype for complex, repeated, unresolved home-broadband
        cases. Synthetic data only. No connection to live Singtel systems.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

shared_key = st.secrets.get("OPENAI_API_KEY", None)

with st.sidebar:
    st.header("Settings")
    st.caption(f"Prototype version: {APP_VERSION}")
    if shared_key:
        st.success("Using the evaluator API key provided by the developer.")
        override_key = st.text_input(
            "Use a different OpenAI API key (optional)",
            type="password",
            help="Leave blank to use the shared evaluator key. "
                 "Your own key is never stored.",
        )
        api_key = override_key.strip() or shared_key
    else:
        api_key = st.text_input(
            "OpenAI API key", type="password",
            help="Your own key. Not stored anywhere.",
        )
    st.markdown("---")
    st.caption(
        "This prototype demonstrates: (1) a real OpenAI API call for case "
        "classification and summarisation, (2) a deterministic Python scoring "
        "function, (3) a human approval gate, and (4) a simulated business "
        "action generated only after approval."
    )

if "result" not in st.session_state:
    st.session_state.result = None
if "priority" not in st.session_state:
    st.session_state.priority = None
if "record" not in st.session_state:
    st.session_state.record = None
if "assessment_run" not in st.session_state:
    st.session_state.assessment_run = 0
if "assessed_inputs" not in st.session_state:
    st.session_state.assessed_inputs = None
if "discrepancy" not in st.session_state:
    st.session_state.discrepancy = None

st.subheader("1. Case input")

# Plain (non-form) widgets, deliberately not wrapped in st.form: the app
# needs to detect, on every rerun, whether these inputs still match the
# values used for the last completed assessment (see the staleness check
# below), which a batched st.form would hide until the next submit.
with st.container(border=True):
    st.markdown("**Service:** Home Broadband")
    case_id = st.text_input("Case ID", value="REG-01")
    prior_contacts = st.number_input("Previous support contacts", min_value=0, value=4)
    unresolved = st.checkbox("Issue still unresolved", value=True)
    work_impact = st.checkbox("Affects customer's work / critical activity", value=True)
    history_text = st.text_area(
        "Previous interaction history",
        value=(
            "Contact 1 (5 days ago): Customer reported intermittent disconnections. "
            "Advised to restart the router.\n"
            "Contact 2 (3 days ago): Issue persisted after restart. A line test "
            "was scheduled.\n"
            "Contact 3 (1 day ago): Customer called again; problem still unresolved.\n"
            "Contact 4 (today): Customer followed up because disconnections "
            "continued to interrupt work calls; no resolution was recorded."
        ),
        height=110,
        help="Prior contact notes for this case, most recent last. The AI reads "
             "this alongside the current message for the summary. The 'Previous "
             "support contacts' count above is what drives the priority score — "
             "keep it consistent with the number of entries here.",
    )
    case_text = st.text_area(
        "Customer message",
        value=(
            "My home broadband keeps disconnecting and this problem has still not "
            "been fixed. I have contacted support four times already. I work from "
            "home and the connection drops are preventing me from joining client "
            "video calls. I am extremely frustrated because I have had to explain "
            "the same problem repeatedly."
        ),
        height=140,
    )
    submitted = st.button("Run AI assessment", type="primary")

current_inputs = {
    "case_id": case_id,
    "prior_contacts": prior_contacts,
    "unresolved": unresolved,
    "work_impact": work_impact,
    "history_text": history_text,
    "case_text": case_text,
}

# Clear outputs immediately when any assessed input changes, not only on submit.
invalidate_changed_inputs(st.session_state, current_inputs)

if submitted:
    with st.spinner("Checking inputs and preparing the assessment..."):
        assessment_error = run_assessment(st.session_state, current_inputs, api_key)
    if assessment_error:
        st.error(assessment_error)
elif st.session_state.get("assessment_invalidated", False):
    st.warning("No current assessment. Previous results and simulated ticket were cleared. Run AI assessment again.")

if st.session_state.result:
    r = st.session_state.result
    p = st.session_state.priority

    # Stale-assessment protection: compare the inputs currently shown on the
    # page against the snapshot taken when this assessment last ran.
    stale = current_inputs != st.session_state.assessed_inputs
    if stale:
        st.warning("Inputs changed after assessment. Run AI assessment again.")

    discrepancy = st.session_state.discrepancy
    if discrepancy and not stale:
        st.error(f"Discrepancy detected — officer verification required. {discrepancy}")

    with st.container(border=True):
        st.subheader("2. AI classification and summary (from OpenAI API)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Complexity", r["complexity"])
        col2.metric("Urgency", r["urgency"])
        col3.metric("Sentiment", r["sentiment"])
        st.write(f"**Intent:** {r['intent']}")
        with st.expander("Fault-detail keywords absent from the current message (reference only)"):
            st.write(r["missing_info"])
            st.caption(
                "This message-only keyword checklist does not inspect interaction "
                "history or structured fields. Absent keywords do not establish a "
                "case-level information gap. Review all case information before "
                "requesting details. Fault duration and troubleshooting may not "
                "apply to routine requests such as changing a Wi-Fi password. "
                "This checklist is not a safety or approval check."
            )
        st.write(f"**Case summary:** {r['summary']}")

    with st.container(border=True):
        st.subheader("3. Deterministic rule result (plain Python, not the AI model)")
        st.write(f"**Priority score:** {p['score']} rule points")
        st.caption(f"Current rules have a maximum of {MAX_RULE_POINTS} points. This is not a percentage.")
        st.write(f"**Breakdown:** {p['breakdown']}")
        st.write(f"**Escalation recommended:** {'Yes' if p['escalation_recommended'] else 'No'}")
        st.write(f"**Escalation trigger:** {p['trigger']}")
        st.caption(
            "Priority score: fixed Python rules use previous contacts, unresolved "
            "status, work impact and AI-classified sentiment. Escalation: separate "
            "Python rules recommend escalation when AI-classified urgency is High, "
            "AI-classified complexity is High, or the priority score is greater "
            "than 70. Urgency and complexity do not add points to the score."
        )

    if p["escalation_recommended"]:
        default_queue = "Specialist broadband escalation queue"
        default_action = "Route for escalation and further investigation."
    else:
        default_queue = "Standard broadband support queue"
        default_action = "Provide standard troubleshooting guidance; no escalation required based on current information."

    with st.container(border=True):
        st.subheader("4. Human decision")
        st.write("A human service officer retains final authority.")
        st.caption(
            f"Suggested based on the rule result (Escalation recommended: "
            f"{'Yes' if p['escalation_recommended'] else 'No'}). The officer can accept, "
            f"change, or override this suggestion."
        )
        run_id = st.session_state.assessment_run
        queue = st.text_input("Assigned queue", value=default_queue, key=f"queue_input_{run_id}")
        action_note = st.text_area("Approved action / notes", value=default_action, key=f"action_input_{run_id}")

        override_reason = st.text_area(
            "Override reason (required for Modify or a changed escalation recommendation)",
            key=f"override_reason_{run_id}",
        )

        blocked = stale or bool(discrepancy) or not case_id.strip()
        if blocked:
            st.caption(
                "Accept, Modify and Escalate are disabled until this is "
                "resolved: rerun the assessment if inputs changed, or "
                "correct the discrepancy above."
            )

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        decision = None
        validation_error = None

        # ACCEPT — only valid when the recommendation has not been changed
        if dcol1.button("Accept", type="primary", disabled=blocked):
            if queue.strip() != default_queue or action_note.strip() != default_action:
                validation_error = (
                    "Accept can only be used when the system recommendation is "
                    "unchanged. Use Modify if you changed the queue or action/notes."
                )
            else:
                decision = "Accept"

        # MODIFY — requires an actual change
        if dcol2.button("Modify", disabled=blocked):
            if queue.strip() == default_queue and action_note.strip() == default_action:
                validation_error = (
                    "Modify requires an actual change to the assigned queue or the "
                    "approved action/notes. Edit one of these fields before selecting "
                    "Modify, or choose Accept if the suggested routing is correct."
                )
            elif not override_reason.strip():
                validation_error = "Enter an override reason before selecting Modify."
            else:
                decision = "Modify"

        # ESCALATE — always places the case on the escalation path. The action
        # note is rebuilt from scratch rather than appended to the non-escalation
        # default, so it never contains a contradictory "no escalation required"
        # phrase alongside "escalated".
        if dcol3.button("Escalate", disabled=blocked):
            queue = "Specialist broadband escalation queue"
            if not p["escalation_recommended"]:
                custom_note = action_note.strip()
                if custom_note == default_action:
                    custom_note = ""
                action_note = (
                    "Route for escalation and further investigation. Officer "
                    "override: the deterministic rules did not recommend escalation."
                )
                if custom_note:
                    action_note += f" Officer note: {custom_note}"
            changed = queue != default_queue or action_note.strip() != default_action
            if changed and not override_reason.strip():
                validation_error = "Enter an override reason before changing the escalation recommendation."
            else:
                decision = "Escalate"

        # NO DECISION — business action must remain blocked
        if dcol4.button("No decision (test block)"):
            decision = "None"

        if validation_error:
            st.session_state.record = None
            st.error(validation_error)

        if decision:
            record = create_simulated_record(
                st.session_state.case_id, p["score"], queue.strip(), decision, action_note.strip(),
                original_queue=default_queue, original_action=default_action,
                override_reason=override_reason,
            )
            st.session_state.record = {**record, "decision": decision, "queue": queue, "action": action_note}

    if st.session_state.record:
        rec = st.session_state.record
        with st.container(border=True):
            st.subheader("5. Simulated business action")
            if rec["status"].startswith("BLOCKED"):
                st.warning(f"NO SIMULATED RECORD CREATED — {rec['status']}")
            else:
                st.success("SIMULATED TICKET — DEMONSTRATION ONLY")
                st.json({
                    "case_id": rec["case_id"],
                    "ticket_id": rec["ticket_id"],
                    "priority_score": rec["priority_score"],
                    "score_unit": rec["score_unit"],
                    "max_rule_points": rec["max_rule_points"],
                    "prototype_version": rec["prototype_version"],
                    "assigned_queue": rec["assigned_queue"],
                    "human_decision": rec["human_decision"],
                    "approved_action": rec["approved_action"],
                    "original_queue": rec["original_queue"],
                    "original_action": rec["original_action"],
                    "override_reason": rec["override_reason"],
                    "timestamp_sgt": rec["timestamp"],
                    "status": rec["status"],
                })
            st.caption(
                "This record is generated by this app's own code, not by the AI model, "
                "and does not exist in any live Singtel system."
            )
