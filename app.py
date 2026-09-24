"""
Singtel AI Complex Case Intelligence and Escalation System — prototype
Streamlit + OpenAI API version.

Pipeline:
1. Synthetic case input (this app, form or free text)
2. AI classification + summary (OpenAI API call — genuine AI functionality)
3. Deterministic rule-based scoring (plain Python — not the LLM)
4. Human decision gate (Accept / Modify / Escalate)
5. Simulated case record (generated only after a decision, by this app,
   with a real timestamp and generated ID — never invented by the model)

No live Singtel systems are contacted. All data is synthetic.
"""

import json
import random
import string
from datetime import datetime, timezone, timedelta

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Singtel Case Triage (Prototype)", page_icon="\U0001F4E1")

SGT = timezone(timedelta(hours=8))

SYSTEM_PROMPT = """You are an AI case-assessment assistant for a prototype called
"Singtel AI Complex Case Intelligence and Escalation System". You provide decision
support for a human service officer handling SIMULATED Singtel home-broadband cases.

Use only the information given in the case. Do not invent facts, customer history,
or Singtel system data. Do not make the final decision — a human service officer
retains final authority.

Classify the case and return STRICT JSON with exactly these keys:
{
  "intent": "<one short sentence>",
  "complexity": "Low" | "Medium" | "High",
  "urgency": "Low" | "Medium" | "High",
  "sentiment": "Positive" | "Neutral" | "Frustrated",
  "missing_info": "<short text, or 'None'>",
  "summary": "<concise grounded summary, based only on the case text>"
}

Missing-information check: before writing "missing_info", check the case text
against this list and name any item it does not clearly state:
- how long the problem has been happening (duration)
- whether it affects all services or only some (scope)
- what troubleshooting, if any, the customer has already tried
- the current status of the service (fully down vs intermittent vs slow)
Do not infer these details from the seriousness of the case or from the
structured fields (previous contacts, unresolved status, work impact). Check
each required item separately against the customer message text alone. If
even one required item is not explicitly stated in the customer message, list
it as missing. A short or vague message that does not cover these should list
them as missing, not "None". Only write "None" if the customer message text
actually addresses all of them explicitly.

Interpret the checklist strictly:
- "ongoing", "keeps happening", or "again" does NOT satisfy duration.
  Duration requires an explicit time reference such as "for three days",
  "since this morning", or "for 20 minutes".
- Contacting customer support does NOT count as troubleshooting.
  Troubleshooting requires an explicit action such as restarting the router,
  checking cables, resetting equipment, or another technical step.
- Mentioning one affected activity or one device does NOT establish scope.
  Scope requires explicit information about whether all devices/services or
  only some are affected.
- "Since this morning" DOES satisfy duration.

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

    escalate = (
        complexity.lower() == "high"
        or urgency.lower() == "high"
        or score > 70
    )

    breakdown_str = " + ".join(f"{label} {val}" for label, val in breakdown) + f" = {score}"

    return {
        "score": score,
        "breakdown": breakdown_str,
        "escalation_recommended": escalate,
    }


# ---------------------------------------------------------------------------
# Step 5: Simulated business action — generated by this app's own code,
# only after an explicit human decision. Never invented by the model.
# ---------------------------------------------------------------------------

def create_simulated_record(case_id: str, priority_score: int, assigned_queue: str,
                             officer_decision: str, approved_action: str) -> dict:
    valid_decisions = {"accept", "modify", "escalate"}
    if officer_decision.lower() not in valid_decisions:
        return {
            "status": "BLOCKED - no valid officer decision (Accept, Modify or Escalate)",
            "ticket_id": "",
            "timestamp": "",
        }

    ticket_id = "SIM-" + datetime.now(SGT).strftime("%Y%m%d") + "-" + \
        "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    timestamp = datetime.now(SGT).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": "Created for simulation",
        "ticket_id": ticket_id,
        "timestamp": timestamp,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
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
    div.stButton > button[kind="primary"] {
        background-color: #C4102C;
        border: none;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #A50E26;
    }
    div.stButton > button {
        border-radius: 6px;
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

st.subheader("1. Case input")

with st.form("case_form"):
    case_id = st.text_input("Case ID", value="REG-01")
    prior_contacts = st.number_input("Previous support contacts", min_value=0, value=4)
    unresolved = st.checkbox("Issue still unresolved", value=True)
    work_impact = st.checkbox("Affects customer's work / critical activity", value=True)
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
    submitted = st.form_submit_button("Run AI assessment", type="primary")

if submitted:
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar first.")
    else:
        with st.spinner("Calling AI model for classification and summary..."):
            try:
                ai_case_input = f"""Previous support contacts: {prior_contacts}
Issue still unresolved: {"Yes" if unresolved else "No"}
Affects work / critical activity: {"Yes" if work_impact else "No"}

Customer message:
{case_text}"""
                ai_result = classify_case(ai_case_input, api_key)
                st.session_state.result = ai_result
                st.session_state.case_id = case_id
                st.session_state.prior_contacts = prior_contacts
                st.session_state.unresolved = unresolved
                st.session_state.work_impact = work_impact
                priority = calculate_priority(
                    prior_contacts, unresolved, work_impact,
                    ai_result["sentiment"], ai_result["complexity"], ai_result["urgency"],
                )
                st.session_state.priority = priority
                st.session_state.record = None
                st.session_state.assessment_run += 1
            except Exception as e:
                st.error(f"AI call failed: {e}")

if st.session_state.result:
    r = st.session_state.result
    p = st.session_state.priority

    st.subheader("2. AI classification and summary (from OpenAI API)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Complexity", r["complexity"])
    col2.metric("Urgency", r["urgency"])
    col3.metric("Sentiment", r["sentiment"])
    st.write(f"**Intent:** {r['intent']}")
    st.write(f"**Missing or ambiguous information:** {r['missing_info']}")
    st.write(f"**Case summary:** {r['summary']}")

    st.subheader("3. Deterministic rule result (plain Python, not the AI model)")
    st.write(f"**Priority score:** {p['score']} / 100")
    st.write(f"**Breakdown:** {p['breakdown']}")
    st.write(f"**Escalation recommended:** {'Yes' if p['escalation_recommended'] else 'No'}")
    st.caption(
        "This score is calculated by a fixed Python function using the classified "
        "factors above. It is not generated by the language model."
    )

    if p["escalation_recommended"]:
        default_queue = "Specialist broadband escalation queue"
        default_action = "Route for escalation and further investigation."
    else:
        default_queue = "Standard broadband support queue"
        default_action = "Provide standard troubleshooting guidance; no escalation required based on current information."

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

    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    decision = None
    validation_error = None

    # ACCEPT — only valid when the recommendation has not been changed
    if dcol1.button("Accept", type="primary"):
        if queue.strip() != default_queue or action_note.strip() != default_action:
            validation_error = (
                "Accept can only be used when the system recommendation is "
                "unchanged. Use Modify if you changed the queue or action/notes."
            )
        else:
            decision = "Accept"

    # MODIFY — requires an actual change
    if dcol2.button("Modify"):
        if queue.strip() == default_queue and action_note.strip() == default_action:
            validation_error = (
                "Modify requires an actual change to the assigned queue or the "
                "approved action/notes. Edit one of these fields before selecting "
                "Modify, or choose Accept if the suggested routing is correct."
            )
        else:
            decision = "Modify"

    # ESCALATE — always places the case on the escalation path. The action
    # note is rebuilt from scratch rather than appended to the non-escalation
    # default, so it never contains a contradictory "no escalation required"
    # phrase alongside "escalated".
    if dcol3.button("Escalate"):
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
        else:
            if action_note.strip() == default_action:
                action_note = default_action
            action_note = "Escalation confirmed by officer. " + action_note.strip()
        decision = "Escalate"

    # NO DECISION — business action must remain blocked
    if dcol4.button("No decision (test block)"):
        decision = "None"

    if validation_error:
        st.error(validation_error)

    if decision:
        record = create_simulated_record(
            st.session_state.case_id, p["score"], queue, decision, action_note,
        )
        st.session_state.record = {**record, "decision": decision, "queue": queue, "action": action_note}

    if st.session_state.record:
        rec = st.session_state.record
        st.subheader("5. Simulated business action")
        if rec["status"].startswith("BLOCKED"):
            st.warning(f"NO SIMULATED RECORD CREATED — {rec['status']}")
        else:
            st.success("SIMULATED TICKET — DEMONSTRATION ONLY")
            st.json({
                "case_id": st.session_state.case_id,
                "ticket_id": rec["ticket_id"],
                "priority_score": p["score"],
                "assigned_queue": rec["queue"],
                "human_decision": rec["decision"],
                "approved_action": rec["action"],
                "timestamp_sgt": rec["timestamp"],
                "status": rec["status"],
            })
        st.caption(
            "This record is generated by this app's own code, not by the AI model, "
            "and does not exist in any live Singtel system."
        )
