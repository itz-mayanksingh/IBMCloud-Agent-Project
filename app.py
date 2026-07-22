# =============================================================================
# NutriWise AI – Personalized Nutrition Coach
# =============================================================================
# A multi-agent AI nutrition assistant powered by IBM watsonx.ai Granite Models
# Built with Python, Flask, Bootstrap 5, and IBM watsonx.ai
#
# Agent Architecture:
#   Agent 1 – Nutrition Knowledge Agent  → answers nutrition questions
#   Agent 2 – Diet Planner Agent         → creates personalized meal plans
#   Agent 3 – Health Advisory Agent      → disease-specific food guidance
#   Agent 4 – Meal Analysis Agent        → analyzes user-entered meals
#
# Orchestrator routes each request to the correct agent.
# All agents call generate_response(prompt) which hits IBM watsonx.ai.
# =============================================================================

import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template_string
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# =============================================================================
# Flask App Initialisation
# =============================================================================
app = Flask(__name__)
app.secret_key = os.urandom(24)

# =============================================================================
# IBM watsonx.ai Configuration
# Credentials are read from environment variables – never hard-coded.
#   WATSONX_API_KEY    – your IBM Cloud API key
#   WATSONX_PROJECT_ID – watsonx.ai project ID
#   WATSONX_URL        – regional endpoint, e.g. https://us-south.ml.cloud.ibm.com
# =============================================================================
WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY",    "your-api-key-here")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "your-project-id-here")
WATSONX_URL        = os.environ.get("WATSONX_URL",        "https://us-south.ml.cloud.ibm.com")

# IBM Granite model identifier — granite-4-h-small (instruct, available on this project)
GRANITE_MODEL_ID = "ibm/granite-4-h-small"

def get_watsonx_model():
    """
    Initialise and return a watsonx.ai ModelInference instance.
    Called fresh for every request so credential changes are picked up
    without restarting the server.
    Uses the chat API (generate_text via messages) to avoid the
    deprecated /ml/v1/text/generation endpoint.
    """
    credentials = Credentials(
        url=WATSONX_URL,
        api_key=WATSONX_API_KEY,
    )
    model = ModelInference(
        model_id=GRANITE_MODEL_ID,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
        params={
            GenParams.MAX_NEW_TOKENS: 1024,
            GenParams.TEMPERATURE:    0.7,
            GenParams.TOP_P:          0.9,
        },
    )
    return model


# =============================================================================
# Core AI Function – all agents route through here
# =============================================================================
def generate_response(prompt: str) -> str:
    """
    Send a prompt to IBM watsonx.ai Granite model and return the text response.
    This is the single integration point for all four agents.

    Uses the chat (messages) API to call the model, which is the current
    recommended approach and avoids the deprecated /ml/v1/text/generation
    endpoint warning.

    Args:
        prompt: The formatted prompt string built by the calling agent.

    Returns:
        str: AI-generated response text, or an error message on failure.
    """
    try:
        # ── IBM watsonx.ai API call (chat messages API) ──────────────────────
        model = get_watsonx_model()
        messages = [{"role": "user", "content": prompt}]
        response = model.chat(messages=messages)
        # Extract the assistant message content from the chat response
        content = (
            response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
        )
        # ─────────────────────────────────────────────────────────────────────
        return content.strip() if content else "No response generated."
    except Exception as e:
        return f"⚠️ AI service error: {str(e)}"


# =============================================================================
# AGENT 1 – Nutrition Knowledge Agent
# Answers general nutrition and food-science questions.
# =============================================================================
def nutrition_knowledge_agent(question: str) -> str:
    """
    Agent 1: Nutrition Knowledge Agent
    Builds an educational nutrition prompt and calls IBM watsonx.ai.
    """
    prompt = f"""You are NutriWise, an expert nutrition scientist and dietitian.
Answer the following nutrition question in a clear, structured, and educational way.
Use bullet points, examples, and practical tips where helpful.
Keep your answer factual and evidence-based.

Question: {question}

Provide a comprehensive yet concise answer covering:
- Direct answer to the question
- Key nutritional facts
- Health benefits or concerns
- Practical dietary tips
- Any important considerations

Answer:"""

    # ── IBM watsonx.ai Granite call via orchestrator ────────────────────────
    return generate_response(prompt)


# =============================================================================
# AGENT 2 – Diet Planner Agent
# Creates a full-day personalised meal plan based on user profile data.
# =============================================================================
def diet_planner_agent(age: str, gender: str, height: str, weight: str,
                       dietary_pref: str, activity_level: str,
                       fitness_goal: str) -> str:
    """
    Agent 2: Diet Planner Agent
    Builds a personalised meal-plan prompt and calls IBM watsonx.ai.
    """
    prompt = f"""You are NutriWise Diet Planner, an expert nutritionist and meal planning specialist.
Create a detailed, personalized one-day meal plan for the following individual:

User Profile:
- Age: {age} years
- Gender: {gender}
- Height: {height} cm
- Weight: {weight} kg
- Dietary Preference: {dietary_pref}
- Activity Level: {activity_level}
- Fitness Goal: {fitness_goal}

Provide a complete meal plan with the following structure:

1. DAILY NUTRITION TARGETS
   - Daily Calorie Target (kcal)
   - Protein Recommendation (g)
   - Carbohydrate Recommendation (g)
   - Fat Recommendation (g)

2. MEAL PLAN
   Breakfast:
   - List specific foods with approximate portions
   
   Morning Snack:
   - List specific foods with approximate portions
   
   Lunch:
   - List specific foods with approximate portions
   
   Evening Snack:
   - List specific foods with approximate portions
   
   Dinner:
   - List specific foods with approximate portions

3. HYDRATION
   - Daily water intake recommendation

4. KEY TIPS
   - 3-4 personalised tips aligned with the fitness goal

Make the plan realistic, culturally diverse, and aligned with the dietary preference.
Response:"""

    # ── IBM watsonx.ai Granite call via orchestrator ────────────────────────
    return generate_response(prompt)


# =============================================================================
# AGENT 3 – Health Advisory Agent
# Provides disease-specific dietary guidance.
# =============================================================================
def health_advisory_agent(conditions: list) -> str:
    """
    Agent 3: Health Advisory Agent
    Builds a disease-specific dietary advice prompt and calls IBM watsonx.ai.
    Always appends the required medical disclaimer.
    """
    conditions_str = ", ".join(conditions) if conditions else "General Health"

    prompt = f"""You are NutriWise Health Advisor, a clinical dietitian specializing in medical nutrition therapy.
Provide comprehensive dietary and lifestyle recommendations for someone managing: {conditions_str}

Structure your response as follows:

1. OVERVIEW
   Brief explanation of how diet affects the mentioned condition(s)

2. FOODS TO INCLUDE ✅
   - List 8-10 beneficial foods with brief reasons

3. FOODS TO AVOID ❌
   - List 8-10 foods to limit or avoid with brief reasons

4. HEALTHY EATING HABITS
   - List 5-6 practical eating habits

5. LIFESTYLE RECOMMENDATIONS
   - List 4-5 lifestyle changes beyond diet

6. SAMPLE DAILY EATING PATTERN
   - Simple meal timing and portions guideline

Provide evidence-based, practical advice that is easy to follow.
Response:"""

    # ── IBM watsonx.ai Granite call via orchestrator ────────────────────────
    ai_response = generate_response(prompt)

    # Mandatory medical disclaimer appended after every health advisory
    disclaimer = (
        "\n\n---\n"
        "⚠️ **MEDICAL DISCLAIMER:** This information is for educational purposes only and "
        "does not constitute medical advice. Always consult a qualified healthcare professional "
        "or registered dietitian before making significant changes to your diet, especially "
        "if you have a medical condition."
    )
    return ai_response + disclaimer


# =============================================================================
# AGENT 4 – Meal Analysis Agent
# Analyses a free-text meal log for nutritional quality and improvements.
# =============================================================================
def meal_analysis_agent(meal_log: str) -> str:
    """
    Agent 4: Meal Analysis Agent
    Builds a meal-analysis prompt and calls IBM watsonx.ai.
    """
    prompt = f"""You are NutriWise Meal Analyst, an expert nutritionist specializing in dietary assessment.
Analyze the following meal log and provide detailed nutritional feedback:

Meal Log:
{meal_log}

Provide a comprehensive analysis with this structure:

1. NUTRITIONAL QUALITY OVERVIEW
   - Overall diet quality score (out of 10) with justification
   - Estimated calorie range for the day
   - General nutritional assessment

2. MACRONUTRIENT ANALYSIS
   - Estimated Protein intake assessment
   - Estimated Carbohydrate intake assessment
   - Estimated Fat intake assessment
   - Fibre assessment

3. MICRONUTRIENT SPOTLIGHT
   - Key vitamins and minerals present
   - Potential micronutrient gaps

4. STRENGTHS 💪
   - List what is good about this meal plan (4-5 points)

5. NUTRITIONAL GAPS & CONCERNS ⚠️
   - List nutritional deficiencies or concerns (4-5 points)

6. HEALTHIER ALTERNATIVES 🔄
   - Suggest healthier swaps for specific foods mentioned

7. IMPROVEMENT RECOMMENDATIONS 📋
   - List 5-6 actionable steps to improve the overall diet

8. SUMMARY
   - Brief motivational closing statement with key takeaways

Be specific, practical, and encouraging in your feedback.
Response:"""

    # ── IBM watsonx.ai Granite call via orchestrator ────────────────────────
    return generate_response(prompt)


# =============================================================================
# Agent Orchestrator
# Routes incoming requests to the appropriate specialised agent.
# =============================================================================
def orchestrator(agent_type: str, data: dict) -> str:
    """
    Central orchestrator that routes requests to the correct agent.

    Supported agent_type values:
        'nutrition'  → Agent 1: Nutrition Knowledge Agent
        'diet'       → Agent 2: Diet Planner Agent
        'health'     → Agent 3: Health Advisory Agent
        'meal'       → Agent 4: Meal Analysis Agent
    """
    if agent_type == "nutrition":
        return nutrition_knowledge_agent(data.get("question", ""))

    elif agent_type == "diet":
        return diet_planner_agent(
            age=data.get("age", ""),
            gender=data.get("gender", ""),
            height=data.get("height", ""),
            weight=data.get("weight", ""),
            dietary_pref=data.get("dietary_pref", ""),
            activity_level=data.get("activity_level", ""),
            fitness_goal=data.get("fitness_goal", ""),
        )

    elif agent_type == "health":
        return health_advisory_agent(data.get("conditions", []))

    elif agent_type == "meal":
        return meal_analysis_agent(data.get("meal_log", ""))

    else:
        return "⚠️ Unknown agent type. Please select a valid feature."


# =============================================================================
# HTML Templates (render_template_string – all HTML lives in app.py)
# =============================================================================

# -----------------------------------------------------------------------------
# Base Layout – shared sidebar, navbar, and Bootstrap 5 setup
# -----------------------------------------------------------------------------
BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{{ title }} – NutriWise AI</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet"/>
  <style>
    :root {
      --primary:   #1a6b3a;
      --primary-d: #124d2b;
      --accent:    #28a745;
      --light-bg:  #f4faf6;
      --card-br:   12px;
      --sidebar-w: 260px;
    }
    * { box-sizing: border-box; }
    body { margin:0; font-family: 'Segoe UI', system-ui, sans-serif;
           background: var(--light-bg); color: #1e2a22; }

    /* ── Sidebar ── */
    #sidebar {
      position: fixed; top:0; left:0; height:100vh; width:var(--sidebar-w);
      background: linear-gradient(180deg, #0f3d22 0%, #1a6b3a 100%);
      display:flex; flex-direction:column; z-index:1000;
      box-shadow: 3px 0 15px rgba(0,0,0,.25);
    }
    #sidebar .brand {
      padding: 24px 20px 16px;
      border-bottom: 1px solid rgba(255,255,255,.12);
    }
    #sidebar .brand h4 { color:#fff; margin:0; font-weight:700; font-size:1.15rem; }
    #sidebar .brand small { color:rgba(255,255,255,.65); font-size:.75rem; }
    #sidebar nav { flex:1; padding:16px 0; overflow-y:auto; }
    #sidebar nav a {
      display:flex; align-items:center; gap:10px;
      padding:11px 22px; color:rgba(255,255,255,.8);
      text-decoration:none; font-size:.9rem; border-radius:0;
      transition:background .2s,color .2s;
      border-left: 3px solid transparent;
    }
    #sidebar nav a:hover,
    #sidebar nav a.active {
      background:rgba(255,255,255,.12);
      color:#fff;
      border-left-color:#6ee09a;
    }
    #sidebar nav a i { font-size:1rem; width:18px; text-align:center; }
    #sidebar .sidebar-footer {
      padding:16px 20px;
      border-top:1px solid rgba(255,255,255,.12);
      color:rgba(255,255,255,.5); font-size:.72rem; text-align:center;
    }

    /* ── Main content ── */
    #main { margin-left:var(--sidebar-w); min-height:100vh; }
    .topbar {
      background:#fff; padding:14px 28px;
      border-bottom:1px solid #d8ead0;
      display:flex; align-items:center; justify-content:space-between;
      position:sticky; top:0; z-index:100;
      box-shadow:0 1px 6px rgba(0,0,0,.06);
    }
    .topbar h5 { margin:0; font-weight:600; color:var(--primary); font-size:1.05rem; }
    .topbar .badge-ibm {
      background:#052b6e; color:#fff; font-size:.72rem;
      padding:4px 10px; border-radius:20px; font-weight:500;
    }
    .content-wrap { padding:28px; max-width:960px; margin:0 auto; }

    /* ── Cards ── */
    .nw-card {
      background:#fff; border-radius:var(--card-br);
      border:1px solid #d8ead0;
      box-shadow:0 2px 8px rgba(0,0,0,.05);
      padding:24px; margin-bottom:22px;
    }
    .nw-card h5 { color:var(--primary); font-weight:700; margin-bottom:4px; }
    .nw-card .sub { color:#5c7a66; font-size:.85rem; margin-bottom:16px; }

    /* ── Agent hero cards (home) ── */
    .agent-card {
      border-radius:var(--card-br); padding:22px;
      border:1px solid #d8ead0; background:#fff;
      transition:transform .2s, box-shadow .2s;
      height:100%;
    }
    .agent-card:hover { transform:translateY(-3px); box-shadow:0 6px 20px rgba(0,0,0,.1); }
    .agent-card .icon-wrap {
      width:52px; height:52px; border-radius:12px;
      display:flex; align-items:center; justify-content:center;
      font-size:1.5rem; margin-bottom:14px;
    }
    .agent-card h6 { font-weight:700; color:#1e2a22; margin-bottom:6px; }
    .agent-card p  { color:#5c7a66; font-size:.85rem; margin:0; }

    /* ── Forms ── */
    .form-label { font-weight:500; font-size:.88rem; color:#2d4a36; }
    .form-control, .form-select {
      border-radius:8px; border:1px solid #b8d4c0;
      font-size:.9rem;
    }
    .form-control:focus, .form-select:focus {
      border-color:var(--accent); box-shadow:0 0 0 3px rgba(40,167,69,.15);
    }
    .btn-nw {
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color:#fff; border:none; border-radius:8px;
      padding:10px 28px; font-weight:600; font-size:.92rem;
      transition:opacity .2s, transform .2s;
    }
    .btn-nw:hover { opacity:.9; transform:translateY(-1px); color:#fff; }
    .btn-nw:disabled { opacity:.6; }

    /* ── Response box ── */
    #response-area {
      display:none; background:#f8fcf9;
      border:1px solid #b8d4c0; border-radius:10px;
      padding:22px; margin-top:20px;
      white-space:pre-wrap; line-height:1.75;
      font-size:.9rem; color:#1e2a22;
      max-height:520px; overflow-y:auto;
    }

    /* ── Loader ── */
    .loader {
      display:none; text-align:center; padding:32px 0;
    }
    .loader .spinner-border { color:var(--accent); width:2.5rem; height:2.5rem; }
    .loader p { color:#5c7a66; margin-top:10px; font-size:.88rem; }

    /* ── Condition checkboxes ── */
    .condition-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(175px,1fr)); gap:10px; }
    .condition-item {
      border:1px solid #c8dece; border-radius:8px; padding:10px 14px;
      display:flex; align-items:center; gap:8px; cursor:pointer;
      transition:background .2s, border-color .2s;
    }
    .condition-item:hover { background:#eef7f1; border-color:var(--accent); }
    .condition-item input[type=checkbox]:checked ~ span { color:var(--primary); font-weight:600; }

    /* ── About page timeline ── */
    .timeline { border-left:3px solid var(--accent); padding-left:20px; margin-left:8px; }
    .timeline-item { margin-bottom:24px; position:relative; }
    .timeline-item::before {
      content:''; position:absolute; left:-27px; top:4px;
      width:11px; height:11px; border-radius:50%;
      background:var(--accent); border:2px solid #fff;
      box-shadow:0 0 0 2px var(--accent);
    }
    .timeline-item h6 { font-weight:700; color:var(--primary); margin-bottom:4px; }
    .timeline-item p  { font-size:.85rem; color:#4a6654; margin:0; }

    /* ── Responsive ── */
    @media(max-width:768px){
      #sidebar { width:220px; }
      #main   { margin-left:0; }
    }
  </style>
</head>
<body>

<!-- ===== Sidebar ===== -->
<div id="sidebar">
  <div class="brand">
    <h4><i class="bi bi-activity me-2"></i>NutriWise AI</h4>
    <small>Personalized Nutrition Coach</small>
  </div>
  <nav>
    <a href="/"            class="{{ 'active' if active=='home'      else '' }}"><i class="bi bi-house-heart-fill"></i>Home</a>
    <a href="/nutrition"   class="{{ 'active' if active=='nutrition' else '' }}"><i class="bi bi-chat-dots-fill"></i>Nutrition Chat</a>
    <a href="/diet-planner" class="{{ 'active' if active=='diet'     else '' }}"><i class="bi bi-calendar2-heart-fill"></i>Diet Planner</a>
    <a href="/health-advisor" class="{{ 'active' if active=='health' else '' }}"><i class="bi bi-heart-pulse-fill"></i>Health Advisor</a>
    <a href="/meal-analyzer" class="{{ 'active' if active=='meal'    else '' }}"><i class="bi bi-search-heart-fill"></i>Meal Analyzer</a>
    <a href="/about"        class="{{ 'active' if active=='about'    else '' }}"><i class="bi bi-info-circle-fill"></i>About</a>
  </nav>
  <div class="sidebar-footer">
    Powered by IBM watsonx.ai<br>Granite Models
  </div>
</div>

<!-- ===== Main ===== -->
<div id="main">
  <div class="topbar">
    <h5><i class="bi bi-{{ icon }} me-2"></i>{{ title }}</h5>
    <span class="badge-ibm"><i class="bi bi-cpu me-1"></i>IBM watsonx.ai · Granite</span>
  </div>
  <div class="content-wrap">
    {{ content | safe }}
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
// Utility – show loader, hide response, then POST to API
function runAgent(endpoint, payload, responseId) {
  const loader   = document.getElementById('loader');
  const respArea = document.getElementById(responseId || 'response-area');
  const btn      = document.getElementById('submit-btn');

  if(loader)   { loader.style.display   = 'block'; }
  if(respArea) { respArea.style.display = 'none';  }
  if(btn)      { btn.disabled = true; btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Thinking…'; }

  fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(r => r.json())
  .then(data => {
    if(loader)   { loader.style.display = 'none'; }
    if(btn)      { btn.disabled = false; btn.innerHTML = btn.dataset.label || 'Generate'; }
    if(respArea) {
      respArea.style.display = 'block';
      respArea.textContent   = data.response || data.error || 'No response.';
      respArea.scrollTop = 0;
    }
  })
  .catch(err => {
    if(loader)   { loader.style.display = 'none'; }
    if(btn)      { btn.disabled = false; btn.innerHTML = btn.dataset.label || 'Generate'; }
    if(respArea) {
      respArea.style.display = 'block';
      respArea.textContent   = 'Error: ' + err.message;
    }
  });
}
</script>
{{ extra_js | safe }}
</body>
</html>"""

# -----------------------------------------------------------------------------
# Home Page
# -----------------------------------------------------------------------------
HOME_CONTENT = """
<div class="nw-card">
  <div class="d-flex align-items-center gap-3 mb-3">
    <div style="background:linear-gradient(135deg,#1a6b3a,#28a745);border-radius:14px;
                width:60px;height:60px;display:flex;align-items:center;
                justify-content:center;font-size:1.8rem;color:#fff;">
      <i class="bi bi-stars"></i>
    </div>
    <div>
      <h4 class="mb-0 fw-700" style="color:#1a6b3a;">Welcome to NutriWise AI</h4>
      <p class="mb-0" style="color:#5c7a66;font-size:.9rem;">
        Your AI-powered personalized nutrition coach — built on IBM watsonx.ai Granite Models
      </p>
    </div>
  </div>
  <p style="color:#3a5a44;font-size:.92rem;">
    NutriWise AI harnesses the power of <strong>IBM Granite large language models</strong> through a
    <strong>multi-agent architecture</strong> to deliver evidence-based nutrition guidance, personalized
    meal plans, and disease-specific dietary recommendations — all tailored to you.
  </p>
</div>

<h6 class="mb-3 fw-600" style="color:#1a6b3a;letter-spacing:.03em;">
  <i class="bi bi-grid-3x3-gap-fill me-2"></i>Meet the Four AI Agents
</h6>
<div class="row g-3 mb-4">
  <div class="col-md-6">
    <div class="agent-card">
      <div class="icon-wrap" style="background:#e8f5e9;color:#1a6b3a;">
        <i class="bi bi-chat-dots-fill"></i>
      </div>
      <h6>Agent 1 – Nutrition Knowledge</h6>
      <p>Ask any nutrition question and receive evidence-based answers powered by IBM Granite AI.</p>
      <a href="/nutrition" class="btn btn-sm btn-nw mt-2">Ask a Question</a>
    </div>
  </div>
  <div class="col-md-6">
    <div class="agent-card">
      <div class="icon-wrap" style="background:#e3f2fd;color:#1565c0;">
        <i class="bi bi-calendar2-heart-fill"></i>
      </div>
      <h6>Agent 2 – Diet Planner</h6>
      <p>Enter your profile details and get a full personalised meal plan with macro targets.</p>
      <a href="/diet-planner" class="btn btn-sm btn-nw mt-2">Plan My Diet</a>
    </div>
  </div>
  <div class="col-md-6">
    <div class="agent-card">
      <div class="icon-wrap" style="background:#fce4ec;color:#c62828;">
        <i class="bi bi-heart-pulse-fill"></i>
      </div>
      <h6>Agent 3 – Health Advisor</h6>
      <p>Select your health conditions to receive tailored dietary and lifestyle guidance.</p>
      <a href="/health-advisor" class="btn btn-sm btn-nw mt-2">Get Advice</a>
    </div>
  </div>
  <div class="col-md-6">
    <div class="agent-card">
      <div class="icon-wrap" style="background:#fff3e0;color:#e65100;">
        <i class="bi bi-search-heart-fill"></i>
      </div>
      <h6>Agent 4 – Meal Analyzer</h6>
      <p>Log your meals in plain text and receive a detailed nutritional analysis and improvement tips.</p>
      <a href="/meal-analyzer" class="btn btn-sm btn-nw mt-2">Analyze Meals</a>
    </div>
  </div>
</div>

<div class="nw-card" style="background:linear-gradient(135deg,#f0faf3,#e8f5e9);border-color:#b8d4c0;">
  <h6 style="color:#1a6b3a;"><i class="bi bi-cpu me-2"></i>Powered by IBM watsonx.ai</h6>
  <p class="mb-0" style="font-size:.88rem;color:#3a5a44;">
    All AI responses are generated by <strong>IBM Granite 3.3 8B Instruct</strong> via the
    <strong>ibm-watsonx-ai</strong> Python SDK. The four-agent orchestrator routes each user request
    to the appropriate specialised agent, which builds a structured prompt and calls
    <code>generate_response()</code> — the single integration point with IBM watsonx.ai.
  </p>
</div>
"""

# -----------------------------------------------------------------------------
# Nutrition Chat Page
# -----------------------------------------------------------------------------
NUTRITION_CONTENT = """
<div class="nw-card">
  <h5><i class="bi bi-chat-dots-fill me-2"></i>Nutrition Knowledge Agent</h5>
  <p class="sub">Ask any nutrition or food-science question. Powered by IBM watsonx.ai Granite.</p>

  <div class="mb-3">
    <label class="form-label">Your nutrition question</label>
    <textarea id="question" class="form-control" rows="3"
      placeholder="e.g. What are the benefits of oats? Which foods are rich in Vitamin B12?"></textarea>
  </div>

  <div class="d-flex flex-wrap gap-2 mb-3">
    <span class="badge rounded-pill" style="background:#e8f5e9;color:#1a6b3a;cursor:pointer;
          border:1px solid #b8d4c0;padding:6px 12px;font-size:.8rem;"
      onclick="document.getElementById('question').value=this.textContent">
      Benefits of oats?
    </span>
    <span class="badge rounded-pill" style="background:#e8f5e9;color:#1a6b3a;cursor:pointer;
          border:1px solid #b8d4c0;padding:6px 12px;font-size:.8rem;"
      onclick="document.getElementById('question').value=this.textContent">
      Foods rich in protein?
    </span>
    <span class="badge rounded-pill" style="background:#e8f5e9;color:#1a6b3a;cursor:pointer;
          border:1px solid #b8d4c0;padding:6px 12px;font-size:.8rem;"
      onclick="document.getElementById('question').value=this.textContent">
      Is paneer healthy for muscle gain?
    </span>
    <span class="badge rounded-pill" style="background:#e8f5e9;color:#1a6b3a;cursor:pointer;
          border:1px solid #b8d4c0;padding:6px 12px;font-size:.8rem;"
      onclick="document.getElementById('question').value=this.textContent">
      What foods contain Vitamin B12?
    </span>
  </div>

  <button id="submit-btn" class="btn btn-nw" data-label="Ask NutriWise AI"
    onclick="askNutrition()">
    <i class="bi bi-send-fill me-2"></i>Ask NutriWise AI
  </button>
</div>

<div class="loader" id="loader">
  <div class="spinner-border"></div>
  <p>IBM Granite is thinking…</p>
</div>
<pre id="response-area"></pre>
"""

NUTRITION_JS = """
<script>
function askNutrition(){
  const q = document.getElementById('question').value.trim();
  if(!q){ alert('Please enter a question.'); return; }
  runAgent('/api/nutrition', {question: q}, 'response-area');
}
document.getElementById('question').addEventListener('keydown', function(e){
  if(e.ctrlKey && e.key==='Enter') askNutrition();
});
</script>"""

# -----------------------------------------------------------------------------
# Diet Planner Page
# -----------------------------------------------------------------------------
DIET_CONTENT = """
<div class="nw-card">
  <h5><i class="bi bi-calendar2-heart-fill me-2"></i>Diet Planner Agent</h5>
  <p class="sub">Enter your profile to receive a personalised one-day meal plan with macro targets.</p>

  <div class="row g-3">
    <div class="col-md-4">
      <label class="form-label">Age (years)</label>
      <input id="age" type="number" class="form-control" placeholder="e.g. 28" min="10" max="100"/>
    </div>
    <div class="col-md-4">
      <label class="form-label">Gender</label>
      <select id="gender" class="form-select">
        <option value="">Select gender</option>
        <option>Male</option>
        <option>Female</option>
        <option>Other</option>
      </select>
    </div>
    <div class="col-md-4">
      <label class="form-label">Height (cm)</label>
      <input id="height" type="number" class="form-control" placeholder="e.g. 170" min="100" max="250"/>
    </div>
    <div class="col-md-4">
      <label class="form-label">Weight (kg)</label>
      <input id="weight" type="number" class="form-control" placeholder="e.g. 70" min="30" max="300"/>
    </div>
    <div class="col-md-4">
      <label class="form-label">Dietary Preference</label>
      <select id="dietary_pref" class="form-select">
        <option value="">Select preference</option>
        <option>Vegetarian</option>
        <option>Vegan</option>
        <option>Non-Vegetarian</option>
        <option>Eggetarian</option>
        <option>Pescatarian</option>
        <option>Gluten-Free</option>
      </select>
    </div>
    <div class="col-md-4">
      <label class="form-label">Activity Level</label>
      <select id="activity_level" class="form-select">
        <option value="">Select activity level</option>
        <option>Sedentary (little or no exercise)</option>
        <option>Lightly Active (1-3 days/week)</option>
        <option>Moderately Active (3-5 days/week)</option>
        <option>Very Active (6-7 days/week)</option>
        <option>Extra Active (athlete/physical job)</option>
      </select>
    </div>
    <div class="col-12">
      <label class="form-label">Fitness Goal</label>
      <div class="d-flex flex-wrap gap-2" id="goal-group">
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="radio" name="goal" id="g1" value="Weight Loss">
          <label class="form-check-label" for="g1">⚖️ Weight Loss</label>
        </div>
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="radio" name="goal" id="g2" value="Weight Gain">
          <label class="form-check-label" for="g2">📈 Weight Gain</label>
        </div>
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="radio" name="goal" id="g3" value="Muscle Gain">
          <label class="form-check-label" for="g3">💪 Muscle Gain</label>
        </div>
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="radio" name="goal" id="g4" value="General Wellness">
          <label class="form-check-label" for="g4">🌿 General Wellness</label>
        </div>
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="radio" name="goal" id="g5" value="Maintain Weight">
          <label class="form-check-label" for="g5">🎯 Maintain Weight</label>
        </div>
      </div>
    </div>
  </div>

  <button id="submit-btn" class="btn btn-nw mt-4" data-label="<i class='bi bi-calendar2-plus me-2'></i>Generate My Meal Plan"
    onclick="generateDietPlan()">
    <i class="bi bi-calendar2-plus me-2"></i>Generate My Meal Plan
  </button>
</div>

<div class="loader" id="loader">
  <div class="spinner-border"></div>
  <p>IBM Granite is crafting your personalised meal plan…</p>
</div>
<pre id="response-area"></pre>
"""

DIET_JS = """
<script>
function generateDietPlan(){
  const age           = document.getElementById('age').value.trim();
  const gender        = document.getElementById('gender').value;
  const height        = document.getElementById('height').value.trim();
  const weight        = document.getElementById('weight').value.trim();
  const dietary_pref  = document.getElementById('dietary_pref').value;
  const activity_level= document.getElementById('activity_level').value;
  const goalEl        = document.querySelector('input[name="goal"]:checked');

  if(!age || !gender || !height || !weight || !dietary_pref || !activity_level || !goalEl){
    alert('Please fill in all fields and select a fitness goal.'); return;
  }
  runAgent('/api/diet', {
    age, gender, height, weight,
    dietary_pref, activity_level,
    fitness_goal: goalEl.value
  }, 'response-area');
}
</script>"""

# -----------------------------------------------------------------------------
# Health Advisor Page
# -----------------------------------------------------------------------------
HEALTH_CONTENT = """
<div class="nw-card">
  <h5><i class="bi bi-heart-pulse-fill me-2"></i>Health Advisory Agent</h5>
  <p class="sub">Select your health conditions to receive evidence-based dietary and lifestyle recommendations.</p>

  <label class="form-label mb-3">Select your health condition(s):</label>
  <div class="condition-grid mb-4">
    <label class="condition-item">
      <input type="checkbox" value="Type 2 Diabetes"/>
      <span>🩸 Diabetes</span>
    </label>
    <label class="condition-item">
      <input type="checkbox" value="Hypertension (High Blood Pressure)"/>
      <span>💊 Hypertension</span>
    </label>
    <label class="condition-item">
      <input type="checkbox" value="Obesity"/>
      <span>⚖️ Obesity</span>
    </label>
    <label class="condition-item">
      <input type="checkbox" value="Heart Disease"/>
      <span>❤️ Heart Disease</span>
    </label>
    <label class="condition-item">
      <input type="checkbox" value="PCOS (Polycystic Ovary Syndrome)"/>
      <span>🔬 PCOS</span>
    </label>
    <label class="condition-item">
      <input type="checkbox" value="High Cholesterol"/>
      <span>🧬 High Cholesterol</span>
    </label>
    <label class="condition-item">
      <input type="checkbox" value="Thyroid Disorder"/>
      <span>🦋 Thyroid Disorder</span>
    </label>
    <label class="condition-item">
      <input type="checkbox" value="Anaemia"/>
      <span>💉 Anaemia</span>
    </label>
    <label class="condition-item">
      <input type="checkbox" value="Irritable Bowel Syndrome (IBS)"/>
      <span>🌿 IBS</span>
    </label>
    <label class="condition-item">
      <input type="checkbox" value="Osteoporosis"/>
      <span>🦴 Osteoporosis</span>
    </label>
  </div>

  <div class="alert" style="background:#fff8e1;border:1px solid #ffe082;border-radius:8px;
       padding:10px 16px;font-size:.83rem;color:#7a5c00;margin-bottom:18px;">
    <i class="bi bi-exclamation-triangle-fill me-2"></i>
    <strong>Disclaimer:</strong> Information provided is for educational purposes only.
    Always consult a qualified healthcare professional for medical advice.
  </div>

  <button id="submit-btn" class="btn btn-nw" data-label="<i class='bi bi-heart-pulse me-2'></i>Get Health Recommendations"
    onclick="getHealthAdvice()">
    <i class="bi bi-heart-pulse me-2"></i>Get Health Recommendations
  </button>
</div>

<div class="loader" id="loader">
  <div class="spinner-border"></div>
  <p>IBM Granite is preparing your health advisory…</p>
</div>
<pre id="response-area"></pre>
"""

HEALTH_JS = """
<script>
function getHealthAdvice(){
  const checked = Array.from(document.querySelectorAll('.condition-item input:checked'))
                       .map(el => el.value);
  if(checked.length === 0){ alert('Please select at least one health condition.'); return; }
  runAgent('/api/health', {conditions: checked}, 'response-area');
}
</script>"""

# -----------------------------------------------------------------------------
# Meal Analyzer Page
# -----------------------------------------------------------------------------
MEAL_CONTENT = """
<div class="nw-card">
  <h5><i class="bi bi-search-heart-fill me-2"></i>Meal Analysis Agent</h5>
  <p class="sub">Enter what you ate today (free text). IBM Granite will analyse your nutritional intake.</p>

  <label class="form-label">Your meal log</label>
  <textarea id="meal_log" class="form-control" rows="9"
    placeholder="Breakfast:&#10;2 rotis&#10;1 bowl dal&#10;&#10;Lunch:&#10;1 cup rice&#10;Paneer curry&#10;Salad&#10;&#10;Dinner:&#10;Vegetable soup&#10;2 chapatis&#10;Curd"></textarea>

  <div class="d-flex flex-wrap gap-2 mt-3 mb-3">
    <small style="color:#5c7a66;align-self:center;">Sample meals:</small>
    <span class="badge rounded-pill" style="background:#e8f5e9;color:#1a6b3a;cursor:pointer;
          border:1px solid #b8d4c0;padding:6px 12px;font-size:.79rem;"
      onclick="loadSample1()">Indian Vegetarian Day</span>
    <span class="badge rounded-pill" style="background:#e8f5e9;color:#1a6b3a;cursor:pointer;
          border:1px solid #b8d4c0;padding:6px 12px;font-size:.79rem;"
      onclick="loadSample2()">High Protein Day</span>
    <span class="badge rounded-pill" style="background:#e8f5e9;color:#1a6b3a;cursor:pointer;
          border:1px solid #b8d4c0;padding:6px 12px;font-size:.79rem;"
      onclick="loadSample3()">Junk Food Day</span>
  </div>

  <button id="submit-btn" class="btn btn-nw" data-label="<i class='bi bi-bar-chart-fill me-2'></i>Analyse My Meals"
    onclick="analyzeMeals()">
    <i class="bi bi-bar-chart-fill me-2"></i>Analyse My Meals
  </button>
</div>

<div class="loader" id="loader">
  <div class="spinner-border"></div>
  <p>IBM Granite is analysing your nutritional intake…</p>
</div>
<pre id="response-area"></pre>
"""

MEAL_JS = """
<script>
function analyzeMeals(){
  const log = document.getElementById('meal_log').value.trim();
  if(!log){ alert('Please enter your meal log.'); return; }
  runAgent('/api/meal', {meal_log: log}, 'response-area');
}
function loadSample1(){
  document.getElementById('meal_log').value =
    "Breakfast:\\n2 rotis with ghee\\n1 bowl poha\\nTea with sugar\\n\\n" +
    "Lunch:\\n1 cup rice\\nDal tadka\\nAloo sabzi\\nSalad\\n\\n" +
    "Snacks:\\nSamosa (2)\\nMasala chai\\n\\n" +
    "Dinner:\\n2 chapatis\\nPaneer butter masala\\nCurd (1 cup)";
}
function loadSample2(){
  document.getElementById('meal_log').value =
    "Breakfast:\\n4 boiled eggs\\nOats porridge\\nMilk 1 glass\\n\\n" +
    "Lunch:\\nGrilled chicken breast 150g\\nBrown rice 1 cup\\nSteamed broccoli\\n\\n" +
    "Snacks:\\nProtein shake\\nBanana\\nAlmonds (handful)\\n\\n" +
    "Dinner:\\nGrilled fish 150g\\nSweet potato\\nSpinach salad";
}
function loadSample3(){
  document.getElementById('meal_log').value =
    "Breakfast:\\nSkipped breakfast\\nCoffee with sugar\\n\\n" +
    "Lunch:\\n2 slices pizza\\nCola (500ml)\\n\\n" +
    "Snacks:\\nChips (large packet)\\nChocolate bar\\n\\n" +
    "Dinner:\\nBurger\\nFries\\nAerated drink";
}
</script>"""

# -----------------------------------------------------------------------------
# About Page
# -----------------------------------------------------------------------------
ABOUT_CONTENT = """
<div class="nw-card">
  <h5><i class="bi bi-info-circle-fill me-2"></i>About NutriWise AI</h5>
  <p class="sub">Architecture overview — multi-agent system powered by IBM watsonx.ai</p>
  <p style="font-size:.9rem;color:#3a5a44;">
    NutriWise AI is a demonstration-grade agentic AI application built for IBM hackathons,
    IBM SkillsBuild showcases, and college AI presentations. It uses a four-agent orchestrator
    pattern on top of IBM Granite large language models to deliver specialised nutrition intelligence.
  </p>
</div>

<div class="row g-3 mb-4">
  <div class="col-md-6">
    <div class="nw-card h-100">
      <h6 style="color:#1a6b3a;"><i class="bi bi-diagram-3-fill me-2"></i>Technology Stack</h6>
      <ul style="font-size:.87rem;color:#3a5a44;padding-left:18px;">
        <li><strong>AI Engine:</strong> IBM watsonx.ai – Granite 3.3 8B Instruct</li>
        <li><strong>SDK:</strong> ibm-watsonx-ai Python SDK</li>
        <li><strong>Backend:</strong> Python 3.10+ / Flask</li>
        <li><strong>Frontend:</strong> Bootstrap 5.3 + Vanilla JS</li>
        <li><strong>Icons:</strong> Bootstrap Icons 1.11</li>
        <li><strong>Deployment:</strong> Single file – app.py</li>
      </ul>
    </div>
  </div>
  <div class="col-md-6">
    <div class="nw-card h-100">
      <h6 style="color:#1a6b3a;"><i class="bi bi-cpu me-2"></i>IBM watsonx.ai Integration</h6>
      <ul style="font-size:.87rem;color:#3a5a44;padding-left:18px;">
        <li>Credentials loaded from environment variables</li>
        <li>Single <code>generate_response(prompt)</code> function</li>
        <li>All four agents call this function</li>
        <li>Model: <code>ibm/granite-3-3-8b-instruct</code></li>
        <li>Parameters: 1024 max tokens, temp 0.7, top-p 0.9</li>
        <li>Orchestrator routes requests by agent type</li>
      </ul>
    </div>
  </div>
</div>

<h6 class="mb-3 fw-600" style="color:#1a6b3a;">
  <i class="bi bi-arrow-right-circle-fill me-2"></i>Agent Architecture
</h6>
<div class="timeline">
  <div class="timeline-item">
    <h6>Agent 1 – Nutrition Knowledge Agent</h6>
    <p>Answers general nutrition and food-science questions. Builds a structured educational prompt
       and sends it to IBM Granite. Returns evidence-based responses with bullet points and tips.</p>
  </div>
  <div class="timeline-item">
    <h6>Agent 2 – Diet Planner Agent</h6>
    <p>Collects user profile data (age, gender, height, weight, dietary preference, activity level,
       fitness goal) and generates a complete one-day meal plan with daily calorie and macro targets.</p>
  </div>
  <div class="timeline-item">
    <h6>Agent 3 – Health Advisory Agent</h6>
    <p>Accepts one or more health conditions (Diabetes, Hypertension, Obesity, etc.) and generates
       foods-to-include, foods-to-avoid, healthy habits, and lifestyle recommendations. Always
       appends the mandatory medical disclaimer.</p>
  </div>
  <div class="timeline-item">
    <h6>Agent 4 – Meal Analysis Agent</h6>
    <p>Accepts free-text meal logs and returns a full nutritional analysis: macro estimates,
       micronutrient gaps, strengths, concerns, healthier alternatives, and actionable improvements.</p>
  </div>
  <div class="timeline-item">
    <h6>Orchestrator</h6>
    <p>The <code>orchestrator(agent_type, data)</code> function receives the request type and
       payload from Flask API endpoints, then routes to the correct agent function.</p>
  </div>
</div>

<div class="nw-card mt-4" style="background:#f0faf3;border-color:#b8d4c0;">
  <h6 style="color:#1a6b3a;"><i class="bi bi-terminal-fill me-2"></i>Quick Start</h6>
  <pre style="background:#1e2a22;color:#6ee09a;border-radius:8px;padding:16px;
              font-size:.83rem;margin:0;overflow-x:auto;">
pip install flask ibm-watsonx-ai

set WATSONX_API_KEY=&lt;your-key&gt;
set WATSONX_PROJECT_ID=&lt;your-project-id&gt;
set WATSONX_URL=https://us-south.ml.cloud.ibm.com

python app.py</pre>
</div>
"""


# =============================================================================
# Helper – render a page using the base layout
# =============================================================================
def render_page(title: str, icon: str, active: str, content: str, extra_js: str = ""):
    return render_template_string(
        BASE_HTML,
        title=title,
        icon=icon,
        active=active,
        content=content,
        extra_js=extra_js,
    )


# =============================================================================
# Flask Routes – UI Pages
# =============================================================================

@app.route("/")
def home():
    return render_page("Home", "stars", "home", HOME_CONTENT)


@app.route("/nutrition")
def nutrition():
    return render_page("Nutrition Chat", "chat-dots-fill", "nutrition",
                       NUTRITION_CONTENT, NUTRITION_JS)


@app.route("/diet-planner")
def diet_planner():
    return render_page("Diet Planner", "calendar2-heart-fill", "diet",
                       DIET_CONTENT, DIET_JS)


@app.route("/health-advisor")
def health_advisor():
    return render_page("Health Advisor", "heart-pulse-fill", "health",
                       HEALTH_CONTENT, HEALTH_JS)


@app.route("/meal-analyzer")
def meal_analyzer():
    return render_page("Meal Analyzer", "search-heart-fill", "meal",
                       MEAL_CONTENT, MEAL_JS)


@app.route("/about")
def about():
    return render_page("About", "info-circle-fill", "about", ABOUT_CONTENT)


# =============================================================================
# Flask Routes – AI API Endpoints
# Each endpoint receives JSON, calls the orchestrator, returns JSON.
# =============================================================================

@app.route("/api/nutrition", methods=["POST"])
def api_nutrition():
    """
    API endpoint for Agent 1 – Nutrition Knowledge Agent.
    Expects JSON: { "question": "..." }
    Returns JSON:  { "response": "..." }
    """
    data = request.get_json(force=True)
    if not data or not data.get("question"):
        return jsonify({"error": "Please provide a 'question' field."}), 400

    # Route to Agent 1 via orchestrator
    result = orchestrator("nutrition", data)
    return jsonify({"response": result})


@app.route("/api/diet", methods=["POST"])
def api_diet():
    """
    API endpoint for Agent 2 – Diet Planner Agent.
    Expects JSON with user profile fields.
    Returns JSON:  { "response": "..." }
    """
    data = request.get_json(force=True)
    required = ["age", "gender", "height", "weight", "dietary_pref",
                "activity_level", "fitness_goal"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # Route to Agent 2 via orchestrator
    result = orchestrator("diet", data)
    return jsonify({"response": result})


@app.route("/api/health", methods=["POST"])
def api_health():
    """
    API endpoint for Agent 3 – Health Advisory Agent.
    Expects JSON: { "conditions": ["Diabetes", "Hypertension", ...] }
    Returns JSON:  { "response": "..." }
    """
    data = request.get_json(force=True)
    if not data or not data.get("conditions"):
        return jsonify({"error": "Please provide at least one health condition."}), 400

    # Route to Agent 3 via orchestrator
    result = orchestrator("health", data)
    return jsonify({"response": result})


@app.route("/api/meal", methods=["POST"])
def api_meal():
    """
    API endpoint for Agent 4 – Meal Analysis Agent.
    Expects JSON: { "meal_log": "Breakfast: ..." }
    Returns JSON:  { "response": "..." }
    """
    data = request.get_json(force=True)
    if not data or not data.get("meal_log"):
        return jsonify({"error": "Please provide a 'meal_log' field."}), 400

    # Route to Agent 4 via orchestrator
    result = orchestrator("meal", data)
    return jsonify({"response": result})


# =============================================================================
# Application Entry Point
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  NutriWise AI – Personalized Nutrition Coach")
    print("  Powered by IBM watsonx.ai Granite Models")
    print("=" * 60)
    print(f"  Model     : {GRANITE_MODEL_ID}")
    print(f"  API Key   : {'[OK] Set' if WATSONX_API_KEY != 'your-api-key-here' else '[!] Not set (set WATSONX_API_KEY)'}")
    print(f"  Project ID: {'[OK] Set' if WATSONX_PROJECT_ID != 'your-project-id-here' else '[!] Not set (set WATSONX_PROJECT_ID)'}")
    print(f"  URL       : {WATSONX_URL}")
    print("=" * 60)
    print("  Visit: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
