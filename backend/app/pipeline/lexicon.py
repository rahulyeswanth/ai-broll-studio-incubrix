"""Activity lexicon: what a clip's title/tags must contain to *show* an action, not merely mention its people.

Why this exists
---------------
Stock-footage metadata is a few words. A clip tagged ``doctor, patient, hospital`` says who and where, but not
whether anyone is measuring blood pressure. Plain keyword coverage lets such clips pass. Each ``ActivityFrame``
therefore declares, for a family of activities:

* ``action_terms``  - words/phrases that indicate the *action* is shown ("watering", "irrigation", "sprinkler");
* ``object_terms``  - words/phrases that indicate the *key object* is present ("blood pressure", "cuff");
* ``foil_terms``    - related-but-different actions that must NOT be mistaken for it ("harvesting" vs "watering");
* ``queries``       - alternative retrieval phrasings (<= 8 words) to try when the first query disappoints.

This is a curated, deterministic, English-only resource. It is a general library covering many everyday activities
(farming, healthcare, software, teaching, science, travel, cooking, sport, construction, business, ...), not a table
of the sample script. Scenes that match no frame fall back to the verb/objects found by the rule-based analyser.
Extend ``FRAMES`` (or pass your own tuple to ``match_frames``) to cover your domain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class ActivityFrame:
    id: str
    triggers: Tuple[str, ...]                 # regexes; ANY must match the lower-cased narration
    action_terms: Tuple[str, ...]
    object_terms: Tuple[str, ...] = ()
    foil_terms: Tuple[str, ...] = ()
    setting_terms: Tuple[str, ...] = ()
    queries: Tuple[str, ...] = ()
    negative_terms: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()            # regexes; ALL must also match (disambiguation)
    gate: bool = True                         # False: improves queries/scoring but never caps a clip (noun-triggered, generic frames)


def _f(id, triggers, action, obj=(), foil=(), setting=(), queries=(), neg=(), requires=()):
    return ActivityFrame(id, tuple(triggers), tuple(action), tuple(obj), tuple(foil), tuple(setting),
                         tuple(queries), tuple(neg), tuple(requires))


FRAMES: Tuple[ActivityFrame, ...] = (
    # ------------------------------------------------------------------ agriculture
    _f("watering_plants", [r"\bwater(s|ed|ing)?\b.*\b(plant|plants|crop|crops|tomato|tomatoes|garden|seedling|seedlings|field|vegetable|vegetables)\b",
                           r"\birrigat\w*\b", r"\bsprinkl\w*\b"],
       action=["watering", "irrigation", "irrigating", "sprinkler", "sprinkling", "hose", "watering can", "drip irrigation", "spraying water"],
       obj=["hose", "watering can", "sprinkler", "irrigation system"],
       foil=["harvesting", "harvest", "picking", "plowing", "ploughing", "tractor", "combine", "planting", "sowing", "pruning", "market"],
       setting=["field", "farm", "garden", "greenhouse", "agriculture"],
       queries=["farmer watering plants in field", "watering tomato plants garden hose", "irrigation sprinkler crops farm",
                "person watering vegetable garden"],
       neg=["harvesting", "market"]),
    _f("harvesting_crops", [r"\bharvest\w*\b", r"\bpick(s|ed|ing)?\b.*\b(fruit|fruits|apples|tomatoes|crops|vegetables|berries)\b"],
       action=["harvesting", "harvest", "picking", "reaping", "combine harvester", "gathering crops"],
       obj=["combine harvester", "harvester", "basket", "crates"],
       foil=["watering", "irrigation", "sprinkler", "planting", "sowing"],
       setting=["field", "farm", "orchard"],
       queries=["farmer harvesting crops in field", "combine harvester wheat field", "picking fruit orchard"]),
    _f("planting_seeds", [r"\b(plant|plants|planting|sow|sows|sowing)\b.*\b(seed|seeds|seedlings|trees|tree|crops)\b"],
       action=["planting", "sowing", "seeding", "transplanting", "seed drill"],
       obj=["seed drill", "seed tray", "trowel"],
       foil=["harvesting", "watering"], setting=["field", "farm", "garden"],
       queries=["farmer planting seeds in soil", "sowing seeds field", "planting seedlings garden"]),
    # ------------------------------------------------------------------ healthcare
    _f("measuring_blood_pressure", [r"blood\s+pressure", r"sphygmo\w*", r"\bhypertension\b.*\b(check|measur|monitor)\w*"],
       action=["blood pressure", "measuring blood pressure", "checking blood pressure", "sphygmomanometer", "bp monitor", "taking blood pressure"],
       obj=["blood pressure", "cuff", "sphygmomanometer", "blood pressure cuff", "bp monitor", "tonometer"],
       foil=["consultation", "talking", "surgery", "surgeon", "x-ray", "corridor", "ambulance", "vaccination", "injection", "reception", "waiting room"],
       setting=["clinic", "hospital", "doctor's office", "medical"],
       queries=["doctor measuring patient blood pressure cuff", "nurse checking blood pressure in clinic",
                "medical professional using blood pressure monitor", "blood pressure measurement arm cuff"],
       neg=["corridor", "ambulance"]),
    _f("patient_examination", [r"\b(examin\w*|check(s|ed|ing)?|consult\w*)\b.*\b(patient|patients)\b"],
       action=["examining", "examination", "checkup", "check-up", "consultation", "diagnosing", "stethoscope", "auscultation"],
       obj=["patient", "stethoscope", "medical examination"],
       foil=["surgery", "ambulance", "reception"], setting=["clinic", "hospital", "doctor's office"],
       queries=["doctor examining patient in clinic", "physician checkup patient stethoscope"]),
    _f("vaccination", [r"\bvaccin\w*\b", r"\binject\w*\b"],
       action=["vaccination", "vaccinating", "injection", "injecting", "vaccine", "jab", "shot"],
       obj=["syringe", "vaccine", "needle"], foil=["blood pressure", "surgery"], setting=["clinic", "hospital"],
       queries=["nurse giving vaccine injection", "vaccination syringe arm"]),
    # ------------------------------------------------------------------ technology
    _f("writing_code", [r"\b(writ(es|ing|e)|typ(es|ing|e)|develop\w*|program\w*|cod(es|ing|e))\b.*\b(code|python|software|program|programs|script|app|website|javascript|java)\b",
                        r"\b(software|python|javascript)\s+(engineer|developer|programmer)\b", r"\bprogrammer\b", r"\bcoding\b"],
       action=["coding", "programming", "writing code", "typing code", "software development", "developer", "programmer", "typing", "debugging"],
       obj=["code", "source code", "python", "software", "code editor", "programming", "javascript"],
       foil=["meeting", "presentation", "whiteboard", "phone", "video call", "server room", "data center", "circuit", "hardware"],
       # a laptop/screen only shows the *place* of the work, not that code is being written -> setting evidence only
       setting=["office", "desk", "workspace", "home office", "laptop", "computer", "screen", "monitor", "keyboard"],
       queries=["software engineer typing code on laptop", "programmer writing python code screen",
                "developer coding at desk monitor", "typing on keyboard source code screen"],
       neg=["server room"]),
    _f("using_computer", [r"\b(laptop|computer|keyboard)\b"],
       action=["typing", "working", "browsing", "using computer"], obj=["laptop", "computer", "keyboard", "screen"],
       foil=["meeting", "phone"], setting=["office", "desk"],
       queries=["person typing on laptop at desk"]),
    # ------------------------------------------------------------------ education / science
    _f("science_experiment", [r"\bexperiment\w*\b", r"\bchemistry\b", r"\bchemical\s+reaction\b", r"\blab(oratory)?\s+(demonstration|class|experiment)\b",
                              r"\bscience\s+(demonstration|demo|class|lesson)\b"],
       action=["experiment", "experiments", "experimenting", "science experiment", "chemistry experiment", "science demonstration",
               "chemical reaction", "mixing chemicals"],
       obj=["beaker", "test tube", "flask", "chemicals", "laboratory", "microscope", "lab coat", "reaction", "bunsen burner", "pipette"],
       foil=["blackboard", "chalkboard", "whiteboard", "lecture", "writing", "exam", "test paper", "reading", "textbook", "graduation"],
       setting=["classroom", "laboratory", "school", "lab"],
       queries=["teacher performing science experiment for students", "chemistry experiment classroom beaker",
                "students watching science demonstration", "laboratory experiment test tubes"],
       neg=["blackboard"]),
    _f("teaching_class", [r"\b(teach(es|ing)?|explain(s|ed|ing)?|lectur\w*)\b", r"\bclassroom\b"],
       action=["teaching", "teacher", "lecture", "explaining", "lesson", "instructor", "classroom"],
       obj=["students", "blackboard", "whiteboard", "classroom", "chalkboard", "lesson"],
       foil=["exam", "graduation"], setting=["classroom", "school"],
       queries=["teacher explaining lesson to students classroom", "teacher teaching class whiteboard"]),
    _f("studying_reading", [r"\b(study|studies|studying|read|reads|reading)\b"],
       action=["studying", "reading", "learning", "homework", "revising"], obj=["book", "books", "notes", "library", "textbook", "desk"],
       foil=["exam hall"], setting=["library", "desk", "school"],
       queries=["student studying with books at desk", "person reading book library"]),
    # ------------------------------------------------------------------ travel
    _f("photographing_landmark", [r"\b(photograph\w*|take\w*\s+(a\s+)?(photo|photos|picture|pictures|selfie)|snap\w*|film\w*)\b.*\b(tower|landmark|monument|cathedral|bridge|temple|statue|museum|palace|mountain)\b",
                                  r"\btourist\w*\b.*\b(photo|photograph|camera|picture)\w*\b", r"\b(photograph\w*|taking\s+photos?)\b.*\b(eiffel|colosseum|taj\s+mahal|big\s+ben|statue\s+of\s+liberty)\b"],
       action=["taking photo", "taking photos", "taking pictures", "taking picture", "photographing", "photographer", "selfie", "camera", "smartphone photo",
               "photography", "taking a photo", "tourist taking"],
       obj=["camera", "smartphone", "phone", "selfie", "selfie stick", "tripod"],
       foil=["aerial", "drone", "time lapse", "timelapse", "night", "lights", "fireworks", "skyline", "sunset", "fountain", "timelapse"],
       setting=["landmark", "tower", "monument", "sightseeing", "travel"],
       queries=["tourist taking photo of famous tower", "tourist photographing landmark with camera",
                "traveler taking pictures sightseeing smartphone", "tourists taking selfie monument"],
       neg=["aerial", "drone", "timelapse"]),
    _f("sightseeing", [r"\b(tourist|tourists|traveler|traveller|travel|sightseeing|visit(s|ing)?)\b"],
       action=["sightseeing", "tourist", "tourists", "traveling", "travelling", "walking", "exploring", "visiting"],
       obj=["landmark", "luggage", "map", "backpack"], foil=["airport"], setting=["city", "street"],
       queries=["tourists sightseeing in city", "traveler exploring old town"]),
    # ------------------------------------------------------------------ everyday / other professions
    _f("cooking", [r"\b(cook|cooks|cooking|chef|chop|chops|chopping|bake|bakes|baking|stir|stirs|stirring|fry|fries|frying)\b"],
       action=["cooking", "chopping", "stirring", "frying", "baking", "chef", "preparing food", "sauteing", "boiling"],
       obj=["pan", "knife", "ingredients", "kitchen", "vegetables", "pot", "dough", "oven"],
       foil=["eating", "restaurant dining", "grocery", "market"], setting=["kitchen", "restaurant"],
       queries=["chef cooking in restaurant kitchen", "person chopping vegetables kitchen", "stirring pot stove"]),
    _f("construction_work", [r"\b(construct\w*|build(s|ing|er|ers)?|weld\w*|bricklay\w*|scaffold\w*)\b", r"\bconstruction\s+(site|worker|workers)\b"],
       action=["construction", "building", "welding", "bricklaying", "working", "crane", "excavator"],
       obj=["bricks", "scaffolding", "crane", "helmet", "concrete", "steel", "excavator", "tools"],
       foil=["architecture model", "blueprint only"], setting=["construction site", "building site"],
       queries=["construction workers building on site", "welder welding metal sparks", "crane lifting building site"]),
    _f("running_exercise", [r"\b(run|runs|running|jog|jogs|jogging|exercis\w*|workout|train(s|ing)?|gym)\b"],
       action=["running", "jogging", "exercise", "exercising", "workout", "training", "sprinting", "gym", "fitness"],
       obj=["treadmill", "weights", "dumbbells", "track", "trail"], foil=["walking", "standing", "sitting"],
       setting=["gym", "park", "track", "trail"],
       queries=["person jogging in park", "athlete running on track", "workout gym weights"]),
    _f("business_meeting", [r"\b(meeting|negotiat\w*|present(s|ed|ing|ation)?|conference|board\s*room|team\s+discuss\w*)\b"],
       action=["meeting", "presentation", "presenting", "discussion", "negotiation", "conference", "brainstorming", "teamwork"],
       obj=["whiteboard", "conference table", "laptop", "projector", "slides", "boardroom", "documents"],
       foil=["interview", "phone call"], setting=["office", "boardroom", "conference room"],
       queries=["business team meeting in office", "person presenting to colleagues boardroom"]),
    _f("driving_vehicle", [r"\b(driv(e|es|ing)|car|truck|traffic|highway|road\s+trip)\b"],
       action=["driving", "traffic", "highway", "road trip", "car", "truck", "driver"],
       obj=["car", "steering wheel", "road", "highway", "traffic"], foil=["parked", "car wash", "repair", "mechanic"],
       setting=["road", "highway", "city street"],
       queries=["car driving on highway", "driver steering wheel road", "city traffic cars"]),
    _f("shopping", [r"\b(shop|shops|shopping|shopper|shoppers|buy|buys|buying|supermarket|market)\b"],
       action=["shopping", "buying", "shoppers", "customers", "browsing", "checkout"],
       obj=["cart", "basket", "store", "supermarket", "market", "shelves", "mall"], foil=["online"],
       setting=["store", "supermarket", "mall", "market"],
       queries=["people shopping in supermarket", "customer buying at market stall"]),
    _f("wildlife_ocean", [r"\b(turtle|turtles|dolphin|dolphins|whale|whales|shark|sharks|fish)\b.*\b(swim|swims|swimming|dive|dives|diving|glide|glides|gliding)\b"],
       action=["swimming", "swim", "diving", "underwater swimming", "gliding underwater"],
       obj=[], foil=["aquarium", "beach", "boat", "cartoon"],
       # the ocean/sea is the PLACE: it narrows the search but is not proof that the animal is shown or swimming
       setting=["ocean", "sea", "reef", "underwater", "marine"],
       queries=["sea turtle swimming underwater", "coral reef fish underwater"]),
    _f("playing_music", [r"\b(play|plays|playing|perform\w*|sing|sings|singing|concert|guitar|piano|violin|drum|drums)\b.*\b(music|song|guitar|piano|violin|drums|concert|instrument)\b",
                         r"\b(musician|musicians|band|orchestra)\b"],
       action=["playing", "performing", "singing", "concert", "playing music", "musician", "band", "orchestra"],
       obj=["guitar", "piano", "violin", "drums", "microphone", "instrument", "stage"], foil=["listening", "headphones"],
       setting=["stage", "studio", "concert"],
       queries=["musician playing guitar on stage", "pianist playing piano close up"]),
    _f("typing_writing", [r"\b(writ(e|es|ing)|journal|notebook|pen)\b"],
       action=["writing", "handwriting", "notebook", "journaling", "pen"], obj=["pen", "notebook", "paper", "desk"],
       foil=["typing", "keyboard"], setting=["desk", "office"],
       queries=["person writing in notebook with pen"]),
)

# Generic frames triggered by a bare noun/role ("laptop", "car", "classroom"): a static clip of that thing is legitimate B-roll,
# so they inform queries and scoring but must never hard-gate.
_NON_GATING = {"using_computer", "sightseeing", "shopping", "driving_vehicle", "teaching_class", "studying_reading", "playing_music",
               "construction_work", "business_meeting", "typing_writing", "running_exercise", "cooking"}
FRAMES = tuple(replace(f, gate=False) if f.id in _NON_GATING else f for f in FRAMES)

# role synonyms: what stock-footage titles/tags tend to call the same person
SUBJECT_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "doctor": ("physician", "doctor", "medical professional", "gp", "clinician", "nurse"),
    "nurse": ("nurse", "medical professional", "doctor", "healthcare worker"),
    "farmer": ("farmer", "farm worker", "gardener", "agricultural worker", "grower"),
    "engineer": ("engineer", "developer", "programmer", "software", "coder", "it professional"),
    "developer": ("developer", "programmer", "software engineer", "coder"),
    "programmer": ("programmer", "developer", "coder", "software engineer"),
    "software engineer": ("software engineer", "developer", "programmer", "coder", "engineer"),
    "teacher": ("teacher", "instructor", "professor", "educator", "tutor", "lecturer"),
    "tourist": ("tourist", "tourists", "traveler", "traveller", "visitor", "sightseer", "backpacker"),
    "chef": ("chef", "cook", "kitchen staff"),
    "student": ("student", "students", "pupil", "pupils", "learner", "school kids"),
    "scientist": ("scientist", "researcher", "chemist", "lab technician"),
    "athlete": ("athlete", "runner", "sportsman", "sportswoman"),
}

_STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "with", "by", "his", "her", "their"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' -]+", " ", (text or "").lower())).strip()


def match_frames(narration: str, frames: Sequence[ActivityFrame] = FRAMES, limit: int = 2) -> List[ActivityFrame]:
    """Frames whose triggers match the narration, in library order (specific frames are listed before generic ones).

    At most ``limit`` frames are returned so that one sentence cannot import unrelated evidence."""
    text = (narration or "").lower()
    out: List[ActivityFrame] = []
    for fr in frames:
        if fr.requires and not all(re.search(r, text) for r in fr.requires):
            continue
        if any(re.search(trig, text) for trig in fr.triggers):
            out.append(fr)
            if len(out) >= limit:
                break
    return out


def subject_synonyms(subject: str) -> List[str]:
    """Subject words plus role synonyms (``doctor`` -> physician, medical professional, ...)."""
    s = _norm(subject)
    out: List[str] = []
    if not s:
        return out
    words = [w for w in s.split() if w not in _STOP]
    if s in SUBJECT_SYNONYMS:
        out += list(SUBJECT_SYNONYMS[s])
    for w in words:
        out += list(SUBJECT_SYNONYMS.get(w, ()))
        out.append(w)
    out.append(s)
    return list(dict.fromkeys(x for x in out if x))
