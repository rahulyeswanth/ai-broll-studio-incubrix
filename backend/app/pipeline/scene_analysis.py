"""Scene understanding: narration text -> structured visual description + ordered search queries.

Default implementation is a transparent, deterministic, English-only rule-based analyser
(``RuleBasedSceneAnalyzer``). It is *not* an LLM. A replaceable ``SceneAnalyzer`` interface and an
``LLMSceneAnalyzer`` (which takes any ``complete(prompt) -> str`` callable) are provided so a stronger
language model can be plugged in without touching the rest of the pipeline.

The original narration text is never modified; the visual description / queries are separate fields.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

from app.services.query_extractor import COMMON_STOP_WORDS
from .safety import content_words as _content_words, entity_standins, sensitive_profile
from .lexicon import match_frames, subject_synonyms
from .models import SceneAnalysis

# ------------------------------------------------------------------ lexicons
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "our", "your", "their", "its", "his", "her",
               "my", "some", "any", "each", "every", "many", "much", "several", "all", "both", "most"}
PRONOUNS = {"i", "we", "you", "he", "she", "it", "they", "there", "here", "who", "which", "that", "one"}
AUXILIARIES = {"is", "are", "was", "were", "be", "been", "being", "am", "has", "have", "had", "will", "would",
               "can", "could", "may", "might", "must", "should", "do", "does", "did"}
SUBORDINATORS = {"after", "before", "while", "during", "because", "when", "as", "since", "although", "though",
                 "until", "once", "if", "which", "whereas", "following"}
SETTING_PREPS = {"in", "at", "on", "inside", "across", "over", "through", "under", "near", "along", "around",
                 "beside", "within", "beneath", "above", "outside", "throughout", "into", "toward", "towards"}
MOTION_PREPS = {"through", "across", "over", "along", "under", "around", "above", "beneath", "toward", "towards"}
ADVERBS_DROP = {"carefully", "slowly", "quickly", "often", "always", "never", "also", "still", "just", "now",
                "then", "suddenly", "finally", "really", "very", "together", "again", "already", "soon"}
ABSTRACT_WORDS = {"important", "essential", "crucial", "vital", "key", "future", "challenge", "challenges",
                  "ways", "way", "thing", "things", "idea", "ideas", "reason", "reasons", "today", "welcome",
                  "overview", "introduction", "better", "best", "great", "good", "amazing", "incredible",
                  "faster", "slower", "bigger", "larger", "smaller", "more", "less", "expected", "than"}
DAMAGE_ADJ = {"damaged", "destroyed", "ruined", "dead", "withered", "infected", "diseased", "dying", "wilted", "burnt", "burned"}
CAUSES = {"pest", "pests", "insect", "insects", "infestation", "locust", "locusts", "drought", "flood", "fire",
          "storm", "disease", "frost", "blight", "hail"}

IRREGULAR_VERBS = {"swam": "swim", "ran": "run", "flew": "fly", "drove": "drive", "rode": "ride", "wrote": "write",
                   "ate": "eat", "grew": "grow", "built": "build", "caught": "catch", "threw": "throw", "fell": "fall",
                   "rose": "rise", "shone": "shine", "sang": "sing", "dug": "dig", "sat": "sit", "stood": "stand",
                   "took": "take", "held": "hold", "led": "lead"}
VERB_BASES = {
    "inspect", "examine", "walk", "run", "jog", "swim", "dive", "fly", "soar", "glide", "float", "sail", "drive",
    "ride", "cycle", "climb", "hike", "jump", "dance", "play", "cook", "bake", "chop", "stir", "pour", "mix",
    "serve", "eat", "drink", "build", "construct", "repair", "fix", "paint", "draw", "write", "type", "read",
    "study", "teach", "learn", "work", "harvest", "plant", "grow", "water", "sow", "pick", "gather", "collect",
    "carry", "lift", "push", "pull", "throw", "catch", "kick", "hit", "swing", "shoot", "pray", "meditate",
    "stretch", "smile", "laugh", "talk", "speak", "discuss", "present", "meet", "shake", "hug", "wave", "watch",
    "observe", "search", "explore", "travel", "fish", "hunt", "feed", "graze", "roam", "wander", "flow", "crash",
    "burn", "shine", "rise", "bloom", "wither", "flood", "destroy", "damage", "clean", "wash", "sweep",
    "operate", "monitor", "measure", "test", "analyze", "scan", "sew", "weave", "craft", "design", "launch",
    "land", "arrive", "depart", "shop", "buy", "sell", "pay", "count", "deliver", "ship", "load", "unload",
    "park", "wait", "queue", "cross", "protect", "rescue", "treat", "heal", "vaccinate", "protest", "march",
    "vote", "celebrate", "perform", "sing", "swarm", "migrate", "hover", "nest", "spray", "irrigate", "plow",
    "fall", "stand", "sit", "gaze", "stare", "look", "hold", "lead", "attend", "inspire", "invest", "negotiate",
    "sign", "negotiate", "cheer", "race", "train", "exercise", "lift", "pedal", "skate", "surf", "ski", "paddle",
    "row", "fish", "forage", "feed", "hatch", "bloom", "sprout", "pollinate", "mine", "drill", "weld", "assemble",
    "manufacture", "package", "inspect", "supervise", "browse", "scroll", "stream", "code", "develop", "debug",
    "explain", "demonstrate", "photograph", "check", "conduct", "capture", "film", "record", "show", "program",
}
VISUAL_GERUND_ALIAS = {"inspect": "examining", "observe": "watching", "study": "studying", "gaze": "looking",
                       "stare": "looking", "destroy": "damaging"}

DEFAULT_SETTINGS = {
    "farmer": "agricultural field", "farmers": "agricultural field", "fisherman": "fishing boat on the sea",
    "fishermen": "fishing boat on the sea", "doctor": "hospital", "doctors": "hospital", "nurse": "hospital",
    "surgeon": "operating room", "teacher": "classroom", "students": "classroom", "student": "classroom",
    "chef": "restaurant kitchen", "cook": "kitchen", "scientist": "laboratory", "scientists": "laboratory",
    "miner": "mine", "welder": "workshop", "pilot": "cockpit", "athlete": "stadium", "athletes": "stadium",
    "shepherd": "pasture", "builder": "construction site", "builders": "construction site",
    "engineer": "workshop", "engineers": "workshop", "programmer": "office", "developers": "office",
}
# abstract concepts -> concrete visual stand-ins (used as an *alternative* query, disclosed in notes)
VISUAL_CONCEPTS = {
    "economy": "city financial district", "inflation": "shopper counting money", "innovation": "engineers building prototype",
    "teamwork": "team collaborating in office", "growth": "plant sprouting", "pollution": "factory smoke",
    "climate": "melting glacier", "sustainability": "wind turbines", "security": "security camera",
    "education": "classroom students", "health": "doctor with patient", "technology": "circuit board",
    "future": "futuristic city skyline", "success": "people celebrating", "stress": "person overwhelmed at desk",
    "privacy": "person using laptop at night", "leadership": "manager leading meeting",
}
# word-sense ambiguities: word -> {context word -> extra query term}, plus terms to avoid when sense is known
AMBIGUOUS = {
    "crane": {"construction": "construction crane", "building": "construction crane", "bird": "crane bird", "wetland": "crane bird"},
    "bank": {"river": "river bank", "money": "bank building", "loan": "bank building", "finance": "bank building"},
    "mouse": {"computer": "computer mouse", "click": "computer mouse", "cheese": "mouse animal", "field": "mouse animal"},
    "apple": {"tree": "apple fruit", "orchard": "apple fruit", "phone": "smartphone", "computer": "laptop"},
    "python": {"snake": "python snake", "jungle": "python snake", "code": "programming code", "programming": "programming code"},
    "bat": {"cave": "bat animal", "night": "bat animal", "cricket": "cricket bat", "baseball": "baseball bat"},
}
# word -> resolved sense (the AMBIGUOUS replacement string) -> metadata words typical for that sense
SENSE_CLUSTERS = {
    "crane": {"construction crane": {"construction", "building", "site", "tower", "machinery", "skyscraper", "lifting"},
              "crane bird": {"bird", "wetland", "wildlife", "feathers", "marsh", "heron", "flying"}},
    "bank": {"river bank": {"river", "riverbank", "stream", "shore", "fishing", "water", "nature", "lake", "angler", "anglers"},
             "bank building": {"money", "finance", "financial", "loan", "atm", "credit", "business", "account", "teller", "office", "building"}},
    "mouse": {"computer mouse": {"computer", "click", "keyboard", "pc", "desk", "cursor", "laptop", "office", "wireless"},
              "mouse animal": {"animal", "rodent", "cheese", "cute", "field", "wildlife", "pet", "whiskers"}},
    "apple": {"apple fruit": {"fruit", "tree", "orchard", "red", "juicy", "harvest", "food", "healthy"},
              "smartphone": {"phone", "iphone", "smartphone", "technology", "device", "screen"},
              "laptop": {"computer", "laptop", "macbook", "technology", "device"}},
    "python": {"python snake": {"snake", "reptile", "jungle", "wildlife", "slither", "constrictor", "animal"},
               "programming code": {"programming", "code", "coding", "software", "developer", "laptop", "computer", "script", "screen"}},
    "bat": {"bat animal": {"animal", "cave", "night", "wildlife", "flying", "mammal"},
            "cricket bat": {"cricket", "sport", "pitch", "stadium", "batsman"},
            "baseball bat": {"baseball", "sport", "stadium", "pitcher", "game"}},
}
DEFAULT_AVOID = ["cartoon", "animation", "3d render", "illustration", "logo", "title card"]

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+")


# ------------------------------------------------------------------ verb helpers
def _to_base(tok: str) -> Optional[str]:
    t = tok.lower()
    if t in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[t]
    if t in VERB_BASES:
        return t
    cands: List[str] = []
    if t.endswith("ies"):
        cands.append(t[:-3] + "y")
    if t.endswith("es"):
        cands.append(t[:-2])
    if t.endswith("s"):
        cands.append(t[:-1])
    if t.endswith("ied"):
        cands.append(t[:-3] + "y")
    if t.endswith("ed"):
        cands += [t[:-2], t[:-1]]
        if len(t) > 4 and t[-3] == t[-4]:
            cands.append(t[:-3])
    if t.endswith("ing"):
        stem = t[:-3]
        cands += [stem, stem + "e"]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            cands.append(stem[:-1])
    for c in cands:
        if c in VERB_BASES:
            return c
    return None


def to_gerund(base: str) -> str:
    if base in VISUAL_GERUND_ALIAS:
        return VISUAL_GERUND_ALIAS[base]
    if base.endswith("ie"):
        return base[:-2] + "ying"
    if base.endswith("e") and not base.endswith(("ee", "ye", "oe")):
        return base[:-1] + "ing"
    if len(base) >= 3 and base[-1] not in "aeiouwxy" and base[-2] in "aeiou" and base[-3] not in "aeiou" and len(re.findall(r"[aeiou]+", base)) == 1:
        return base + base[-1] + "ing"
    return base + "ing"


def _is_ing_form(tok: str) -> bool:
    t = tok.lower()
    return t.endswith("ing") and _to_base(t) is not None


def _clean_np(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t.lower() not in DETERMINERS and t.lower() not in ADVERBS_DROP
            and t.lower() not in ABSTRACT_WORDS and t.lower() not in AUXILIARIES]


def _singular(w: str) -> str:
    wl = w.lower()
    if wl.endswith("ies") and len(wl) > 4:
        return wl[:-3] + "y"
    if wl.endswith("s") and not wl.endswith("ss") and len(wl) > 3:
        return wl[:-1]
    return wl


# ------------------------------------------------------------------ interface
class SceneAnalyzer(ABC):
    @abstractmethod
    def analyze(self, narration_text: str, visual_hint: Optional[str] = None) -> SceneAnalysis:
        ...


class RuleBasedSceneAnalyzer(SceneAnalyzer):
    MAX_QUERY_WORDS = 8

    def analyze(self, narration_text: str, visual_hint: Optional[str] = None) -> SceneAnalysis:
        text = (narration_text or "").strip()
        notes: List[str] = []
        raw_tokens = _TOKEN_RE.findall(text)
        low = [t.lower() for t in raw_tokens]

        # --- split main clause and context (subordinate) clause --------------------
        split_at = next((i for i, t in enumerate(low) if t in SUBORDINATORS and i > 0), None)
        main, ctx = (raw_tokens[:split_at], raw_tokens[split_at + 1:]) if split_at is not None else (raw_tokens, [])
        # also cut main clause at a coordinating ", and" / "but" — keep only first clause for the *visual*
        secondary_action: Optional[str] = None
        secondary_object = ""
        for i, t in enumerate(main):
            if t.lower() in {"but", "and"} and i > 2 and any(_to_base(x) for x in main[:i]):
                tail = main[i + 1:]
                # "A doctor examines a patient AND checks the patient's blood pressure": the 2nd verb is also visible action
                if tail and t.lower() == "and" and _to_base(tail[0]) is not None and tail[0].lower() not in AUXILIARIES:
                    secondary_action = to_gerund(_to_base(tail[0]))
                    sec_np = [re.sub(r"['\u2019]s$", "", x) for x in _clean_np(tail[1:])]
                    cut = next((k for k, x in enumerate(sec_np) if x.lower() in SETTING_PREPS | SUBORDINATORS), len(sec_np))
                    secondary_object = " ".join(sec_np[:cut]).lower()
                ctx = main[i + 1:] + ctx
                main = main[:i]
                break
        mlow = [t.lower() for t in main]

        # --- verb detection ---------------------------------------------------------
        verb_idx, verb_base = None, None
        for i, t in enumerate(mlow):
            if t in AUXILIARIES:
                if i + 1 < len(mlow) and _is_ing_form(mlow[i + 1]):
                    verb_idx, verb_base = i + 1, _to_base(mlow[i + 1])
                    break
                continue
            b = _to_base(t)
            if b is not None:
                # a bare plural noun right after a determiner/adjective ("the crops") is a noun, not a verb
                if i > 0 and mlow[i - 1] in DETERMINERS:
                    continue
                if i > 0 and mlow[i - 1] in DAMAGE_ADJ:
                    continue
                verb_idx, verb_base = i, b
                break
        subject_tokens: List[str] = []
        object_tokens: List[str] = []
        pre_setting: List[str] = []
        pre_prep = "in"
        if verb_idx is not None:
            subj_raw = main[:verb_idx]
            for i, t in enumerate(subj_raw):  # "Workers on a crane lift ..." -> subject workers, place crane
                if t.lower() in SETTING_PREPS and i > 0:
                    pre_prep = t.lower() if t.lower() in MOTION_PREPS else "in"
                    pre_setting = _clean_np([x for x in subj_raw[i + 1:] if x.lower() not in AUXILIARIES])
                    subj_raw = subj_raw[:i]
                    break
            subject_tokens = _clean_np(subj_raw)
            rest = main[verb_idx + 1:]
        else:
            aux_i = next((i for i, t in enumerate(mlow) if t in AUXILIARIES), None)
            if aux_i is not None and aux_i > 0:   # copula scene: "<subject> are <predicate>"
                subject_tokens = _clean_np(main[:aux_i])
                rest = main[aux_i + 1:]
            else:
                subject_tokens = []
                rest = main
            notes.append("No action verb recognised; scene treated as a noun phrase.")
        # gerund-led abstract scene ("Protecting our oceans is ...") has no subject
        if verb_idx is not None and verb_idx == 0 and _is_ing_form(mlow[0]):
            subject_tokens = []

        # --- split object / setting (prepositional phrase) -------------------------
        setting_tokens: List[str] = []
        setting_prep = "in"
        rlow = [t.lower() for t in rest]
        for i, t in enumerate(rlow):
            if t in SETTING_PREPS:
                object_tokens = rest[:i]
                tail = rest[i + 1:]
                setting_prep = t if t in MOTION_PREPS else "in"
                for j, u in enumerate(tail):
                    if u.lower() in SETTING_PREPS | AUXILIARIES | PRONOUNS:
                        tail = tail[:j]
                        break
                setting_tokens = _clean_np(tail)
                break
        else:
            object_tokens = rest
        # trailing copula clause: "oceans is important" -> stop at the auxiliary
        for j, u in enumerate(object_tokens):
            if u.lower() in AUXILIARIES:
                object_tokens = object_tokens[:j]
                break
        recipients: List[str] = []
        for j, u in enumerate(object_tokens):   # "explains an experiment to students": recipients are a separate object
            if u.lower() == "to" and j > 0:
                recipients = _clean_np(object_tokens[j + 1:])
                object_tokens = object_tokens[:j]
                break
        for j, u in enumerate(object_tokens):   # "faster than expected" is not a visual object
            if u.lower() in {"than", "because"}:
                object_tokens = object_tokens[:j]
                break
        object_tokens = [t for t in _clean_np(object_tokens) if t.lower() not in COMMON_STOP_WORDS
                         and t.lower() not in {"home", "part", "kind", "type", "lot", "way"}] if verb_idx is None else _clean_np(object_tokens)
        if verb_idx is None and not subject_tokens:  # bare noun-phrase scene: first NP is subject
            subject_tokens, object_tokens = object_tokens[:2], object_tokens[2:]
        if subject_tokens and subject_tokens[0].lower() in PRONOUNS:
            subject_tokens = []
        subject_tokens = [t for t in subject_tokens if t.lower() not in PRONOUNS][-3:]

        subject = " ".join(subject_tokens).lower() or None
        action = to_gerund(verb_base) if verb_base else None
        objects_np = " ".join(object_tokens).lower()
        if not setting_tokens and pre_setting:
            setting_tokens, setting_prep = pre_setting, pre_prep
        setting = " ".join(setting_tokens).lower() or None
        context_np = " ".join(_clean_np(ctx)).lower() or None

        # --- context-driven refinement: "damaged crops after a pest infestation" -----
        ctx_low = {w.lower() for w in ctx}
        cause = next((w for w in ("pest", "disease", "drought", "flood", "fire", "storm", "frost", "locust", "insect") if w in ctx_low or w + "s" in ctx_low), None)
        if cause is None and ("infestation" in ctx_low):
            cause = "pest"
        obj_words = objects_np.split()
        if cause and any(w in DAMAGE_ADJ for w in obj_words):
            head = obj_words[-1]
            if head in {"crops", "crop"} and cause in {"pest", "disease", "insect", "locust"}:
                head = "crop leaves"
            elif head.endswith("s") is False:
                pass
            objects_np = f"{cause}-damaged {head}"
            notes.append(f"Object refined using context clause ('{cause}').")

        # --- default / inferred setting ---------------------------------------------
        if setting is None and subject:
            for w in reversed(subject.split()):
                if w in DEFAULT_SETTINGS:
                    setting = DEFAULT_SETTINGS[w]
                    notes.append(f"Setting '{setting}' is inferred from the subject, not stated in the narration.")
                    break
        if setting is None and objects_np:
            for w in objects_np.split():
                if w in {"crops", "crop", "harvest", "wheat", "rice", "corn", "tractor"}:
                    setting = "agricultural field"
                    notes.append(f"Setting '{setting}' is inferred from the objects, not stated in the narration.")
                    break

        # --- explicit user hint wins over everything ---------------------------------
        hint = (visual_hint or "").strip()

        # --- ambiguity handling ------------------------------------------------------
        avoid = list(DEFAULT_AVOID)
        sense_good: set = set()
        sense_bad: set = set()
        all_words = set(low)
        for amb, senses in AMBIGUOUS.items():
            if amb in all_words:
                for ctxw, replacement in senses.items():
                    if ctxw in all_words:
                        if subject and amb in subject.split():
                            subject = subject.replace(amb, replacement)
                        if amb in objects_np.split():
                            objects_np = objects_np.replace(amb, replacement)
                        notes.append(f"Word '{amb}' disambiguated as '{replacement}' using context '{ctxw}'.")
                        clusters = SENSE_CLUSTERS.get(amb, {})
                        good_set = set(clusters.get(replacement, set())) | {ctxw}
                        bad_set = set().union(*[v for k, v in clusters.items() if k != replacement]) if clusters else set()
                        sense_good |= good_set
                        sense_bad |= (bad_set - good_set - {amb})
                        break
                else:
                    notes.append(f"Word '{amb}' is ambiguous and no disambiguating context was found.")

        def _dedupe_words(s: str) -> str:
            out: List[str] = []
            for w in s.split():
                if not out or out[-1] != w:
                    out.append(w)
            return " ".join(out)
        objects_np = _dedupe_words(objects_np)
        subject = _dedupe_words(subject) if subject else subject
        objects = [o for o in re.split(r"\s+and\s+|,", objects_np) if o.strip()]
        if recipients:
            objects.append(" ".join(recipients).lower())
        if secondary_object and secondary_object not in objects:
            objects.append(secondary_object)
        if secondary_action:
            notes.append(f"Second coordinated action recognised: {secondary_action} {secondary_object}".strip() + ".")

        # --- queries -------------------------------------------------------------------
        def limit(parts: List[str]) -> str:
            words = " ".join(p for p in parts if p).split()[: self.MAX_QUERY_WORDS]
            while words and words[-1].lower() in SETTING_PREPS:
                words.pop()
            return " ".join(words)

        if setting and setting.split()[-1] in {"laptop", "computer", "screen", "keyboard", "phone", "tablet"}:
            setting_prep = "on"
        elif setting and setting.split()[-1] in {"desk", "table"}:
            setting_prep = "at"
        set_phrase = f"{setting_prep} {setting}" if setting else ""
        queries: List[str] = []
        if hint:
            queries.append(limit([hint]))
        # full-fidelity query keeps subject + action + object + setting relationship
        q_full_parts = [subject or "", action or "", objects_np, set_phrase]
        if len(" ".join(q_full_parts).split()) > self.MAX_QUERY_WORDS:  # drop lowest-priority bits, keep S+A+setting
            q_full_parts = [subject or "", action or "", " ".join(objects_np.split()[-2:]), set_phrase]
        queries.append(limit(q_full_parts))
        if secondary_action:   # the more distinctive action first: "doctor checking patient blood pressure"
            queries.append(limit([subject or "", secondary_action, secondary_object]))
        if subject or action:
            queries.append(limit([subject or "", action or "", set_phrase]))
        if objects_np:
            queries.append(limit([objects_np, set_phrase]))
        if subject and objects_np and not action:
            queries.insert(len(queries) - 1 if len(queries) > 1 else 0, limit([subject, objects_np, set_phrase]))
        if subject and setting:
            queries.append(limit([subject, set_phrase]))
        if subject:
            queries.append(limit([subject, action or ""]))
        # concept fall-back for abstract narration
        concept_hit = next((VISUAL_CONCEPTS[w] for w in low if w in VISUAL_CONCEPTS), None)
        subject_is_abstract = bool(subject) and all(w in VISUAL_CONCEPTS or w in ABSTRACT_WORDS for w in subject.split())
        if concept_hit and (not (subject and action) or subject_is_abstract):
            queries.append(concept_hit)
            notes.append(f"Abstract concept mapped to concrete visual stand-in: '{concept_hit}'.")
        nouns = [w for w in re.findall(r"[a-z]+", " ".join([subject or "", objects_np, setting or ""]))
                 if w not in COMMON_STOP_WORDS and len(w) > 2]
        if nouns:
            queries.append(" ".join(dict.fromkeys(nouns[:3])))
        if not any(q.strip() for q in queries):
            fallback = [w for w in low if w not in COMMON_STOP_WORDS and len(w) > 2][:3]
            queries.append(" ".join(fallback) or "nature landscape")
            notes.append("Narration contained no concrete visual terms; generic query used.")
        # --- activity frames: alternative retrieval phrasings + action/object evidence -------------------------
        frames = match_frames(f"{text} {hint}")
        frame_queries = [q for fr in frames for q in fr.queries]
        if frame_queries:
            # keep the narration-faithful queries first; interleave the frame variants right after them so that a
            # failed first query is retried with a *different* phrasing instead of an identical one
            head = queries[: 2 if hint else 1]
            rest_q = queries[len(head):]
            queries = head + frame_queries[:2] + rest_q[:2] + frame_queries[2:] + rest_q[2:]
            notes.append("Alternative search phrasings added from activity frame(s): " + ", ".join(fr.id for fr in frames) + ".")
        ent_queries, ent_words = entity_standins(text)
        sensitive, sens_queries, sens_words = sensitive_profile(f"{text} {hint}")
        queries = ent_queries + sens_queries + queries
        if ent_queries:
            notes.append("Named brand/product shown through a generic stand-in (no logos or real people): " + ", ".join(ent_queries) + ".")
        if sensitive:
            notes.append("Sensitive story detected: sombre queries used and upbeat clips are rejected.")
        uniq: List[str] = []
        for q in queries:
            q = " ".join(re.sub(r"\s+", " ", q).strip().split()[: self.MAX_QUERY_WORDS])
            if q and q.lower() not in {u.lower() for u in uniq}:
                uniq.append(q)

        # --- weights for ranking ----------------------------------------------------------
        weights: Dict[str, float] = {}
        def add(words: str, w: float):
            for tok in re.findall(r"[a-z]+", words.lower()):
                if tok in COMMON_STOP_WORDS or tok in DETERMINERS or len(tok) < 3:
                    continue
                weights[tok] = max(weights.get(tok, 0.0), w)
        add(subject or "", 3.0)
        if verb_base:
            add(verb_base + " " + (action or ""), 2.0)
        add(objects_np, 2.0)
        add(setting or "", 1.5)
        add(context_np or "", 1.0)
        if hint:
            add(hint, 3.0)
        if not weights:
            add(" ".join(uniq), 1.0)
        # singular/plural variants so "crops" matches tag "crop"
        for tok in list(weights):
            weights.setdefault(_singular(tok), weights[tok])

        desc_parts = [subject or "", action or "", objects_np, set_phrase]
        description = " ".join(p for p in desc_parts if p).strip()
        if secondary_action:
            description += f", {secondary_action} {secondary_object}".rstrip()
        description = (description[:1].upper() + description[1:] + ".") if description else uniq[0].capitalize() + "."
        if hint:
            description = f"{description} (User hint: {hint})"
        narr_low = " ".join(low)
        primary = frames[0] if frames else None
        # With a known activity frame only ITS terms count as proof of the action; the narration's own verb is usually
        # generic ("examines", "explains") and secondary frames only contribute extra search phrasings.
        verb_terms = [a for a in ([action] if action else []) + ([verb_base] if verb_base else [])
                      + ([secondary_action] if secondary_action else []) if a]
        action_terms = list(dict.fromkeys((list(primary.action_terms) if primary else []) + ([] if primary else verb_terms)))
        object_terms = list(primary.object_terms) if primary else []
        target_terms = [o for o in objects if o]
        foil_terms = [x for x in (primary.foil_terms if primary else ())
                      if x.lower() not in narr_low and x.lower() not in {a.lower() for a in action_terms}]
        negative_terms = [x for x in (primary.negative_terms if primary else ()) if x.lower() not in narr_low]
        setting_terms = list(dict.fromkeys(([setting] if setting else []) + [x for fr in frames[:1] for x in fr.setting_terms]))
        if verb_base or frames:
            scene_type = "activity"
        elif concept_hit:
            scene_type = "abstract"
        else:
            scene_type = "place" if setting and not objects_np else "object"
        return SceneAnalysis(
            subject=subject, action=action, setting=setting, objects=objects,
            visual_description=description, queries=uniq, avoid_terms=list(dict.fromkeys(avoid + negative_terms)),
            search_terms=sorted(weights, key=lambda k: -weights[k]), term_weights=weights, notes=notes,
            context=context_np, scene_type=scene_type, frame_ids=[fr.id for fr in frames], gated_frame=bool(primary and primary.gate),
            subject_terms=list(dict.fromkeys(subject_synonyms(subject or "") + ent_words + sens_words)), action_terms=action_terms, object_terms=object_terms,
            target_terms=target_terms, setting_terms=setting_terms, foil_terms=list(dict.fromkeys(foil_terms)),
            negative_terms=negative_terms, sense_good_terms=sorted(sense_good), sense_bad_terms=sorted(sense_bad),
            content_words=_content_words(text, COMMON_STOP_WORDS),
            grounding_terms=list(dict.fromkeys(_content_words(text, COMMON_STOP_WORDS) + ent_words + sens_words
                                               + (re.findall(r"[a-z]+", concept_hit) if concept_hit else []))),
            sensitive=sensitive, strict_grounding=bool(ent_words),
        )


class LLMSceneAnalyzer(SceneAnalyzer):
    """Wraps any ``complete(prompt) -> str`` callable (e.g. an LLM client). Falls back to the rule-based
    analyser on any failure and records that in ``notes``."""

    PROMPT = (
        "You prepare stock-footage searches for one narration scene. Return ONLY JSON with keys: "
        "subject, action, setting, objects (list), visual_description, queries (3-5 progressively broader "
        "stock-footage search queries of at most 8 words that preserve subject-action-setting), avoid_terms (list).\n"
        "Narration: {text}\nUser visual hint: {hint}\n"
    )

    def __init__(self, complete: Callable[[str], str], fallback: Optional[SceneAnalyzer] = None, name: str = "llm"):
        self.complete, self.fallback, self.name = complete, fallback or RuleBasedSceneAnalyzer(), name

    def analyze(self, narration_text: str, visual_hint: Optional[str] = None) -> SceneAnalysis:
        base = self.fallback.analyze(narration_text, visual_hint)
        try:
            raw = self.complete(self.PROMPT.format(text=narration_text, hint=visual_hint or "none"))
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
            if not queries:
                raise ValueError("no queries returned")
            return SceneAnalysis(
                subject=data.get("subject") or base.subject, action=data.get("action") or base.action,
                setting=data.get("setting") or base.setting,
                objects=[str(o) for o in data.get("objects", base.objects)],
                visual_description=str(data.get("visual_description") or base.visual_description),
                queries=queries + [q for q in base.queries if q not in queries],
                avoid_terms=[str(a) for a in data.get("avoid_terms", base.avoid_terms)] or base.avoid_terms,
                search_terms=base.search_terms, term_weights=base.term_weights, method=f"llm:{self.name}",
                notes=base.notes, context=base.context, scene_type=base.scene_type, frame_ids=base.frame_ids, gated_frame=base.gated_frame,
                subject_terms=base.subject_terms, action_terms=base.action_terms, object_terms=base.object_terms,
                target_terms=base.target_terms, setting_terms=base.setting_terms, foil_terms=base.foil_terms, negative_terms=base.negative_terms,
                content_words=base.content_words, grounding_terms=base.grounding_terms, sensitive=base.sensitive,
                strict_grounding=base.strict_grounding,
            )
        except Exception as exc:  # noqa: BLE001 - any LLM/JSON failure must degrade gracefully
            base.notes.append(f"LLM analysis failed ({type(exc).__name__}); rule-based analysis used instead.")
            return base
