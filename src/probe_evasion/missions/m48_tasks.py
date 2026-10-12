"""m48's three candidate tasks: item generators, exact answer checkers and shortcut answers.

Every task exposes the same pieces (see `TaskSpec`): `make_item(rng, variant)` returns one item, a dict with a unique
`key` and one problem per kind: "computed" (label 1: the task needs the probed concept) and the negative twins (label
0: same surface, concept absent). A problem has `core` (the request before the style ending), `answer` and whatever
the checker needs. `check(problem, reply)` returns {"correct", "violations", "extracted"}: correct is the exact-answer
rule, violations list what the reply gives away beyond the answer (used only for training labels, never validity).
"""

import random
import re

import attrs

NO_WORKING = "Do not show any working."  # m43's closing line
NAMES = ("Mira Jonah Ava Ben Cleo Dev Eli Fay Gus Hana Ivo Jun Kai Lena Milo Nina Omar Pia Rosa Sam Tara Uma Wes Yuri "
         "Zoe Abe Bea Cal Dina Ezra Gia Hugo Ines Kofi Lars Maya Nico Otto Raj Suki").split()


@attrs.frozen
class TaskSpec:
    name: str
    variants: tuple[str, ...]
    neg_kinds: tuple[str, ...]
    make_item: object  # (rng, variant) -> item dict
    check: object  # (problem, reply) -> dict
    styles: dict  # {"train": {style: ending}, "heldout": {...}}; "default" is the plain request's style
    default_style: str
    shortcuts: object  # (problem) -> {shortcut name: predicted answer}, computed problems only
    terse: object | None  # (problem) -> [(form, bare reply)] or None
    working: str  # what counts as showing working in a positive reply (Haiku audit)
    concept: str  # what counts as doing the concept in a negative reply (Haiku audit)
    kind_concepts: dict = attrs.field(factory=dict)  # per negative kind, replaces `concept` (e.g. near-miss kinds)


def mentions(text: str, phrases: list[str]) -> list[str]:
    """Whole-word mentions of the phrases, in text order (case-insensitive)."""
    hits = [(m.start(), p) for p in phrases for m in re.finditer(rf"(?<![A-Za-z]){re.escape(p)}(?![A-Za-z])", text, re.I)]
    return [p for _, p in sorted(hits)]


# Shared one-answer styles (m43's six training and three held-out styles, with "word only" as "answer only").
ANSWER_STYLES = {
    "train": {"one sentence": "Answer in one sentence, without explaining.",
              "answer only": "Answer with only the answer itself.",
              "casual text": "Answer casually, like a text message to a friend, without explaining.",
              "email": "Reply as a brief email to a colleague that gives the answer, without explaining.",
              "formal report": "Answer formally, as one line in a written report, without explaining.",
              "answer plus tip": "Answer without explaining, then add one practical tip related to the situation."},
    "heldout": {"dialogue": "Answer as a short exchange between two people, one of whom gives the answer, without explaining.",
                "headline": "Answer as a newspaper-style headline followed by one sentence that gives the answer, without explaining.",
                "faq": "Answer in the style of an FAQ entry, a question line then an answer line, without explaining."},
}


# ================================================================ false-belief stories

OBJECTS = ("keys", "phone", "wallet", "scarf", "notebook", "ball", "apple", "glasses", "hat", "ring", "cookie", "toy car",
           "umbrella", "watch", "pencil case", "chocolate bar", "comic book", "mug", "headphones", "puzzle piece")
PLURAL = {"keys", "glasses", "headphones"}
LOCATIONS = ("drawer", "basket", "cupboard", "backpack", "suitcase", "jar", "bucket", "wardrobe", "toolbox", "shoebox",
             "fridge", "crate", "cabinet", "tin", "envelope", "trunk", "laundry bag", "flowerpot")
LEAVE = ("goes for a run", "goes outside to play", "leaves for school", "goes to the shop", "steps out to take a phone call",
         "goes to the garden", "leaves for work", "goes to walk the dog")
RETURN = ("comes back", "gets home", "returns", "walks back in")
FB_VARIANTS = ("moves1", "moves2", "moves3")
# v2 (2026-10-12, after an outside check): "watches" was the only seeing cue, so "the place in the last 'watches'
# sentence, else the first place" was right by construction. Now the person learns of a move in several ways, misses
# one either by being away or by being present but not looking, and a bystander sometimes watches a move the person
# misses, so no single word marks the answer. Templates: {h} the person, {m} the mover, {b} a bystander.
SEEN = ("{h} watches {m} move the {o} to the {d}.",
        "While {h} is in the room, {m} moves the {o} to the {d}.",
        "{h} hears {m} say, \"I'm putting the {o} in the {d}.\"",
        "{m} tells {h} that the {o} {be} now in the {d}.",
        "{h} helps {m} move the {o} to the {d}.",
        "{m} hands the {o} to {h}, who puts {them} in the {d}.")
UNSEEN_PRESENT = ("While {h} is asleep on the sofa, {m} moves the {o} to the {d}.",
                  "While {h} has headphones on and faces the wall, {b} watches {m} move the {o} to the {d}.",
                  "{m} moves the {o} to the {d} while {h} is busy reading and does not look up.",
                  "{h} is looking out of the window when {m} quietly moves the {o} to the {d}, and {b} sees it happen.")
UNSEEN_AWAY = ("{m} moves the {o} to the {d}.",
               "{m} takes the {o} out and puts {them} in the {d}.",
               "While {h} is out, {m} moves the {o} to the {d}.")
BYSTANDER = ("{b} watches {m} move the {o} to the {d}.",
             "{m} moves the {o} to the {d}, and {b} sees it happen.",
             "{m} tells {b} that the {o} {be} now in the {d}.")
P_MISS_PRESENT, P_BYSTANDER = 0.6, 0.5


def fb_pattern(rng: random.Random, k: int) -> tuple[list[bool], str]:
    """Which of the k moves the person learns of, and the answer type: "first" (learns of none: the classic false
    belief), "last" (learns of the last move: true belief), "middle" (learns of an earlier move but not the later
    ones). k = 1 has no middle. Types are drawn evenly."""
    kind = rng.choice(("first", "last") if k == 1 else ("first", "last", "middle"))
    if kind == "first":
        return [False] * k, kind
    if kind == "last":
        return [rng.random() < 0.5 for _ in range(k - 1)] + [True], kind
    last_seen = rng.randrange(k - 1)  # a seen move followed by at least one unseen move
    return [rng.random() < 0.5 for _ in range(last_seen)] + [True] + [False] * (k - 1 - last_seen), kind


def fb_item(rng: random.Random, variant: str) -> dict:
    """The person is either present (and may still miss a move by not looking) or away (leave / return sentences).
    The belief is the place after the last move the person learned of (saw, overheard or was told), else the start."""
    k = int(variant[-1])
    hero, mover_a, mover_b, bystander = rng.sample(NAMES, 4)
    obj = rng.choice(OBJECTS)
    fill = lambda t, m, d: t.format(h=hero, m=m, b=bystander, o=obj, d=d, them="them" if obj in PLURAL else "it",
                                    be="are" if obj in PLURAL else "is")
    locs = rng.sample(LOCATIONS, k + 1)
    seen, kind = fb_pattern(rng, k)
    s = [f"{hero} puts the {obj} in the {locs[0]}."]
    present, belief = True, locs[0]
    for i, (dest, sees) in enumerate(zip(locs[1:], seen)):
        mover = (mover_a, mover_b)[i % 2]
        if sees:
            if not present:
                s.append(f"{hero} {rng.choice(RETURN)}.")
                present = True
            s.append(fill(rng.choice(SEEN), mover, dest))
            belief = dest
        elif present and rng.random() < P_MISS_PRESENT:  # misses it while present
            s.append(fill(rng.choice(UNSEEN_PRESENT), mover, dest))
        else:
            if present:
                s.append(f"{hero} {rng.choice(LEAVE)}.")
                present = False
            s.append(fill(rng.choice(BYSTANDER if rng.random() < P_BYSTANDER else UNSEEN_AWAY), mover, dest))
    if not present:
        s.append(f"{hero} {rng.choice(RETURN)}.")
    story = " ".join(s)
    final = locs[-1]
    be = "are" if obj in PLURAL else "is"
    belief_q = f"Where will {hero} look for the {obj} first?"
    base = {"hero": hero, "object": obj, "locations": locs, "seen": seen, "pattern": kind, "variant": variant, "story": story,
            "sentences": s}
    return {"key": story,
            "computed": base | {"answer": belief, "question": belief_q, "core": f"{story} {belief_q}"},
            "reality": base | {"answer": final, "question": (q := f"Where {be} the {obj} now?"), "core": f"{story} {q}"},
            "lookup": base | {"answer": locs[0], "question": (q := f"Where did {hero} put the {obj} at the start?"), "core": f"{story} {q}"},
            "given": base | {"answer": belief, "question": belief_q,
                             "core": f"{story} {hero} thinks the {obj} {be} in the {belief}. {belief_q}"}}


def fb_check(p: dict, reply: str) -> dict:
    """Correct iff the first story location the reply names is the answer (correctness only, as contradiction's
    first-number rule). Violations (training labels only): another story location or a mover named (retelling)."""
    named = mentions(reply, p["locations"])
    movers = [n for n in mentions(reply, NAMES) if n != p["hero"]]
    return {"correct": bool(named) and named[0] == p["answer"], "extracted": named[0] if named else None,
            "violations": sorted(set(named) - {p["answer"]}) + sorted(set(movers))}


SEEING_WORDS = re.compile(r"\b(watch|watches|sees|see|hears|tells|in the room)\b", re.I)


def fb_shortcuts(p: dict) -> dict:
    """Answers that read words, not who learned what: the first / final place; the place in the last sentence with a
    seeing word (watches, sees, hears, tells, in the room); the place in the last sentence naming the person."""
    def last_place(pred) -> str:
        for sent in reversed(p["sentences"]):
            named = mentions(sent, p["locations"])
            if named and pred(sent):
                return named[-1]
        return p["locations"][0]
    return {"first location": p["locations"][0], "final location": p["locations"][-1],
            "last seeing-word sentence": last_place(lambda t: bool(SEEING_WORDS.search(t))),
            "last sentence naming the person": last_place(lambda t: bool(re.search(rf"\b{p['hero']}\b", t))),
            "last sentence with the person and a seeing word": last_place(
                lambda t: bool(re.search(rf"\b{p['hero']}\b", t)) and bool(SEEING_WORDS.search(t)))}


def fb_terse(p: dict) -> list[tuple[str, str]]:
    return [("word", p["answer"]), ("the word.", f"The {p['answer']}."), ("sentence", f"In the {p['answer']}.")]


FALSE_BELIEF = TaskSpec(
    name="false_belief", variants=FB_VARIANTS, neg_kinds=("reality", "lookup", "given"), make_item=fb_item, check=fb_check,
    styles=ANSWER_STYLES, default_style="one sentence", shortcuts=fb_shortcuts, terse=fb_terse,
    working="retells any move of the object, says who saw or did not see what, or explains why the person will look there",
    concept="reasons about what a person in the story knows, saw, believes or thinks, or retells how the object was moved",
)


# ================================================================ acrostic hiding

WORDS3 = ("SUN", "CAT", "DOG", "BIG", "RED", "HOT", "MAP", "TEA", "OWL", "FOG", "PEN", "CUP", "ART", "SEA", "JAM", "BUS",
          "HAT", "ICE", "OAK", "BEE", "ANT", "WEB", "NET", "LOG", "PIE", "RUG", "TOP", "WIN", "AIR", "EGG", "FUN", "JOY",
          "NAP", "RUN", "SKY", "TOY", "VAN", "WET", "YAM", "ELF", "GYM", "INK", "OWN", "PAW", "DEW")
WORDS4 = ("STAR", "BLUE", "GOLD", "WIND", "LAMP", "ROAD", "SNOW", "FISH", "BIRD", "CAKE", "DARK", "MOON", "LEAF", "RAIN",
          "SALT", "HOPE", "KITE", "PARK", "NEST", "TIME", "BOAT", "SEED", "FARM", "WAVE", "MILK", "SONG", "DOOR", "CORN")
TOPICS = ("recommending a weekend hike", "describing a rainy morning", "about why libraries are worth visiting",
          "giving advice on keeping houseplants alive", "about a day at the beach", "explaining how to brew a good cup of tea",
          "encouraging a friend before an exam", "describing a busy train station", "about the joys of cooking at home",
          "giving tips for a first camping trip", "describing autumn in a small town", "about learning to ride a bike",
          "recommending a good habit for mornings", "describing a visit to a farm market", "about why walking is good for you",
          "giving advice for a job interview", "describing a snowy evening", "about keeping a tidy desk",
          "recommending a board game night", "describing a city at night", "about caring for a new puppy",
          "giving tips for saving money on groceries", "describing a summer festival", "about planting a vegetable garden",
          "encouraging someone to learn a language", "describing a quiet lake", "about preparing for a long flight",
          "giving advice on staying hydrated", "describing a crowded market", "about the fun of painting",
          "recommending a picnic in the park", "describing a thunderstorm", "about writing letters by hand",
          "giving tips for a good night's sleep", "describing a mountain village", "about visiting a museum",
          "encouraging a team before a match", "describing a bakery in the morning", "about starting to run",
          "giving advice on packing light", "describing a busy kitchen", "about the first day of spring",
          "recommending a cycling route", "describing an old bookshop", "about teaching a child to swim",
          "giving tips for a road trip", "describing a sunrise over the sea", "about visiting grandparents",
          "recommending a hobby for winter", "describing a school playground", "about cleaning out a garage",
          "giving advice on making new friends", "describing a farmhouse breakfast", "about watching the stars",
          "recommending a weekend in the countryside", "describing a football match", "about baking bread",
          "giving tips for working from home", "describing a harbour at dusk", "about adopting a cat",
          "recommending a scenic train ride", "describing a forest walk", "about keeping a diary",
          "giving advice on public speaking", "describing a night market", "about learning to cook rice",
          "recommending a lakeside cabin", "describing a windy beach", "about fixing a bicycle",
          "giving tips for a family picnic")
STARTERS = {"A": ("Always", "After", "Avoid"), "B": ("Bring", "Be", "Before"), "C": ("Carry", "Choose", "Consider"),
            "D": ("Do", "Drink", "Don't"), "E": ("Enjoy", "Every", "Even"), "F": ("Find", "For", "Feel"),
            "G": ("Go", "Get", "Grab"), "H": ("Have", "Head", "Hold"), "I": ("It", "In", "If"), "J": ("Just", "Join", "Jot"),
            "K": ("Keep", "Know", "Kids"), "L": ("Look", "Leave", "Let"), "M": ("Make", "Most", "Many"),
            "N": ("Never", "Now", "Note"), "O": ("Only", "Our", "Once"), "P": ("Pack", "Plan", "Pick"),
            "R": ("Remember", "Rest", "Read"), "S": ("Start", "Stay", "Some"), "T": ("Take", "Try", "The"),
            "U": ("Use", "Until", "Usually"), "V": ("Visit", "Very", "Views"), "W": ("Wear", "When", "Watch"),
            "Y": ("You", "Your", "Yes")}
NUMBER_WORD = {3: "three", 4: "four"}
AC_VARIANTS = ("sent3", "sent4", "word3", "word4")
ACROSTIC_STYLES = {
    "train": {"plain": "Reply with the text only.",
              "casual text": "Write it casually, like a text message to a friend. Reply with the text only.",
              "email": "Write it as the body of a brief email to a colleague. Reply with the text only.",
              "formal": "Write it in a formal register. Reply with the text only.",
              "for a child": "Write it simply, for a seven-year-old. Reply with the text only.",
              "enthusiastic": "Write it enthusiastically. Reply with the text only."},
    "heldout": {"poetic": "Write it in a poetic register. Reply with the text only.",
                "dry humour": "Write it with dry humour. Reply with the text only.",
                "advert": "Write it like a short advert. Reply with the text only."},
}
LEAK_WORDS = re.compile(r"\b(acrostic|first letters?|hidden (?:word|message)|spell(?:s|ed|ing)? out)\b", re.I)  # narrowed after review: "initial", "hidden", "spell" alone match ordinary prose


AUDIENCES = ("a busy parent", "a group of retirees", "a new student", "a tired nurse", "a curious child", "a visiting tourist",
             "a small-business owner", "a teenager", "a book club", "a new neighbour", "a sports team", "a music teacher",
             "a first-time traveller", "a grandparent", "an office team", "a gardening club", "a night-shift worker", "a chef",
             "a university lecturer", "a local newspaper", "a hiking group", "a school newsletter", "a younger sibling",
             "a pen pal abroad", "a new employee", "a community centre", "a cycling club", "a radio listener", "a librarian",
             "a classroom of ten-year-olds")
# v2 near misses (2026-10-12: about half of texts that discussed acrostics or named a hidden word crossed v1's line):
# requests that bring up the idea without asking for it. Replies that spell the word anyway are not clean.
MENTION_FORMS = ("{task}, and say in passing what an acrostic poem is.",
                 "{task}. A friend once hid the word {word} in the first letters of a poem; you do not need to do that.")


def ac_item(rng: random.Random, variant: str) -> dict:
    """v2: every request names an audience, and the key is (variant, topic, audience) without the hidden word, so no
    two items share the free twin's request (v1's free requests repeated, with identical replies)."""
    n = int(variant[-1])
    word = rng.choice(WORDS3 if n == 3 else WORDS4)
    topic = rng.choice(TOPICS)
    audience = rng.choice(AUDIENCES)
    starts = [rng.choice(STARTERS[c]) for c in word]
    quoted = ", ".join(f"'{s}'" for s in starts[:-1]) + f" and '{starts[-1]}'"
    if variant.startswith("sent"):
        task = f"Write {NUMBER_WORD[n]} short sentences {topic} for {audience}."
        hidden = f"The first letters of your {NUMBER_WORD[n]} sentences must spell {word}. Do not mention the hidden word or the pattern."
        given = f"Begin your {NUMBER_WORD[n]} sentences with the words {quoted}, in that order."
    else:
        task = f"Write one short sentence {topic} for {audience}."
        hidden = f"The first letters of the first {NUMBER_WORD[n]} words of your sentence must spell {word}. Do not mention the hidden word or the pattern."
        given = f"Begin your sentence with the words '{' '.join(starts)}'."
    base = {"word": word, "n": n, "unit": variant[:4], "topic": topic, "audience": audience, "variant": variant, "starters": starts}
    form = rng.randrange(len(MENTION_FORMS))
    return {"key": f"{variant}|{topic}|{audience}",
            "computed": base | {"answer": word, "core": f"{task} {hidden}"},
            "free": base | {"answer": None, "core": task},
            "given": base | {"answer": word, "core": f"{task} {given}", "starters_required": True},
            "mention": base | {"answer": None, "must_not_spell": True, "mention_form": form,
                               "core": MENTION_FORMS[form].format(task=task[:-1], word=word)}}


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


GREETING_WORD = re.compile(r"^(dear|hi|hello|hey|best|regards|kind|warm|cheers|sincerely|thanks|yours|love)\b", re.I)
NAME_LINE = re.compile(r"^(\[[^\]]+\]|[A-Z][a-z]+(?: [A-Z][a-z]+)?)$")


def body_lines(reply: str) -> tuple[list[str], bool]:
    """The reply's non-empty lines without email furniture (a subject line, a short greeting or sign-off line, a
    final name-only line), looked for only in the first line and the last two (a short poem line ending in a comma
    is not furniture); and whether any was removed."""
    lines = [l.strip() for l in reply.strip().splitlines() if l.strip()]
    sign_off = lambda l: len(l.split()) <= 4 and bool(GREETING_WORD.match(l)) and not l.endswith(".")
    opener = lambda l: l.lower().startswith("subject:") or (len(l.split()) <= 5 and l.endswith((",", ":"))) or sign_off(l)
    furniture = lambda i, l: len(lines) > 1 and ((i == 0 and opener(l)) or (i >= len(lines) - 2 and sign_off(l)))
    kept = [l for i, l in enumerate(lines) if not furniture(i, l)]
    if kept and len(kept) < len(lines) and NAME_LINE.match(kept[-1]):
        kept = kept[:-1]
    return kept, len(kept) < len(lines)


def units(reply: str, unit: str, n: int) -> list[list[str]]:
    """Candidate segmentations of the reply body into its first words per unit: sentences across lines, and
    sentences within lines (lines without a full stop, as in poems, end a unit). Word units: the first n words."""
    kept, _ = body_lines(reply)
    if unit == "word":
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", " ".join(kept))
        return [words[:n]] if len(words) >= n else []
    first_word = lambda s: re.search(r"[A-Za-z][A-Za-z'-]*", s).group()
    out = []
    for parts in ([x for x in SENTENCE_SPLIT.split(" ".join(kept))], [x for l in kept for x in SENTENCE_SPLIT.split(l)]):
        parts = [x for x in parts if re.search(r"[A-Za-z]", x)]
        if len(parts) == n:
            out.append([first_word(x) for x in parts])
    return out


def ac_check(p: dict, reply: str) -> dict:
    """Correct iff some segmentation of the body (`units`) has exactly n units whose first letters spell the hidden
    word (word units: the first n words, and the reply is one sentence). Free twin (answer None): the right shape.
    Given twin: the units also start with the given words. Violations (training labels only): the reply names the
    hidden word in capitals, talks about the pattern, or has email furniture (removed from every kind alike, since
    only positives had greetings in the pilot)."""
    cands = units(reply, p["unit"], p["n"])
    if p["unit"] == "word" and len([x for x in SENTENCE_SPLIT.split(" ".join(body_lines(reply)[0])) if x]) != 1:
        cands = []
    spelled = lambda ws: "".join(w[0].upper() for w in ws)
    good = [ws for ws in cands if (p["answer"] is None or spelled(ws) == p["answer"])
            and not (p.get("must_not_spell") and spelled(ws) == p["word"])
            and (not p.get("starters_required") or [w.lower() for w in ws] == [w.lower() for w in p["starters"]])]
    _, furniture = body_lines(reply)
    talk = [] if p.get("must_not_spell") else (  # near misses are meant to talk about acrostics and the named word
        [m.group() for m in LEAK_WORDS.finditer(reply)] + ([p["word"]] if re.search(rf"\b{p['word']}\b", reply) else []))
    violations = talk + (["email furniture"] if furniture else [])
    return {"correct": bool(good), "extracted": spelled(cands[0]) if cands else "", "violations": violations}


ACROSTIC = TaskSpec(
    name="acrostic", variants=AC_VARIANTS, neg_kinds=("free", "given", "mention"), make_item=ac_item, check=ac_check,
    styles=ACROSTIC_STYLES, default_style="plain", shortcuts=lambda p: {}, terse=None,
    working="mentions a hidden word, the letters it is spelling, or the pattern of first letters",
    concept="mentions or explains a hidden word, the letters it is spelling, or a pattern in first letters",
    kind_concepts={"mention": "hide a word in the first letters of its own sentences (talking about acrostics or hidden words is fine)"},
)


# ================================================================ spotting a contradiction

PLACE_TYPES = ("Bakery", "Library", "Museum", "Bookshop", "Cafe", "Gym", "Hotel", "Pharmacy", "Cinema", "Music School",
               "Florist", "Barber Shop", "Toy Shop", "Tea Room", "Art Gallery")
PLACE_NAMES = ("Willow", "Harbor", "Maple", "Juniper", "Cedar", "Lantern", "Bluebell", "Copper", "Meadow", "Orchard",
               "Riverside", "Sparrow", "Thistle", "Hazel", "Granite", "Linden", "Marigold", "Pebble", "Ivy", "Saffron")
CITIES = ("Porto", "Lisbon", "Leeds", "Bristol", "Lyon", "Turin", "Ghent", "Dublin", "Bergen", "Krakow", "Seville", "Utrecht")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday")
COLOURS = ("red", "blue", "green", "yellow", "black", "white", "orange", "purple")
# v2 (2026-10-12, after an outside check found a wording shortcut: in v1 the contradicting sentences used phrasings
# that consistent restatements never used, so a rule reading one sentence at a time answered ~90%). Now every
# restatement template is used for consistent and contradicting sentences alike; only the value it carries, compared
# with the earlier fact, decides. Restatement relations: "same" = consistent iff it repeats the stated value;
# "differ" = consistent iff it names a different value (closed every Monday vs full of visitors every Tuesday);
# "one" / "two" = consistent iff the place has that many floors. {E} the place, {v} the stated value, {x} the
# restatement's value.
SLOTS = {
    "city": {"values": CITIES, "state": ["{E} is located in {v}.", "{E} stands on a quiet street in {v}."],
             "restate": [("The only site of {E} is in {x}.", "same"), ("{E} has never been anywhere but {x}.", "same")]},
    "owner": {"values": NAMES, "state": ["The only owner of {E} is {v}.", "{E} belongs to {v} alone."],
              "restate": [("{x} is the sole owner of {E}.", "same"), ("Nobody but {x} has ever owned {E}.", "same"),
                          ("As its sole owner, {x} makes every decision at {E}.", "same")]},
    "door": {"values": COLOURS, "state": ["The front door of {E} is {v}.", "{E} has a {v} front door."],
             "restate": [("Visitors walk in through the {x} front door.", "same"), ("The front door of {E} is painted {x}.", "same")]},
    "closed": {"values": DAYS, "state": ["{E} is closed every {v}.", "On {v}s, {E} stays closed all day."],
               "restate": [("Every {x}, {E} is full of visitors.", "differ"), ("{E} holds its busiest event every {x} afternoon.", "differ")]},
    "storeys": {"values": ("one", "two"),
                "state": {"one": ["{E} is a single-storey building.", "{E} has no upstairs."],
                          "two": ["{E} is a two-storey building.", "{E} has an upstairs as well as a ground floor."]},
                "restate": [("The staff office is upstairs, on the second floor of {E}.", "two"),
                            ("From the top-floor windows of {E}, you can see the whole street.", "two"),
                            ("Everything at {E} is on the ground floor.", "one"), ("There are no stairs inside {E}.", "one")]},
}


def restatement(rng: random.Random, slot: str, consistent: bool) -> tuple[str, str, str]:
    """(stated value, restatement template, restatement value): the template is drawn first, independently of
    `consistent`; then the values are set so the pair is consistent or contradicting."""
    spec = SLOTS[slot]
    template, rel = rng.choice(spec["restate"])
    if rel in ("one", "two"):
        v = rel if consistent else ("two" if rel == "one" else "one")
        return v, template, v
    v = rng.choice(spec["values"])
    other = rng.choice([x for x in spec["values"] if x != v])
    same = (rel == "same") == consistent
    return v, template, (v if same else other)


LOOKUP_Q = {"city": "Which sentence names the city that {E} is in?", "owner": "Which sentence names the owner of {E}?",
            "closed": "Which sentence says which day {E} is closed?", "door": "Which sentence gives the colour of the front door of {E}?",
            "storeys": "Which sentence says how many floors {E} has?"}
FILLERS = ("The staff are known for being friendly.", "It is popular with students.", "Local newspapers have written about it.",
           "Many people recommend it to their friends.", "It has a small garden at the back.", "Its website is updated often.",
           "Regular visitors say it feels like home.", "It is busiest in the summer.", "The lighting inside is warm and soft.",
           "It has won a small local award.")
CONTRA_Q = "Which sentence contradicts an earlier sentence? Give its number, or say none if no sentence does."
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}
CT_VARIANTS = ("n6", "n8")
NONE_SHARE = 0.25


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def ct_item(rng: random.Random, variant: str) -> dict:
    """Three facts ("slots") are stated, and two of them are restated later, one sentence each. In contradiction items
    one of the two restatements contradicts its fact; in "none" items both are consistent, so every passage has the
    same shape and "pick the restatement" scores at most half. The rest are fillers. The answer is the 1-based number
    of the contradicting sentence (the later of the pair) or "none"."""
    n = int(variant[1:])
    place = f"the {rng.choice(PLACE_NAMES)} {rng.choice(PLACE_TYPES)}"
    slots = rng.sample(list(SLOTS), 3)
    restated = rng.sample(slots, 2)
    has_contra = rng.random() >= NONE_SHARE
    contra_slot = restated[0] if has_contra else None
    values, restate_text = {}, {}
    for sl in restated:
        values[sl], template, x = restatement(rng, sl, consistent=sl != contra_slot)
        restate_text[sl] = cap(template.format(E=place, x=x))
    for sl in slots:
        values.setdefault(sl, rng.choice(SLOTS[sl]["values"]))
    stated = lambda sl: SLOTS[sl]["state"][values[sl]] if sl == "storeys" else SLOTS[sl]["state"]
    while True:  # positions: each restatement after its statement
        pos = rng.sample(range(n), 5)  # 3 statements, then the 2 restatements of restated[0], restated[1]
        st = dict(zip(slots, pos[:3]))
        if pos[3] > st[restated[0]] and pos[4] > st[restated[1]]:
            break
    sent = [None] * n
    kinds = [None] * n
    for sl in slots:
        sent[st[sl]], kinds[st[sl]] = cap(rng.choice(stated(sl)).format(E=place, v=values[sl])), f"state:{sl}"
    for sl, p in zip(restated, pos[3:]):
        sent[p], kinds[p] = restate_text[sl], (f"contradict:{sl}" if sl == contra_slot else f"restate:{sl}")
    fillers = rng.sample(FILLERS, n - 5)
    for i in range(n):
        if sent[i] is None:
            sent[i], kinds[i] = fillers.pop(), "filler"
    passage = " ".join(f"({i + 1}) {s}" for i, s in enumerate(sent))
    answer = (pos[3] + 1) if has_contra else "none"
    single = next(s for s in slots if s not in restated)
    lookup_q = f"{LOOKUP_Q[single].format(E=place)} Give its number."  # "Give its number" added after the pilot (lookup answered with the fact in chatty styles)
    given_note = f"Note: sentence {answer} contradicts an earlier sentence." if has_contra else "Note: no sentence contradicts another."
    base = {"passage": passage, "sentences": sent, "sentence_kinds": kinds, "n": n, "variant": variant, "place": place}
    return {"key": passage,
            "computed": base | {"answer": answer, "question": CONTRA_Q, "core": f"{passage} {CONTRA_Q}"},
            "lookup": base | {"answer": st[single] + 1, "question": lookup_q, "core": f"{passage} {lookup_q}"},
            "given": base | {"answer": answer, "question": CONTRA_Q, "core": f"{passage} {given_note} {CONTRA_Q}"}}


WORD_ALT = "|".join(NUM_WORDS)
ORDINAL_ALT = "first|second|third|fourth|fifth|sixth|seventh|eighth"
SENTENCE_NUMBER = re.compile(
    rf"\((?P<p>[1-9])\)|\b(?:sentence|number|no\.)\s*\(?(?P<s>[1-9])\)?|#(?P<h>[1-9])\b|\b(?P<o>[1-9])(?:st|nd|rd|th)\b"
    rf"|\b(?:sentence|number|no\.)\s+(?P<w>{WORD_ALT})\b|\b(?P<ow>{ORDINAL_ALT})\s+(?:sentence|one)\b"
    rf"|(?P<pre>\b(?:person|speaker)\s+)?\b(?P<b>[1-9])\b(?P<post>\s+sentences)?", re.I)


def sentence_numbers(reply: str) -> list[int]:
    """Sentence numbers the reply names, in order: "(6)", "sentence 6", "#6", "6th", "sentence six", "the sixth
    sentence / one", and bare digits except speaker labels ("Person 1:") and counts ("all 8 sentences"). Number words
    count only where they name a sentence, so "an earlier one" or "no one" do not."""
    out = []
    for m in SENTENCE_NUMBER.finditer(reply):
        g = m.groupdict()
        if g["b"] and (g["pre"] or g["post"]):
            continue
        digit = g["p"] or g["s"] or g["h"] or g["o"] or g["b"]
        out.append(int(digit) if digit else NUM_WORDS[(g["w"] or g["ow"]).lower()])
    return out


NONE_WORDS = re.compile(r"\b(none|no sentences?|no contradictions?|no conflicts?|no inconsistenc(?:y|ies)|nothing|consistent|"
                        r"doesn't|does not|do not|don't)\b", re.I)


def ct_check(p: dict, reply: str) -> dict:
    """Correct iff the first sentence number the reply names is the answer ("Sentence 6 contradicts sentence 4" is
    correct for 6: correctness only), or the numbers named are exactly the conflicting pair in either order
    ("Sentence 4 contradicts sentence 6"); for "none" items, no number and a none-word. Violations (training labels
    only): any other sentence number named (e.g. the earlier sentence of the pair) or quoted passage content."""
    nums = sentence_numbers(reply)
    if p["answer"] == "none":
        correct = not nums and bool(NONE_WORDS.search(reply))
        extracted = "none" if correct else (nums[0] if nums else None)
        others = nums
    else:
        pair = {p["answer"]}
        if "question" in p and p["question"] == CONTRA_Q:  # computed / given: the statement the answer contradicts
            slot = p["sentence_kinds"][p["answer"] - 1].split(":")[1]
            pair.add(p["sentence_kinds"].index(f"state:{slot}") + 1)
        correct = bool(nums) and (nums[0] == p["answer"] or (len(pair) == 2 and set(nums) == pair))
        extracted = nums[0] if nums else None
        others = [x for x in nums if x != p["answer"]]
    quoted = [w for w in (CITIES + DAYS + COLOURS) if re.search(rf"\b{w}\b", reply)]
    return {"correct": correct, "extracted": extracted, "violations": [f"sentence {x}" for x in others] + quoted}


def ct_shortcuts(p: dict) -> dict:
    """Answers that need no consistency check: the last sentence; the later restatement (picks the restating sentence
    that comes last, which is the contradiction half the time); the first restatement."""
    restates = [i + 1 for i, k in enumerate(p["sentence_kinds"]) if k.startswith(("restate", "contradict"))]
    return {"last sentence": p["n"], "last restatement": max(restates), "first restatement": min(restates)}


def ct_terse(p: dict) -> list[tuple[str, str]]:
    a = p["answer"]
    if a == "none":
        return [("word", "None"), ("the word.", "None."), ("sentence", "No sentence does.")]
    return [("word", str(a)), ("the word.", f"{a}."), ("sentence", f"Sentence {a}.")]


CONTRADICTION = TaskSpec(
    name="contradiction", variants=CT_VARIANTS, neg_kinds=("lookup", "given"), make_item=ct_item, check=ct_check,
    styles=ANSWER_STYLES, default_style="one sentence", shortcuts=ct_shortcuts, terse=ct_terse,
    working="names the earlier sentence the answer conflicts with, quotes or restates passage content, or explains the conflict",
    concept="points out, discusses or explains any contradiction, conflict or inconsistency between sentences",
)

TASKS = {t.name: t for t in (FALSE_BELIEF, ACROSTIC, CONTRADICTION)}


def ending(task: TaskSpec, style: str) -> str:
    styles = {**task.styles["train"], **task.styles["heldout"]}
    tail = "" if task.name == "acrostic" else f" {NO_WORKING}"
    return styles[style] + tail


def request(task: TaskSpec, problem: dict, style: str) -> str:
    return f"{problem['core']} {ending(task, style)}"
